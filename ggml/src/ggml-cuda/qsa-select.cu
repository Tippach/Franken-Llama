#include "qsa-select.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// strixllama SLICE 3: fused QSA indexer-select kernel.
//
// Replaces the I13+I14 chain (qwen4exp_apply_compact_visibility -> the TOP_K of
// qwen4exp_select_complete_blocks) with ONE kernel. The cell expansion (argsort -> get_rows ->
// concat tail) stays in the already-fused qsa_expand kernel downstream, which reads the two
// buffers this kernel produces: masked_score (the visibility ADD output) and blocks (the TOP_K
// output). So this kernel only needs score/starts/tails as inputs.
//
// Per query row (one workgroup):
//   1. eff[b] = (tails[q] > starts[b]) ? score[b] : -inf   (bit-exact fused mask, SPEC A.3)
//      -> written to masked_score (the ADD output). REPEAT/SUB/STEP/LOG never hit HBM.
//   2. top-`budget` select of eff[b], tie-break ASCENDING block index (matches
//      ggml-cuda/top-k.cu:411-442 top_k_gather_equal; the parallel radix path the chain uses
//      produces the same SET, and qsa_expand re-sorts by score + qsa3 unions to a set).
//      -> written to blocks (the TOP_K output).
//
// Only the compact single-seq path is fused (seq_blk == nullptr). The matcher rejects anything
// whose backward chain from ADD is not exactly ADD(score, LOG(STEP(SUB(REPEAT(tails), starts)))).

#define QSEL_NEG_INF (-INFINITY)

// Same ordered-uint mapping as ggml-cuda/top-k.cu:57 so the radix threshold is bit-identical.
static __device__ __forceinline__ uint32_t qsel_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

// One block per query row. NO dynamic shared (SLICE 5b): the fused masked row lives in the
// `masked` global output (L2-warm) and the radix passes read it back, so the kernel launches at
// any n_blocks. (The old design put the eff row in dynamic LDS = n_blocks*4 bytes, which overflowed
// the 64 KB gfx1151 limit at 131k n_blocks=32256 -> 129 KB -> HANG. The kernel comment had
// ASSUMED LLAMA_QSA_SCORE_BOUNDS capped n_blocks to ~4608; it does not.) Static shared only:
// 256-bin radix histogram, selection array (budget ints), ballot counts, counters.
template <int BLOCK_SIZE>
static __global__ __launch_bounds__(BLOCK_SIZE) void qsa_select_kernel(
        const float * __restrict__ score,    // [n_blocks, n_query] f32, contiguous (raw, pre-mask)
        const int   * __restrict__ limits,   // i32 input: starts at [starts_off], tails at [tails_off]
        float       * __restrict__ masked,   // [n_blocks, n_query] f32 OUT (= visibility ADD output)
        int         * __restrict__ blocks,   // [budget, n_query] i32 OUT (= TOP_K output)
        const int64_t starts_off,
        const int64_t tails_off,
        const int n_blocks,
        const int budget) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_bucket;
    __shared__ uint32_t s_above;
    __shared__ int sel[2048];                   // final selection, budget <= 2048 (matcher-enforced)
    __shared__ int warp_counts[32];
    __shared__ int out_count;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = BLOCK_SIZE / 32;
    const int row = blockIdx.x;

    // ---- fused visibility into shared + masked_score ----
    // The chain casts limits to f32 then compares; positions are non-negative ints < 2^24 so the
    // f32 cast is exact and (int)a > (int)b == (float)a > (float)b. Reading i32 directly is
    // bit-identical to the SUB/STEP/LOG/ADD chain (SPEC A.3).
    const int tail = __ldg(&limits[tails_off + row]);
    const float * score_row = score + (size_t) row * n_blocks;
    float * masked_row = masked + (size_t) row * n_blocks;
    for (int b = tid; b < n_blocks; b += BLOCK_SIZE) {
        const float e = (tail > __ldg(&limits[starts_off + b])) ? __ldg(&score_row[b]) : QSEL_NEG_INF;
        masked_row[b] = e;
    }
    __syncthreads();   // masked_row (global) now visible to the whole block; radix reads it back below

    // ---- radix select the `budget`-th largest ordered value -> threshold ----
    // Mirrors top_k_radix_select_cuda (top-k.cu:444-517): bucket/above go through shared so every
    // thread advances prefix/desired identically.
    uint32_t prefix = 0;
    uint32_t desired = (uint32_t) budget;
#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int bin = tid; bin < 256; bin += BLOCK_SIZE) hist[bin] = 0;
        __syncthreads();

        const uint32_t high_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        const uint32_t prefix_high = prefix & high_mask;
        for (int b = tid; b < n_blocks; b += BLOCK_SIZE) {
            const uint32_t key = qsel_ordered(masked_row[b]);
            if ((key & high_mask) == prefix_high) {
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            uint32_t above = 0;
            uint32_t bucket = 0;
            for (int bin = 255; bin >= 0; --bin) {
                const uint32_t count = hist[bin];
                if (above + count >= desired) { bucket = (uint32_t) bin; break; }
                above += count;
            }
            s_bucket = bucket;
            s_above = above;
        }
        __syncthreads();
        prefix |= s_bucket << shift;
        desired -= s_above;
        __syncthreads();
    }
    const uint32_t threshold = prefix;

    // ---- Phase A: emit all strictly-greater-than-threshold (any order; set only) ----
    if (tid == 0) out_count = 0;
    __syncthreads();
    for (int b = tid; b < n_blocks; b += BLOCK_SIZE) {
        if (qsel_ordered(masked_row[b]) > threshold) {
            const int pos = atomicAdd(&out_count, 1);
            sel[pos] = b;
        }
    }
    __syncthreads();                 // out_count == n_gt now, all > written

    // ---- Phase B: == threshold, ASCENDING block index, fill sel[gt .. budget-1] ----
    // top_k_gather_equal pattern (top-k.cu:411-442): offset = gt, limit = budget.
    int count = out_count;
    for (int base = 0; base < n_blocks && count < budget; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool equal = (col < n_blocks) && (qsel_ordered(masked_row[col]) == threshold);
        const unsigned long long mask = __ballot(equal);
        if (lane == 0) warp_counts[warp] = __popcll(mask);
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) before += warp_counts[w];
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1ULL;
        const int pos = before + __popcll(mask & lane_mask);
        if (equal && pos < budget) {
            sel[pos] = col;
        }
        __syncthreads();
    }

    // ---- write the selected block indices (TOP_K output) ----
    int * blocks_row = blocks + (size_t) row * budget;
    for (int k = tid; k < budget; k += BLOCK_SIZE) {
        blocks_row[k] = sel[k];
    }
}

bool ggml_cuda_qsa_select_enabled() {
    static const int v = getenv("LLAMA_QSA_SELECT_FUSE") ? atoi(getenv("LLAMA_QSA_SELECT_FUSE")) : 0;
    return v != 0;
}

// True if `t` is referenced (as a src) by no node outside the span [lo, hi]. A skipped span node
// must be consumed only inside the span, or skipping it would starve an outside consumer. (add/topk
// are exempt: the fused kernel still writes their buffers, so their outside readers stay correct.)
static bool qsel_consumed_within(const ggml_cgraph * g, const ggml_tensor * t, int lo, int hi) {
    for (int k = 0; k < g->n_nodes; ++k) {
        if (k >= lo && k <= hi) continue;
        const ggml_tensor * n = g->nodes[k];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t) return false;
        }
    }
    return true;
}

// Walk a f32 tensor (a starts/tails view of the i32 `limits` input, possibly through a CAST and a
// RESHAPE) down to the i32 base tensor, accumulating the byte offset across every view. A CAST is a
// CPY with no view_src, so we follow src[0]; a VIEW/RESHAPE carries view_src + view_offs. Returns the
// base (expected i32) or null if the chain does not terminate in a single non-view tensor.
static const ggml_tensor * qsel_walk_to_i32(const ggml_tensor * t, int64_t & bytes) {
    int guard = 0;
    while (t && guard++ < 12) {
        if (t->view_src) { bytes += (int64_t) t->view_offs; t = t->view_src; continue; }
        if (t->op == GGML_OP_CPY && t->src[0]) { t = t->src[0]; continue; }
        break;  // reached the base (no view_src, not a cast)
    }
    return t;
}

// Matcher, anchored at the REPEAT of the compact single-seq visibility chain. The chain is built
// by qwen4exp_apply_compact_visibility + the TOP_K of qwen4exp_select_complete_blocks:
//   REPEAT(tails,score) -> SUB(rep, starts) -> STEP(sub) -> LOG(step) -> ADD(score, log) -> TOP_K(add)
// We capture score and the raw i32 `limits` (starts/tails are CAST/VIEW of it) and write
// masked=ADD, blocks=TOP_K. The starts CAST/VIEW sit inside the span, so we read limits directly
// with element offsets instead of relying on the (skipped) f32 cast buffers. Multi-seq inserts a MUL
// between STEP and LOG and the non-compact path ADDs a bias (not a LOG), so both fail the edges.
int ggml_cuda_match_qsa_select(const ggml_cgraph * g, int i, ggml_cuda_qsa_select_args & a) {
    if (!ggml_cuda_qsa_select_enabled() || i + 5 >= g->n_nodes) return 0;
    const ggml_tensor * rep = g->nodes[i];
    if (rep->op != GGML_OP_REPEAT) return 0;
    const bool dbg = getenv("LLAMA_QSA_SELECT_DEBUG") != nullptr;
    static unsigned nrej = 0;
    auto reject = [&](const char * why) {
        if (dbg && nrej++ < 40) {
            fprintf(stderr, "QSA_SELECT_REJECT i=%d at=%s window:", i, why);
            for (int k = i; k <= i + 10 && k < g->n_nodes; ++k) fprintf(stderr, " %s", ggml_op_name(g->nodes[k]->op));
            fprintf(stderr, "\n");
        }
        return 0;
    };
    const ggml_tensor * tails = rep->src[0];
    if (!tails) return reject("no tails");
    // forward-scan for the TOP_K that closes the chain (within a small window)
    int j = -1;
    for (int k = i + 1; k <= i + 10 && k < g->n_nodes; ++k) {
        if (g->nodes[k]->op == GGML_OP_TOP_K) { j = k; break; }
    }
    if (j < 0) return reject("no topk in window");
    const ggml_tensor * topk = g->nodes[j];
    const ggml_tensor * add  = topk->src[0];
    if (!add || add->op != GGML_OP_ADD) return reject("topk src not ADD");
    const ggml_tensor * lg = add->src[1];
    if (!lg || lg->op != GGML_OP_LOG) return reject("add src[1] not LOG");
    const ggml_tensor * step = lg->src[0];
    if (!step || step->op != GGML_OP_UNARY || ggml_get_unary_op(step) != GGML_UNARY_OP_STEP) return reject("log src not STEP");
    const ggml_tensor * sub = step->src[0];
    if (!sub || sub->op != GGML_OP_SUB) return reject("step src not SUB");
    if (sub->src[0] != rep) return reject("sub src[0] not rep");
    const ggml_tensor * starts = sub->src[1];
    if (!starts) return reject("no starts");
    const ggml_tensor * score = add->src[0];
    if (!score) return reject("no score");
    // NB: ggml_repeat stores only src[0]=tails (the shape comes from the second arg, not a src),
    // so we cannot cross-check score against rep->src[1]. The ADD(score, LOG) edge is authoritative.
    // f32, contiguous, single stream
    if (score->type != GGML_TYPE_F32 || !ggml_is_contiguous(score)) return reject("score not f32 contig");
    if (add->type  != GGML_TYPE_F32 || !ggml_is_contiguous(add)) return reject("add not f32 contig");
    if (topk->type != GGML_TYPE_I32 || !ggml_is_contiguous(topk)) return reject("topk not i32 contig");
    if (starts->type != GGML_TYPE_F32 || tails->type != GGML_TYPE_F32) return reject("starts/tails not f32");
    if (score->ne[2] != 1 || score->ne[3] != 1) return reject("score not single stream");
    const int64_t n_blocks = score->ne[0];
    const int64_t n_query  = score->ne[1];
    const int64_t budget   = topk->ne[0];
    if (n_blocks < 512 || n_blocks > 65536) return reject("n_blocks range");
    if (n_query < 1 || n_query > 65536) return reject("n_query range");
    if (budget < 1 || budget > 2048) return reject("budget range");
    if (!ggml_are_same_shape(score, add)) return reject("score/add shape");
    // starts is a per-block row (ne[0]==n_blocks); tails broadcasts per-query (ne[0]==1)
    if (starts->ne[0] != n_blocks) return reject("starts ne0");
    if (tails->ne[0] != 1) return reject("tails ne0");
    // resolve both bounds to the same i32 `limits` input
    int64_t sb = 0, tb = 0;
    const ggml_tensor * lim_s = qsel_walk_to_i32(starts, sb);
    const ggml_tensor * lim_t = qsel_walk_to_i32(tails, tb);
    if (!lim_s || lim_s->type != GGML_TYPE_I32) return reject("starts not i32-backed");
    if (lim_t != lim_s) return reject("tails/starts different base");
    if (sb % 4 != 0 || tb % 4 != 0) return reject("offsets not 4-aligned");
    // every node strictly between the anchor REPEAT and TOP_K must be a span-only CAST or VIEW
    // (the starts/tails casts). Any other compute node in the span => reject.
    for (int k = i + 1; k < j; ++k) {
        const ggml_tensor * n = g->nodes[k];
        if (n == sub || n == step || n == lg || n == add) continue;
        if (n->op != GGML_OP_CPY && n->op != GGML_OP_VIEW && n->op != GGML_OP_RESHAPE) return reject("span has non-cast node");
        if (!qsel_consumed_within(g, n, i, j)) return reject("span cast has outside consumer");
    }
    // the intermediate chain nodes we skip (rep/sub/step/log) must be consumed only inside the span;
    // add/topk are exempt because the fused kernel still writes their buffers for downstream readers.
    if (!qsel_consumed_within(g, rep, i, j)) return reject("rep outside consumer");
    if (!qsel_consumed_within(g, sub, i, j)) return reject("sub outside consumer");
    if (!qsel_consumed_within(g, step, i, j)) return reject("step outside consumer");
    if (!qsel_consumed_within(g, lg, i, j)) return reject("log outside consumer");
    a.score  = score;
    a.limits = lim_s;
    a.starts_off = sb / 4;
    a.tails_off  = tb / 4;
    a.masked = (ggml_tensor *) add;
    a.blocks = (ggml_tensor *) topk;
    a.n_blocks = (int) n_blocks;
    a.n_query  = (int) n_query;
    a.budget   = (int) budget;
    if (getenv("LLAMA_QSA_SELECT_DEBUG")) {
        static unsigned nmatch = 0;
        if (nmatch++ < 3) {
            fprintf(stderr, "QSA_SELECT_MATCH anchor=%d topk=%d span=%d nb=%lld nq=%lld budget=%lld\n",
                    i, j, j - i, (long long) n_blocks, (long long) n_query, (long long) budget);
            fprintf(stderr, "  starts_off=%lld tails_off=%lld limN=%lld masked_ne=[%lld,%lld] blocks_ne=[%lld,%lld]\n",
                    (long long) (sb / 4), (long long) (tb / 4), (long long) lim_s->ne[0],
                    (long long)add->ne[0], (long long)add->ne[1],
                    (long long)topk->ne[0], (long long)topk->ne[1]);
        }
    }
    return j - i;  // consume REPEAT..TOP_K inclusive
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_select_args & args) {
    const int n_blocks = args.n_blocks;
    const int n_query  = args.n_query;
    constexpr int BLOCK_SIZE = 512;

    // No dynamic shared: the fused row lives in `masked` (global, L2-warm), so this launches at
    // any n_blocks (SLICE 5b — the old eff_bytes=n_blocks*4 overflowed the 64KB LDS at 131k).
    qsa_select_kernel<BLOCK_SIZE><<<n_query, BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) args.score->data,
        (const int *)   args.limits->data,
        (float *) args.masked->data,
        (int *)   args.blocks->data,
        args.starts_off, args.tails_off,
        n_blocks, args.budget);
    CUDA_CHECK(cudaGetLastError());

    static unsigned hits = 0;
    static int hi_max = 0;
    if (getenv("LLAMA_QSA_SELECT_DEBUG")) {
        ++hits;
        if (n_blocks > hi_max) hi_max = n_blocks;
        // Print every fire in the LDS-overflow crossover region (>=16000 blocks), plus the
        // first 24 overall. The old cap hid the high-n_blocks ubatches that overflowed LDS.
        if (n_blocks >= 16000 || hits <= 24) {
            fprintf(stderr, "QSA_SELECT_FUSE fired #%u: n_blocks=%d n_query=%d budget=%d hi_max=%d\n",
                    hits, n_blocks, n_query, args.budget, hi_max);
        }
    }
}
