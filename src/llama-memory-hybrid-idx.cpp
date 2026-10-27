#include "llama-memory-hybrid-idx.h"

#include <cstdlib>
#include <limits>

#include "prefix.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>
#include <type_traits>

//
// llama_memory_hybrid_idx
//

// A qwen4exp MTP context is built with a recurrent filter that matches nothing (the nextn layer is not recurrent) so
// it can carry an indexer cache for sparse draft attention. Skip the recurrent child there: an empty recurrent cache
// still refuses partial seq_rm, which aborts the server on the first cache trim.
static bool hybrid_idx_no_recr(const llama_memory_recurrent * r) {
    if (!r) { return true; }
    for (ggml_tensor * t : r->r_l) { if (t) { return false; } }
    for (ggml_tensor * t : r->s_l) { if (t) { return false; } }
    return true;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp upstream: key | pooled; our qwen4exp has no k-pool)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // strixllama: the indexer keys stay f16 whatever the attention cache is - they are pooled into the block keys
        // the selection is scored on, and they are ~1/8 of the attention cache
        return new llama_kv_cache(
            model, hparams_idx, GGML_TYPE_F16, GGML_TYPE_F16, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // strixllama: block-key cache, one F16 [idx_dim, kv_size + 1] tensor per indexer layer, allocated next to
    // that layer's indexer keys (LLAMA_QSA_BLOCK_KEY_CACHE=0 turns it off and restores the full rebuild)
    const char * kb_env = getenv("LLAMA_QSA_BLOCK_KEY_CACHE");
    if (mem_idx && (kb_env == nullptr || atoi(kb_env) != 0)) {
        const uint32_t kv_size = mem_idx->get_size();
        const int64_t  idx_dim = model.hparams.indexer_head_size;
        const uint32_t n_layer = model.hparams.n_layer_all;

        // the rows' residue classes follow the block ratio of the indexer layers (one ratio for all of them, as the
        // graph inputs assume); anything else keeps a row per cell in order
        uint32_t ratio = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (filter_idx(il)) {
                const uint32_t ril = model.hparams.dsv4_compress_ratios[il];
                ratio = ratio == 0 || ratio == ril ? ril : UINT32_MAX;
            }
        }
        kb_r = ratio >= 1 && ratio <= 64 ? ratio : 1;
        kb_s = (kv_size + 1 + kb_r - 1)/kb_r;

        std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!filter_idx(il)) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (offload) {
                buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
            }
            ggml_context * ctx = nullptr;
            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = { size_t(2u*n_layer*ggml_tensor_overhead()), nullptr, true };
                ctx = ggml_init(params);
                if (ctx == nullptr) {
                    throw std::runtime_error("failed to create ggml context for the QSA block-key cache");
                }
                ctx_map[buft] = ctx;
                kb_ctxs.emplace_back(ctx);
            } else {
                ctx = it->second;
            }
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, idx_dim, (int64_t) kb_r*kb_s);
            ggml_format_name(t, "cache_idx_kb_l%d", il);
            kb_map[(int32_t) il] = t;
        }
        size_t bytes = 0;
        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error("failed to allocate the QSA block-key cache");
            }
            ggml_backend_buffer_clear(buf, 0);
            bytes += ggml_backend_buffer_get_size(buf);
            kb_bufs.emplace_back(buf);
        }
        LLAMA_LOG_INFO("%s: QSA block-key cache: %zu layers, %.1f MiB, rows by residue mod %u\n", __func__, kb_map.size(),
                bytes/1024.0/1024.0, kb_r);
    }

    // strixllama: IndexShare rows for the MTP layer's selection (LLAMA_MTP_INDEX_SHARE=1), when that layer has an indexer
    {
        const char * env = getenv("LLAMA_MTP_INDEX_SHARE");
        const uint32_t il = model.hparams.n_layer();
        const int64_t  r  = il < model.hparams.n_layer_all ? (int64_t) model.hparams.dsv4_compress_ratios[il] : 0;
        if (env && atoi(env) != 0 && mem_idx && model.hparams.n_layer_nextn > 0 && il < model.hparams.n_layer_all &&
                filter_idx(il) && r > 0 && model.hparams.indexer_top_k > 0) {
            share_w     = (int64_t) model.hparams.indexer_top_k + r - 1;
            share_n_seq = n_seq_max;
            ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("failed to create ggml context for the IndexShare rows");
            }
            share_ctx.reset(ctx);
            share_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, share_w, (int64_t) n_seq_max*share_ring + 2);
            ggml_format_name(share_t, "cache_idx_share_l%u", il);
            ggml_backend_buffer_type_t buft = offload ? ggml_backend_dev_buffer_type(model.dev_layer(il)) : ggml_backend_cpu_buffer_type();
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (buf == nullptr) {
                throw std::runtime_error("failed to allocate the IndexShare rows");
            }
            ggml_backend_buffer_clear(buf, 0);   // 0 = no cell, so the all-zero row selects nothing
            share_buf.reset(buf);
            share_pos.assign((size_t) n_seq_max*share_ring, -1);
            LLAMA_LOG_INFO("%s: IndexShare: %u sequences x %u positions, %lld cells a selection\n", __func__,
                    n_seq_max, share_ring, (long long) share_w);
        }
    }

    // strixllama: regions (llama_kv_cache::set_regions) - with several sequences in one unified pool, each
    // conversation keeps one run of cells and a batch's graph views only its own. The indexer mirrors the
    // attention cells, so both caches take the window, and the moves, together. LLAMA_KV_REGIONS=0 turns it off.
    {
        const char * env = getenv("LLAMA_KV_REGIONS");
        const bool on = unified && n_seq_max > 1 && (env == nullptr || atoi(env) != 0);
        get_mem_attn()->set_regions(on);
        if (mem_idx) {
            mem_idx->set_regions(on);
            if (get_mem_attn()->get_regions() != mem_idx->get_regions()) {
                get_mem_attn()->set_regions(false);
                mem_idx->set_regions(false);
            }
        }
        if (get_mem_attn()->get_regions()) {
            // every move of the attention cells - a rebalance for a batch or for a restore - moves the indexer's
            // cells and the block keys with them, in the same step
            get_mem_attn()->set_move_hook([this](const llama_kv_cache::cell_move_vec_t & moves, bool sync) {
                if (mem_idx) {
                    GGML_ASSERT(mem_idx->can_move());
                    mem_idx->move_cells(moves, sync);
                }
                kb_move_rows(moves);
            });
            fprintf(stderr, "kv regions: on, one run of cells per conversation, %u sequences, %u cells\n", n_seq_max, kv_size);
        }
    }
}

ggml_tensor * llama_memory_hybrid_idx::get_kb(int32_t il) const {
    const auto it = kb_map.find(il);
    return it == kb_map.end() ? nullptr : it->second;
}

uint32_t llama_memory_hybrid_idx::kb_scratch_row() const {
    return mem_idx ? kb_row(mem_idx->get_size()) : 0;
}

bool llama_memory_hybrid_idx::kb_needs_full(const llama_ubatch & ubatch) const {
    if (!ubatch.seq_id || !ubatch.n_seq_id) {
        return false;
    }
    // strixllama: STRIXLLAMA_KB_NOFULL=1 forces the incremental branch always, so the full block-key
    // rebuild graph is never captured. TIMING EXPERIMENT ONLY - block keys go stale, so selection and
    // text are wrong; it exists to price the rebuild, which no node ablation can reach (the chain is
    // swallowed by a fusion at capture, so SKIP_OPS never sees a node to skip: CONT@128/ADD@128 give
    // SKIP_HIT=0 while the byte census shows 48 CONT + 36 ADD at ne=[128,n_blocks] in the replayed
    // 7936-node decode graph). If the depth slope collapses with this on, the per-token rebuild of all
    // n_blocks keys is the slope, and the fix is Bubo's: pool only the blocks a step completes.
    static const bool kb_nofull = getenv("STRIXLLAMA_KB_NOFULL") && atoi(getenv("STRIXLLAMA_KB_NOFULL"));
    if (kb_nofull) {
        return false;
    }
    // strixllama: STRIXLLAMA_KB_DBG=1 names WHY a graph takes the full-rebuild branch. The decode
    // graph was measured (STRIX_SKIP_DUMP=3 byte census) to contain build_block_keys at
    // nb_out = n_blocks - 48 CONT + 36 ADD at ne=[128,24320] at 97k - i.e. every token rebuilds
    // every block key, which is O(depth) per token and the whole depth slope. Bubo pools only the
    // blocks a step completes (PoolIndexerBlocks) and never re-materialises the array. This prints
    // the trigger (stale bit vs pending overflow) so the fix targets the real cause.
    static const bool kb_dbg = getenv("STRIXLLAMA_KB_DBG") && atoi(getenv("STRIXLLAMA_KB_DBG"));
    constexpr llama_pos none = std::numeric_limits<llama_pos>::max();
    bool pending = false;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s < 0 || s >= LLAMA_MAX_SEQ) {
                continue;
            }
            if (kb_stale.test(s)) {
                if (kb_dbg) { fprintf(stderr, "KB_FULL: reason=stale seq=%d n_tok=%u\n", (int) s, ubatch.n_tokens); }
                return true;
            }
            pending |= kb_from[s] != none;
        }
    }
    if (!pending) {
        return false;
    }
    // strixllama: a pending span longer than the dirty list takes on top of the ubatch's own blocks
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> own;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ && kb_from[s] != none) {
                auto & r = own.try_emplace(s, none, -1).first->second;
                r.first  = std::min(r.first,  ubatch.pos[i]);
                r.second = std::max(r.second, ubatch.pos[i]);
            }
        }
    }
    for (const auto & [s, r] : own) {
        if (kb_pending_extra(s, r.first, r.second) > kb_pending_max) {
            if (kb_dbg) { fprintf(stderr, "KB_FULL: reason=pending seq=%d extra=%lld n_tok=%u\n",
                                (int) s, (long long) kb_pending_extra(s, r.first, r.second), ubatch.n_tokens); }
            return true;
        }
    }
    return false;
}

llama_pos llama_memory_hybrid_idx::kb_pending_extra(llama_seq_id seq_id, llama_pos pmin, llama_pos pmax) const {
    const llama_pos from = kb_from[seq_id];
    const llama_pos hi   = get_mem_attn()->seq_pos_max(seq_id);
    if (from == std::numeric_limits<llama_pos>::max() || hi < from) {
        return 0;
    }
    if (pmax < pmin) {
        return hi - from + 1;
    }
    const llama_pos below = std::max<llama_pos>(0, std::min(hi, pmin - 1) - from + 1);
    const llama_pos above = std::max<llama_pos>(0, hi - std::max(from, pmax + 1) + 1);
    return below + above;
}

void llama_memory_hybrid_idx::kb_note_unrefreshed(const llama_ubatch & ubatch) const {
    if (kb_map.empty() || !ubatch.seq_id || !ubatch.n_seq_id) {
        return;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ) {
                kb_from[s] = std::min(kb_from[s], ubatch.pos[i]);
            }
        }
    }
}

void llama_memory_hybrid_idx::kb_mark_full(const llama_ubatch & ubatch) const {
    if (!ubatch.seq_id || !ubatch.n_seq_id) {
        return;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ) {
                kb_stale.reset(s);
                kb_from[s] = std::numeric_limits<llama_pos>::max();   // a full rebuild refreshes the pending blocks too
            }
        }
    }
}

void llama_memory_hybrid_idx::kb_mark_stale(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ) {
        kb_stale.set();
    } else {
        kb_stale.set(seq_id);
    }
}

bool llama_memory_hybrid_idx::kb_pos_dup() const {
    return kb_dup;
}

bool llama_memory_hybrid_idx::kb_seq_unusable(const llama_ubatch & ubatch) const {
    if (kb_seq_off.none()) {
        return false;
    }
    if (!ubatch.seq_id || !ubatch.n_seq_id) {
        return false;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            if (s >= 0 && s < LLAMA_MAX_SEQ && kb_seq_off.test(s)) {
                return true;
            }
        }
    }
    return false;
}

int32_t llama_memory_hybrid_idx::share_row(llama_seq_id seq_id, llama_pos pos) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < share_n_seq && pos >= 0);
    return (int32_t) (seq_id*share_ring + (uint32_t) pos % share_ring);
}

int32_t llama_memory_hybrid_idx::share_scratch_row() const {
    return (int32_t) (share_n_seq*share_ring);
}

int32_t llama_memory_hybrid_idx::share_none_row() const {
    return (int32_t) (share_n_seq*share_ring + 1);
}

void llama_memory_hybrid_idx::share_note(llama_seq_id seq_id, llama_pos pos) const {
    if (share_t && seq_id >= 0 && (uint32_t) seq_id < share_n_seq && pos >= 0) {
        share_pos[share_row(seq_id, pos)] = pos;
    }
}

bool llama_memory_hybrid_idx::share_ready(llama_seq_id seq_id, llama_pos pos) const {
    return share_t && seq_id >= 0 && (uint32_t) seq_id < share_n_seq && pos >= 0 && share_pos[share_row(seq_id, pos)] == pos;
}

llama_pos llama_memory_hybrid_idx::share_latest(llama_seq_id seq_id, llama_pos below) const {
    if (!share_t || seq_id < 0 || (uint32_t) seq_id >= share_n_seq) {
        return -1;
    }
    llama_pos best = -1;
    for (uint32_t k = 0; k < share_ring; ++k) {
        const llama_pos p = share_pos[seq_id*share_ring + k];
        if (p >= 0 && p < below && p > best) {
            best = p;
        }
    }
    return best;
}

void llama_memory_hybrid_idx::share_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<int64_t> & out) const {
    out.clear();
    const auto & sp = get_mem_attn()->get_cells(seq_id).seq_pos_cells(seq_id);
    for (auto it = sp.lower_bound({ p0, 0 }); it != sp.end() && it->first <= p1; ++it) {
        out.push_back((int64_t) it->second);
    }
}

bool llama_memory_hybrid_idx::share_capture_due(const llama_ubatch & ubatch, llama_pos refresh) const {
    if (!share_t || !ubatch.seq_id || !ubatch.n_seq_id) {
        return false;
    }
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> own;   // per sequence: its first and last position here
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id s = ubatch.seq_id[i][0];
        auto & r = own.try_emplace(s, ubatch.pos[i], ubatch.pos[i]).first->second;
        r.first  = std::min(r.first,  ubatch.pos[i]);
        r.second = std::max(r.second, ubatch.pos[i]);
    }
    for (const auto & [s, r] : own) {
        const llama_pos c = share_latest(s, r.first);
        if (c < 0 || r.second - c > refresh) {
            return true;
        }
    }
    return false;
}

void llama_memory_hybrid_idx::share_forget(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (!share_t) {
        return;
    }
    const llama_pos lo = p0 < 0 ? 0 : p0;
    const llama_pos hi = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    for (uint32_t s = 0; s < share_n_seq; ++s) {
        if (seq_id >= 0 && (uint32_t) seq_id != s) {
            continue;
        }
        for (uint32_t k = 0; k < share_ring; ++k) {
            llama_pos & p = share_pos[s*share_ring + k];
            if (p >= lo && p < hi) {
                p = -1;
            }
        }
    }
}

// A block's key sits at the row of its first cell (kb_row). The moves lay a sequence out in position order, so its keys
// can follow their cells only if its cells were in that order already (then every block keeps its first cell);
// a sequence whose cells were not has its keys rebuilt once instead.
void llama_memory_hybrid_idx::kb_move_rows(const llama_kv_cache::cell_move_vec_t & moves) {
    llama_kv_cache::cell_move_vec_t rows;
    for (const auto & m : moves) {
        share_forget(m.seq);   // strixllama: the IndexShare rows name the cells before the move
        if (m.ordered) {
            // the cells src + i, i = j, j + kb_r, ... sit in consecutive rows from kb_row(src + j), and so do their
            // destinations: a run per residue class. The cell runs do not overlap, so neither do these
            for (uint32_t j = 0; j < kb_r && j < m.n; ++j) {
                llama_kv_cache::cell_move rm = m;
                rm.src = kb_row(m.src + j);
                rm.dst = kb_row(m.dst + j);
                rm.n   = (m.n - j + kb_r - 1)/kb_r;
                rows.push_back(rm);
            }
        } else {
            kb_mark_stale(m.seq);
        }
    }
    for (const auto & [il, t] : kb_map) {
        get_mem_attn()->copy_rows(t, rows);
    }
}

// strixllama: does this ubatch carry a position per axis, i.e. an image under M-RoPE? A text token has
// the same value on every axis (the whole batch is then "position scalar"), an image token does not,
// and the cells it writes repeat one position across the image. Mirrors qwen4exp_pos_scalar().
static bool hybrid_idx_ubatch_pos_dup(const llama_ubatch & ubatch) {
    if (!ubatch.pos || ubatch.n_tokens == 0) {
        return false;
    }
    // a text token comes as [p, p, p, 0] since llama_batch_ext (the fourth section is unused here): three decide
    for (uint32_t axis = 1; axis < std::min<uint32_t>(ubatch.n_pos, 3); ++axis) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) {
                return true;
            }
        }
    }
    return false;
}

// strixllama: the cells the dense inputs of this batch span. Under regions the graph views only the run of cells that
// holds the batch's own conversations (llama_kv_cache::get_kv_window), so that run plus the new tokens - not the pool up
// to its last used cell - is what the bound below has to cover. Sizing it by the pool made one conversation's ubatch
// split, and with it its greedy output, depend on what the other slots held and had held.
static uint64_t hybrid_idx_dense_n_kv(const llama_kv_cache & kv, const llama_batch & batch) {
    const auto & cells = kv.get_cells(0);
    const uint64_t pool = (uint64_t) cells.used_max_p1() + batch.n_tokens;

    static const bool window = [] {
        const char * e = getenv("LLAMA_KV_WINDOW");
        return !e || atoi(e) != 0;
    }();
    if (!window || !kv.get_regions()) {
        return pool;
    }

    std::bitset<LLAMA_MAX_SEQ> seen;
    int64_t lo = INT64_MAX;
    int64_t hi = -1;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        for (int32_t k = 0; k < batch.n_seq_id[i]; ++k) {
            const llama_seq_id s = batch.seq_id[i][k];
            if (s < 0 || s >= LLAMA_MAX_SEQ || seen.test(s)) {
                continue;
            }
            seen.set(s);
            const int64_t c0 = cells.seq_cell_min(s);
            if (c0 >= 0) {
                lo = std::min(lo, c0);
                hi = std::max(hi, cells.seq_cell_max(s) + 1);
            }
        }
    }

    // the window is padded to 256 cells (get_kv_window)
    const uint64_t own = (hi > lo ? (uint64_t) (hi - lo) : 0) + batch.n_tokens + 256;
    return std::min(pool, own);
}

// strixllama (issue #13): a text ubatch of one conversation that holds images keeps the compact sparse-attention
// inputs when every image cell of the conversation lies before the ubatch's first position: such a cell is visible to
// every query by its position alone (the M-RoPE rule only orders cells that share the query's position), and
// set_input_qsa ranks the cells and fills the compact metadata in that rank space, as it does for the MTP draft after
// an image. It used to take the dense inputs for the rest of the conversation - a mask and a bias over n_kv x n_tokens
// built on the host, its ubatch held to 2^27 cell-token pairs - and prefill after one image fell from ~1,500 to
// ~200 t/s at 170K. STRIX_IMAGE_TEXT_COMPACT=0: the dense inputs again.
static bool hybrid_idx_image_text_compact() {
    static const bool on = [] {
        const char * e = getenv("STRIX_IMAGE_TEXT_COMPACT");
        return !e || atoi(e) != 0;
    }();
    return on;
}

// whether a cell of `seq` holds an image token (its 2-D position differs from its 1-D one) at or after `pmin`
static bool hybrid_idx_image_from(const llama_kv_cells & cells, llama_seq_id seq, llama_pos pmin, uint32_t off, uint32_t n) {
    for (uint32_t j = off; j < off + n && j < cells.size(); ++j) {
        if (!cells.is_empty(j) && cells.seq_has(j, seq) && cells.pos_get(j) >= pmin &&
                cells.ext_get(j).is_2d_gt(cells.pos_get(j), cells.pos_get(j))) {
            return true;
        }
    }
    return false;
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // strixllama: once a conversation holds an image (or a position gap, kb_dup), its ubatches take the dense
    // sparse-attention inputs - a KQ mask and a per-block bias over n_kv x n_tokens, ~6.6 bytes a pair staged in
    // pinned host memory and again on the GPU. They grow with every image, and each time the pinned buffer grew
    // ROCm on Windows kept the old one: 27 images of 1920x1080 on a 58K conversation took 12 GiB of RAM, and a 14K
    // text append at batch 8192 then faulted (issue #2). Such ubatches are held to n_kv x n_tokens <= 2^27: about
    // 1024 tokens at 128K cells, 512 at 256K. The dense path is bound by attention over the whole cache, so this
    // costs it little: the same 27 images prefilled at the same rate, the append at 245 t/s instead of 261 at 2^28,
    // and the machine kept 11 GiB of RAM instead of 3. Text before any image keeps the full ubatch.
    // STRIX_DENSE_UBATCH_BUDGET sets the bound in cell-token pairs (0 = no bound).
    // strixllama (issue #13): only batches that take the dense inputs are bound - an image (embeddings without tokens),
    // several conversations, or text that an image cell does not wholly precede. One conversation's text after its
    // images takes the compact inputs (qsa_scalar_visibility), and so does the MTP draft head (tokens with embeddings),
    // whose catch-up after an image was held to the bound for nothing.
    bool dense_batch = kb_dup;
    if (dense_batch && hybrid_idx_image_text_compact()) {
        const llama_batch & b = balloc.get_batch();
        bool one_text_seq = b.token != nullptr && b.n_tokens > 0 && b.pos && b.n_seq_id && b.seq_id &&
                b.n_seq_id[0] == 1 && b.seq_id[0][0] >= 0 && b.seq_id[0][0] < LLAMA_MAX_SEQ;
        llama_pos pmin = one_text_seq ? b.pos[0] : 0;
        for (int32_t i = 0; one_text_seq && i < b.n_tokens; ++i) {
            one_text_seq = b.n_seq_id[i] == 1 && b.seq_id[i][0] == b.seq_id[0][0];
            pmin = std::min(pmin, b.pos[i]);
        }
        if (one_text_seq) {
            const auto & cells = get_mem_idx()->get_cells(b.seq_id[0][0]);
            dense_batch = hybrid_idx_image_from(cells, b.seq_id[0][0], pmin, 0, cells.size());
        }
    }
    if (dense_batch) {
        static const uint64_t budget = [] {
            const char * e = getenv("STRIX_DENSE_UBATCH_BUDGET");
            return e ? (uint64_t) strtoull(e, nullptr, 10) : (uint64_t) 1 << 27;
        }();
        const uint64_t n_kv = hybrid_idx_dense_n_kv(*get_mem_attn(), balloc.get_batch());
        if (budget > 0 && n_kv > 0) {
            const uint32_t cap = (uint32_t) std::max<uint64_t>(256, budget / n_kv / 256 * 256);
            if (cap < n_ubatch) {
                static uint32_t logged = 0;
                if (cap != logged) {
                    logged = cap;
                    LLAMA_LOG_INFO("%s: image in the conversation - ubatch %u -> %u tokens at %llu cells (dense inputs bounded)\n",
                                   __func__, n_ubatch, cap, (unsigned long long) n_kv);
                }
                n_ubatch = cap;
            }
        }
    }
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                // strixllama: sequences of different lengths in one ubatch (llama_ubatch::ragged), where the model
                // (qwen4exp, the only user of this memory) builds its recurrent layers per group of lengths;
                // STRIX_RAGGED=0: the equal-length split
                static const bool ragged = [] {
                    const char * e = getenv("STRIX_RAGGED");
                    return e == nullptr || atoi(e) != 0;
                }();
                if (ragged && unified && !balloc.has_coupled()) {
                    ubatch = balloc.split_ragged(n_ubatch, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
                } else {
                    ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
                }
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // strixllama: decide here whether the block-key cache may be wired into the graph built from this
        // context (see kb_pos_dup), because set_input_qsa will then rank cells instead of using their
        // position, and the cache cannot track ranked cells. Two ways to get there, both sticky:
        //   - an image ubatch: M-RoPE gives it a position per axis and repeats one position across the
        //     image, so cells share positions (`dup`);
        //   - a gap: this ubatch does not continue the sequence, so cells and positions drift apart and
        //     a cell can land outside the block window (`oor`). The MTP draft is where this bites -
        //     common_speculative_impl_draft_mtp::process() skips embedding batches, so the draft never
        //     stores the image's tokens and every later position is shifted past its cell.
        {
            // strixllama: per token and per sequence. A ubatch that serves several slots at once
            // (unified cache, equal-length split) carries tokens of several sequences, so its first
            // and last tokens belong to different sequences; comparing them as one made the flag
            // trip on any multi-stream decode, and it is sticky, so the block-key cache then stayed
            // off for every sequence, single streams included (measured: ROPE and GET_ROWS over
            // every block of the pool back in each step, 8% slower single-stream decode).
            std::map<llama_seq_id, llama_pos> next_pos;
            for (const auto & ub : ubatches) {
                if (hybrid_idx_ubatch_pos_dup(ub)) {
                    kb_dup = true;
                    break;
                }
                if (ub.n_tokens == 0 || !ub.pos || !ub.seq_id) {
                    continue;
                }
                bool gap = false;
                for (uint32_t i = 0; i < ub.n_tokens && !gap; ++i) {
                    if (!ub.seq_id[i]) {
                        continue;
                    }
                    const llama_seq_id s  = ub.seq_id[i][0];
                    // strixllama: a conversation that starts at position 0 is a fresh prompt (a mid-chain
                    // restore begins at pos_lo > 0), so its cells are contiguous from the origin and the
                    // block-key cache can track them again - clear the restore flag for this sequence.
                    if (ub.pos[i] == 0 && kb_seq_off.test(s)) {
                        kb_seq_off.reset(s);
                    }
                    const auto         it = next_pos.find(s);
                    const llama_pos    expect = it != next_pos.end()
                        ? it->second
                        : get_mem_attn()->seq_pos_max(s) + 1;
                    gap = ub.pos[i] > expect;
                    next_pos[s] = ub.pos[i] + 1;
                }
                if (gap) {
                    kb_dup = true;
                    break;
                }
            }
        }

        // strixllama: regions - when the batch does not fit after its conversations' last cells, the pool is
        // rebalanced first; the hook set in the constructor moves the indexer and the block keys along
        if (get_mem_attn()->can_move() && (!mem_idx || mem_idx->can_move())) {
            get_mem_attn()->move_cells(get_mem_attn()->plan_layout(ubatches));
        }

        // prepare the recurrent batches first
        if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    share_forget(-1);    // strixllama: IndexShare rows name cells
    kb_mark_stale(-1);   // strixllama: block keys depend on positions and cell contents; rebuild them once
    kb_dup = false;   // strixllama: no cells left, so no image cells either
    kb_seq_off.reset(); // strixllama: no restored sequences left either
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        // strixllama: a tail-only kv-chain restore keeps the PREVIOUS turn's attention rows in RAM and reloads
        // only the recurrent tail slice of the last matched chunk. If that blob carries a state position above
        // the matched rung (the run saved its tail at the end of the answer, not at the chunk boundary), the
        // fill loop's trim [rung, end) is a partial remove the recurrent cache cannot serve - its rollback is
        // bounded by n_rs_seq / the last recorded batch. Upstream returns false there and the server's caller
        // GGML_ABORTs ("failed to remove sequence %d with p0=%d, p1=%d", common_context_seq_rm). The delta-net
        // cannot be rewound to the rung, but dropping the whole sequence is always consistent, and the server
        // right after the remove detects the out-of-step cache and re-prefills the prompt from its start
        // ("the cache is out of step with its tokens" in server-context.cpp). Measured on the deployed build:
        // task 5528, restored 49152 cells tail-only, trim at p0=49152 aborted on the recurrent refusal.
        // STRIXLLAMA_SEQ_RM_DROP=0 restores the historical behaviour (propagate the refusal -> abort). Not
        // stripped by the server's env-bake (^STRIX_ does not match STRIXLLAMA_), so it works from a plain
        // environment like STRIXLLAMA_KB_RESTORE above.
        static const bool seq_rm_drop = !getenv("STRIXLLAMA_SEQ_RM_DROP") || atoi(getenv("STRIXLLAMA_SEQ_RM_DROP")) != 0;
        if (seq_rm_drop && seq_id >= 0 && p0 > 0 && p1 < 0) {
            LLAMA_LOG_WARN("%s: sequence %d cannot roll back to %d (recurrent tail too far ahead); dropping the sequence\n",
                    __func__, seq_id, p0);
            p0 = 0;
            // redo the recurrent removal over the full range; this is the one case where it cannot refuse:
            // it clears the sequence's rows entirely and resets its rollback index
            if (!get_mem_recr()->seq_rm(seq_id, 0, -1)) {
                return false;
            }
            // and the attention/indexer removal below completes the drop
        } else {
            return false;
        }
    }

    // strixllama: only a request that drops every sequence is proof the image cells are gone. A whole-
    // sequence seq_rm is NOT: the server calls it when it reuses a slot by longest-common-prefix and
    // then keeps the cached prefix, image cells included, so clearing the flag there wires the cache
    // back in under ranked cells and aborts. Measured that failure directly (tmp/vis_stress.py).
    if (seq_id < 0 && p0 <= 0 && p1 < 0) {
        kb_dup = false;
        kb_seq_off.reset();   // strixllama: every sequence dropped, so no restored cells left
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
    }

    // strixllama: a selection names cells of its own and earlier positions, so every selection from p0 on goes (a
    // later one may name a removed cell); one of an earlier position names only cells that stay
    share_forget(seq_id, p0, -1);

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // strixllama: the copy shares the source's cells, and with them block keys that may still have to be rebuilt
    if (seq_id_src >= 0 && seq_id_src < LLAMA_MAX_SEQ && kb_stale.test(seq_id_src)) {
        kb_mark_stale(seq_id_dst);
    }
    if (seq_id_src >= 0 && seq_id_src < LLAMA_MAX_SEQ && seq_id_dst >= 0 && seq_id_dst < LLAMA_MAX_SEQ) {
        kb_from[seq_id_dst] = std::min(kb_from[seq_id_dst], kb_from[seq_id_src]);   // and the blocks it left pending
    }

    // k-pool: only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single
    // pool grouping
    GGML_ASSERT((get_kpool() == 0 || (p0 <= 0 && p1 < 0)) && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }

    share_forget(seq_id_dst);   // strixllama: the destination's cells changed under its IndexShare rows
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    for (uint32_t s = 0; s < share_n_seq; ++s) {
        if ((llama_seq_id) s != seq_id) {
            share_forget((llama_seq_id) s);
        }
    }
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    share_forget(-1);    // strixllama: positions move under the IndexShare rows
    kb_mark_stale(-1);   // strixllama: block keys depend on positions; a shifted cell can be shared, so all of them
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    share_forget(-1);    // strixllama: positions move under the IndexShare rows
    kb_mark_stale(-1);   // strixllama: block keys depend on positions; a shifted cell can be shared, so all of them
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos pos_lo, llama_pos pos_limit) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags, pos_lo, pos_limit);
    }
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_write(io, seq_id, flags, pos_lo, pos_limit); }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags, pos_lo, pos_limit);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos pos_lo, llama_pos pos_limit) {
    share_forget(seq_id);   // strixllama: the restored cells are not the ones the IndexShare rows name
    kb_mark_stale(seq_id);   // strixllama: the restored cells' block keys were never written; rebuild them once
    // strixllama: a restore places cells through state_read_sinfo, which bypasses init_batch - the only other
    // place that decides whether the block-key cache may be wired. Restored cells need not start at position 0
    // (a mid-chain replay lands at pos_lo>0), so pos/r runs past the block window and set_input_qsa ranks them;
    // the block-key cache cannot track ranked cells and asserts. Mark this sequence's keys uncacheable so the
    // graph gate stops wiring the cache in (and can_reuse rebuilds any graph built before the restore without
    // it). Per-sequence, and NOT the global kb_dup: kb_dup also caps the ubatch for the dense-input budget, which
    // a plain text restore does not need and which would tank prefill at long context. Sticky, like the image case.
    // strixllama R2 (STRIXLLAMA_KB_RESTORE=1): the blanket disable above is over-conservative for the common
    // case. What actually breaks the cache is cells that are NOT dense from position 0, because then a cell's
    // block index is its rank rather than pos/r and the fill path asserts !ranked. A full kv-chain restore
    // tiles [0, n) contiguously (chunk 0 at pos_lo=0, later chunks APPEND), so the sequence stays dense from 0
    // and the cache tracks it fine - measured: forcing the cache through such a restore is bit-identical and
    // saves 11.53 ms/tok at 131k (slope_investigation_log.md C18/C19). A genuine mid-chain replay starts at
    // pos_min > 0 and still gets disabled. Decided AFTER the cells are placed, below; with the gate OFF this
    // is byte-identical to the historical behaviour (always disable).
    // PROMOTED 2026-10-16 (default ON): STRIXLLAMA_KB_RESTORE=0 restores the old blanket disable. Measured
    // on the shipping config: d131k 31.30 -> 36.64 t/s, fair decay 43.9% -> 16.9%, output bit-identical
    // through a kv-chain restore in both AR and MTP (slope_investigation_log.md C19-C21). Note this var is
    // NOT wiped by the server's env-bake (STRIXLLAMA_ does not match ^STRIX_), so the default lives here
    // rather than in the launcher, and the escape hatch works from a plain environment.
    static const bool kb_restore_ok = !getenv("STRIXLLAMA_KB_RESTORE") || atoi(getenv("STRIXLLAMA_KB_RESTORE")) != 0;
    if (!kb_restore_ok && seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        kb_seq_off.set(seq_id);
    }
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    // strixllama: does this blob actually carry attention (and so indexer) cells? The attention cache writes an
    // EMPTY section whenever PARTIAL_ONLY/TAIL_ONLY/COMP_ONLY is set (llama_kv_cache::state_write), so a TAIL_ONLY
    // or COMP_ONLY restore places no cells - it only refills the recurrent rings / compressed rows. The section
    // gate below is the looser (flags & PARTIAL_ONLY) == 0 so the empty bytes are still consumed in write order,
    // but the DESTRUCTIVE drop of the indexer's old cells must happen ONLY for a restore that repopulates it.
    // Otherwise (this was the crash): a kv-chain restore runs ATTN_ONLY (mirror intact) then TAIL_ONLY, whose
    // empty attn section wiped the indexer via seq_rm and restored nothing -> the indexer lost the restored
    // prefix, its graph window (get_kv_window) diverged from the attention cache's, and the graph's
    // "indexer cache must track the attention cache cell for cell" assert fired on the next prefill.
    const bool attn_has_data = (flags & (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY |
                                         LLAMA_STATE_SEQ_FLAGS_COMP_ONLY)) == 0;

    // strixllama: the kv-chain restore replays a prefix one ubatch-sized chunk at a time and passes APPEND on every
    // chunk after the first so the earlier chunks' cells survive (llama_kv_cache::state_read derives this from the
    // flag itself). The mirrored path below calls state_read_sinfo directly, so it has to derive it too - hardcoding
    // false there made every chunk wipe the ones before it, leaving only the last chunk restored while the server
    // still counted the whole prefix as cached.
    const bool append = (flags & LLAMA_STATE_SEQ_FLAGS_APPEND) != 0;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            // strixllama: the attention restore drops the sequence's old cells and may rebalance the pool before
            // it places the new ones (regions); the move hook repeats that rebalance on the indexer, which must
            // then hold the same cells - so the indexer drops them first too (its own restore would, later)
            if (attn_has_data && !append && mem_idx && seq_id >= 0 && get_mem_attn()->get_regions()) {
                mem_idx->seq_rm(seq_id, -1, -1);
            }
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, pos_lo, pos_limit, mem_idx ? &sinfos_attn : nullptr, nullptr, append);
        }

        if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_read(io, seq_id, flags, pos_lo, pos_limit); }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, pos_lo, pos_limit, nullptr, &sinfos_attn, append);
                // a restore that rewrites cells does it behind the pool layout's back
                if (!sinfos_attn.empty()) {
                    mem_idx_stale_set(seq_id, 0);
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }

    // strixllama R2: decide the cache flag now that the cells are actually placed. What the fill path needs is
    // a sequence whose cells are dense from position 0, so that a cell's block index is pos/r and not its
    // rank. A full kv-chain restore tiles [0, n) contiguously across its APPEND chunks, so pos_min stays 0
    // and the cache tracks it; a genuine mid-chain replay leaves pos_min > 0 and must stay disabled. The
    // density assumption (no interior gaps) is the same one kb_window_blocks already makes at n_cells =
    // p1 - p0 + 1, so this adds no new assumption to the file. kb_mark_stale above still forces exactly one
    // correct full rebuild for the restored cells, so the keys the cache serves afterwards are fresh.
    if (kb_restore_ok && seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        const llama_pos pmin = get_mem_attn()->seq_pos_min(seq_id);
        if (pmin == 0) {
            kb_seq_off.reset(seq_id);
        } else {
            kb_seq_off.set(seq_id);
        }
    }

    // strixllama: diagnostic - after a mirror restore the two caches must hold the same cells for the sequence.
    if (getenv("STRIX_KV_WINDOW_DEBUG") && atoi(getenv("STRIX_KV_WINDOW_DEBUG")) != 0 && seq_id >= 0) {
        const auto & ac = get_mem_attn()->get_cells(seq_id);
        fprintf(stderr, "KVWIN[state_read seq=%d flags=%u pos=[%d,%d)] attn cmin=%lld cmax=%lld",
                seq_id, (unsigned) flags, (int) pos_lo, (int) pos_limit,
                (long long) ac.seq_cell_min(seq_id), (long long) ac.seq_cell_max(seq_id));
        if (mem_idx) {
            const auto & ic = mem_idx->get_cells(seq_id);
            fprintf(stderr, " idx cmin=%lld cmax=%lld",
                    (long long) ic.seq_cell_min(seq_id), (long long) ic.seq_cell_max(seq_id));
        }
        fprintf(stderr, "\n");
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->seq_rm(seq_id, -1, -1); }

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

size_t llama_memory_hybrid_idx::kv_row_size() const {
    const size_t attn = get_mem_attn()->row_size();
    const size_t idx  = mem_idx ? mem_idx->row_size() : 0;
    return attn == 0 || (mem_idx && idx == 0) ? 0 : attn + idx;
}

bool llama_memory_hybrid_idx::kv_rows_get(llama_seq_id seq_id, llama_pos p0, uint32_t n, uint8_t * dst) const {
    if (kv_row_size() == 0) {
        return false;
    }
    return get_mem_attn()->seq_rows_get(seq_id, p0, n, dst) &&
           (!mem_idx || mem_idx->seq_rows_get(seq_id, p0, n, dst + (size_t) n*get_mem_attn()->row_size()));
}

bool llama_memory_hybrid_idx::kv_rows_set(llama_seq_id seq_id, llama_pos p0, uint32_t n, const uint8_t * src, uint32_t src_rows) {
    if (kv_row_size() == 0) {
        return false;
    }
    share_forget(seq_id);
    return get_mem_attn()->seq_rows_set(seq_id, p0, n, src, src_rows) &&
           (!mem_idx || mem_idx->seq_rows_set(seq_id, p0, n, src + (size_t) src_rows*get_mem_attn()->row_size(), src_rows));
}

bool llama_memory_hybrid_idx::kv_rows_copy(llama_seq_id seq_src, llama_seq_id seq_dst, llama_pos p0, uint32_t n) {
    if (kv_row_size() == 0) {
        return false;
    }
    share_forget(seq_dst);
    return get_mem_attn()->seq_rows_copy(seq_src, seq_dst, p0, n) &&
           (!mem_idx || mem_idx->seq_rows_copy(seq_src, seq_dst, p0, n));
}

bool llama_memory_hybrid_idx::kv_alloc(llama_seq_id seq_id, const llama_token * tokens, uint32_t n) {
    if (kv_row_size() == 0) {
        return false;
    }

    // the recurrent state goes too: the caller restores the one for position n next
    seq_rm(seq_id, -1, -1);
    kb_mark_stale(seq_id);                          // no block key of these cells was ever computed here

    // the indexer mirrors the attention cache cell for cell, as a restore does (state_read)
    // both with the attention cache's position sections, as a decode gives both the one batch
    const uint32_t n_pos = get_mem_attn()->n_pos_per_embd();
    llama_kv_cache::slot_info sinfo;
    if (!get_mem_attn()->seq_alloc(seq_id, tokens, n, n_pos, nullptr, &sinfo)) {
        return false;
    }
    if (mem_idx && !mem_idx->seq_alloc(seq_id, tokens, n, n_pos, n > 0 ? &sinfo : nullptr, nullptr)) {
        get_mem_attn()->seq_rm(seq_id, -1, -1);
        return false;
    }
    return true;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias, const qsa_kb_inputs * kb,
        uint32_t kv_off) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias, kb, nullptr, false, kv_off);
}

void llama_memory_hybrid_idx::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio, const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed,
        bool active_only, uint32_t kv_off) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true, kb, mixed, active_only, kv_off);
}

// strixllama: the indexer cells as the graph's view sees them, from its first cell (llama_kv_cache::get_kv_window)
struct hybrid_idx_cells_view {
    const llama_kv_cells & cells;
    const uint32_t         off;

    bool                              is_empty   (uint32_t j)                 const { return cells.is_empty(off + j); }
    const llama_kv_cells::seq_set_t & seq_get_all(uint32_t j)                 const { return cells.seq_get_all(off + j); }
    bool                              seq_has    (uint32_t j, llama_seq_id s) const { return cells.seq_has(off + j, s); }
    llama_pos                         pos_get    (uint32_t j)                 const { return cells.pos_get(off + j); }
    const llama_kv_cell_ext &         ext_get    (uint32_t j)                 const { return cells.ext_get(off + j); }
    llama_pos                         seq_pos_min(llama_seq_id s)             const { return cells.seq_pos_min(s); }
};

// strixllama: set_input_qsa_scan's inputs for its common case, from the runs instead of the cells: a ubatch whose sequences - one
// decoding, a prompt continuing its conversation, or several conversations decoding in one step - are each one run of cells
// (llama_kv_cells::seq_run: text only) sharing no cell with any other sequence, compact, with block selection. A sequence's
// blocks are its whole position buckets; the scan lists them bucket by bucket, and within a bucket the sequences in reverse
// order of their first cell (its groups are made in cell order and pushed onto the bucket's list at the head). So every input
// follows from the runs' first cells and positions. The scan is O(cells) with a bitset test each - 1.6 ms a step for one
// conversation at 110K cells, 10-14 ms for eight at 40K each - and grew with them; this is O(blocks) of plain writes.
// False: not that case (the scan takes it, and asserts what it asserts).
bool llama_memory_hybrid_idx::set_input_qsa_run(qsa_run_inputs & out, ggml_tensor * cell_blk, ggml_tensor * blk_cells,
        ggml_tensor * blk_pos, ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio,
        const qsa_kb_inputs * kb, const qsa_mixed_inputs * mixed, bool active_only, uint32_t kv_off) const {
    const int64_t r        = ratio;
    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;
    if (!tail_idxs || bias->type != GGML_TYPE_I32 || r < 2 || r > 64 || cell_blk->ne[1] != 1 || get_mem_idx() == nullptr ||
            n_tokens == 0 || !ubatch->pos || !ubatch->seq_id || !ubatch->n_seq_id) {
        return false;
    }
    const int64_t n_blocks = blk_pos->ne[0]/4;
    if (ggml_nelements(bias) != n_blocks + n_tokens || tail_idxs->ne[0] != r - 1 || tail_idxs->ne[1] != n_tokens ||
            tail_idxs->ne[2] != 1 || !bias->data || !tail_idxs->data) {
        return false;
    }
    const int64_t n_pb = active_only ? std::max(n_blocks, (n_kv + r - 1)/r) : n_blocks;

    // the ubatch's sequences, one a token, positions the same on every axis
    std::vector<llama_seq_id> useqs;
    for (int64_t i = 0; i < n_tokens; ++i) {
        if (ubatch->n_seq_id[i] != 1) {
            return false;
        }
        const llama_seq_id s = ubatch->seq_id[i][0];
        if (s < 0 || s >= LLAMA_MAX_SEQ) {
            return false;
        }
        if (std::find(useqs.begin(), useqs.end(), s) == useqs.end()) {
            useqs.push_back(s);
        }
        for (uint32_t axis = 1; axis < std::min<uint32_t>(ubatch->n_pos, 3); ++axis) {   // [p, p, p, 0]: see above
            if (ubatch->pos[i + axis*n_tokens] != ubatch->pos[i]) {
                return false;
            }
        }
    }
    const bool mixed_seqs = mixed != nullptr && mixed->seq_blk != nullptr && mixed->seq_tok != nullptr;
    const int64_t n_slots = mixed_seqs ? mixed->seq_blk->ne[0] : 0;
    if (useqs.size() > 1 && !mixed_seqs) {
        return false;
    }
    if (mixed_seqs && (!mixed->seq_blk->data || !mixed->seq_tok->data || !ubatch->seq_idx || !ubatch->seq_id_unq ||
            n_slots != (int64_t) ubatch->n_seqs_unq || mixed->seq_blk->ne[1] != n_blocks || mixed->seq_tok->ne[0] != n_slots ||
            mixed->seq_tok->ne[1] != n_tokens || mixed->seq_blk->type != GGML_TYPE_F32 || mixed->seq_tok->type != GGML_TYPE_F32)) {
        return false;
    }

    // each sequence one run inside the window; no cell of another sequence among any of them
    struct run_t {
        llama_seq_id s;
        uint32_t     c0, c1;
        llama_pos    p0, p1;
        int64_t      first, last;   // its whole buckets
        int64_t      base = 0;      // its first whole bucket's block, then one block a bucket from there (see blk_of)
    };
    const auto & cells = get_mem_idx()->get_cells(useqs[0]);
    std::vector<run_t> runs;
    for (const llama_seq_id s : useqs) {
        run_t R = {};
        R.s = s;
        if (!cells.seq_run(s, R.c0, R.c1, R.p0) || R.c0 < kv_off || (int64_t) R.c1 >= (int64_t) kv_off + n_kv || R.p0 < 0) {
            return false;
        }
        R.p1 = R.p0 + (llama_pos) (R.c1 - R.c0);
        if (R.p1/r >= n_pb) {
            return false;
        }
        R.first = (R.p0 + r - 1)/r;
        R.last  = (R.p1 + 1)/r - 1;
        runs.push_back(R);
    }
    for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
        if (cells.seq_pos_min(sq) < 0) {
            continue;
        }
        const int64_t m0 = cells.seq_cell_min(sq), m1 = cells.seq_cell_max(sq);
        for (const auto & R : runs) {
            if (R.s != sq && !(m1 < (int64_t) R.c0 || m0 > (int64_t) R.c1)) {
                return false;
            }
        }
    }

    // the block list: bucket by bucket, the runs covering a bucket in descending first cell
    std::vector<int> order(runs.size());
    for (size_t k = 0; k < runs.size(); ++k) {
        order[k] = (int) k;
    }
    std::sort(order.begin(), order.end(), [&](int x, int y) { return runs[x].c0 > runs[y].c0; });
    std::vector<int32_t> & blk_run = run_blk_run;   // per block: its run
    std::vector<int32_t> & blk_pb  = run_blk_pb;    // and its bucket
    blk_run.clear();
    blk_pb.clear();
    blk_run.reserve(n_blocks);
    blk_pb.reserve(n_blocks);
    {
        int64_t lo = INT64_MAX, hi = -1;
        for (const auto & R : runs) {
            if (R.first <= R.last) {
                lo = std::min(lo, R.first);
                hi = std::max(hi, R.last);
            }
        }
        // between two bucket numbers where a run starts or ends the list pattern repeats
        std::vector<int64_t> cuts;
        for (const auto & R : runs) {
            if (R.first <= R.last) {
                cuts.push_back(R.first);
                cuts.push_back(R.last + 1);
            }
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        for (size_t c = 0; c + 1 < cuts.size(); ++c) {
            std::vector<int> cover;
            for (const int k : order) {
                if (runs[k].first <= cuts[c] && cuts[c] <= runs[k].last) {
                    cover.push_back(k);
                }
            }
            for (int64_t pb = cuts[c]; pb < cuts[c + 1]; ++pb) {
                for (const int k : cover) {
                    blk_run.push_back(k);
                    blk_pb.push_back((int32_t) pb);
                }
            }
        }
        GGML_UNUSED(lo);
        GGML_UNUSED(hi);
    }
    const int64_t n_bid = (int64_t) blk_run.size();
    if (n_bid > n_blocks) {
        return false;
    }
    // each run's blocks, by bucket
    std::vector<std::vector<int32_t>> run_blk(runs.size());
    for (size_t k = 0; k < runs.size(); ++k) {
        if (runs[k].first <= runs[k].last) {
            run_blk[k].assign(runs[k].last - runs[k].first + 1, -1);
        }
    }
    for (int64_t b = 0; b < n_bid; ++b) {
        const auto & R = runs[blk_run[b]];
        run_blk[blk_run[b]][blk_pb[b] - R.first] = (int32_t) b;
    }
    auto cell = [&](const run_t & R, int64_t p) { return (int32_t) ((int64_t) R.c0 - kv_off + (p - R.p0)); };
    auto bpos = [&](int64_t b) { return (int32_t) (blk_pb[b]*r); };
    auto bcell = [&](int64_t b) { return cell(runs[blk_run[b]], bpos(b)); };
    auto run_of = [&](llama_seq_id s) -> const run_t & {
        for (const auto & R : runs) {
            if (R.s == s) {
                return R;
            }
        }
        GGML_ABORT("qsa run: no run for the sequence");
    };

    // strixllama: one conversation - every decode step of one, which is the case that costs - enumerates its blocks in
    // one stretch: the list above pushes (0, R.first), (0, R.first+1), ..., so blk_run is all 0 and blk_pb is
    // R.first + b. Every array below is then an arithmetic sequence and fills consecutively, instead of through the
    // lambdas' indirection (blk_run[b], then runs[..], then blk_pb[b]), which at 262K measured ~2.9 ns an element -
    // 2.7 GB/s, so the pass was bound by its own indexing and not by the memory it writes. blk_cells steps by 1
    // because a block's r cells are consecutive; bid_rows steps by 1 because the block-key cache lays rows by residue
    // mod r (kb_row), so consecutive blocks of one run land in consecutive rows. The general path stays for several
    // runs, where neither holds.
    const bool one_run = runs.size() == 1;
    const run_t & R0 = runs[0];
    const int64_t pbase = R0.first*r;                                  // blk_pos and limits: pbase + b*r
    const int64_t cbase = (int64_t) R0.c0 - kv_off + pbase - R0.p0;    // blk_cells: cbase + i, i = b*r + k
    const int64_t xbase = (int64_t) R0.c0 + pbase - R0.p0;             // bid_rows: kb_row(xbase + b*r) = +b

#ifndef NDEBUG
    if (one_run) {   // the claim above, checked once a step on the debug build
        GGML_ASSERT(n_bid == (R0.first <= R0.last ? R0.last - R0.first + 1 : 0));
        for (int64_t b = 0; b < n_bid; ++b) {
            GGML_ASSERT(blk_run[b] == 0 && blk_pb[b] == R0.first + b);
        }
    }
#endif

    if (blk_cells->data) {
        out.blk_cells.assign(r*n_blocks, 0);
        if (one_run) {
            for (int64_t i = 0, n = r*n_bid; i < n; ++i) {
                out.blk_cells[i] = (int32_t) (cbase + i);
            }
        } else {
            for (int64_t b = 0; b < n_bid; ++b) {
                for (int64_t k = 0; k < r; ++k) {
                    out.blk_cells[b*r + k] = cell(runs[blk_run[b]], bpos(b) + k);
                }
            }
        }
    }
    if (blk_pos->data) {
        out.blk_pos.assign(4*n_blocks, 0);
        for (int64_t sec = 0; sec < 4; ++sec) {
            int32_t * dst = out.blk_pos.data() + sec*n_blocks;
            if (one_run) {
                for (int64_t b = 0; b < n_bid; ++b) {
                    dst[b] = (int32_t) (pbase + b*r);
                }
            } else {
                for (int64_t b = 0; b < n_bid; ++b) {
                    dst[b] = bpos(b);
                }
            }
        }
    }
    out.limits.resize(n_blocks + n_tokens);
    if (one_run) {
        for (int64_t b = 0; b < n_bid; ++b) {
            out.limits[b] = (int32_t) (pbase + b*r);
        }
        std::fill(out.limits.begin() + n_bid, out.limits.begin() + n_blocks, INT32_MAX);
    } else {
        for (int64_t b = 0; b < n_blocks; ++b) {
            out.limits[b] = b < n_bid ? bpos(b) : INT32_MAX;
        }
    }
    if (mixed_seqs) {
        out.seq_blk.assign(n_slots*n_blocks, 0.0f);
        // each run's slot, once: a block's row is one-hot (its sequence is one of the ubatch's, each in one slot)
        std::vector<int64_t> run_slot(runs.size(), -1);
        for (size_t k = 0; k < runs.size(); ++k) {
            for (int64_t sl = 0; sl < n_slots; ++sl) {
                if (ubatch->seq_id_unq[sl] == runs[k].s) {
                    run_slot[k] = sl;
                }
            }
        }
        for (int64_t b = 0; b < n_bid; ++b) {
            const int64_t sl = run_slot[blk_run[b]];
            if (sl >= 0) {
                out.seq_blk[b*n_slots + sl] = 1.0f;
            }
        }
        out.seq_tok.assign(n_slots*n_tokens, 0.0f);
        for (int64_t ii = 0; ii < n_tokens; ++ii) {
            const int32_t sl = ubatch->seq_idx[ubatch->seq_id[ii][0]];
            if (sl < 0 || sl >= n_slots) {
                return false;
            }
            out.seq_tok[ii*n_slots + sl] = 1.0f;
        }
    }
    out.tail.assign((r - 1)*n_tokens, -1);
    for (int64_t ii = 0; ii < n_tokens; ++ii) {
        const run_t & R          = run_of(ubatch->seq_id[ii][0]);
        const int64_t q          = ubatch->pos[ii];
        const int64_t tail_start = (q + 1)/r*r;
        if (tail_start < 0 || tail_start > 16777216) {
            return false;
        }
        if (q + 1 > tail_start && tail_start/r < n_pb) {
            for (int64_t slot = 0; slot < q + 1 - tail_start; ++slot) {
                const int64_t p = tail_start + slot;
                if (p >= R.p0 && p <= R.p1) {
                    out.tail[ii*(r - 1) + slot] = cell(R, p);
                }
            }
        }
        out.limits[n_blocks + ii] = (int32_t) tail_start;
    }

    if (kb) {
        if (!kb->bid_rows || kb->bid_rows->ne[0] != n_blocks || !kb->bid_rows->data) {
            return false;
        }
        const int32_t scratch = (int32_t) kb_scratch_row();
        out.bid_rows.resize(n_blocks);
        // strixllama: kb_row lays a cell's row by residue mod kb_r, so one run's consecutive blocks - whose cells
        // step by r exactly - land in consecutive rows: kb_row(xbase + b*r) = kb_row(xbase) + b. The cache is sized
        // that way (kb_r = ratio), but the step is only taken when it holds, so a differently sized cache keeps the
        // general path instead of writing wrong rows.
        const bool rows_step1 = one_run && kb_r == r;
        if (rows_step1) {
            const int32_t xrow = (int32_t) kb_row(xbase);
            for (int64_t b = 0; b < n_bid; ++b) {
                out.bid_rows[b] = xrow + (int32_t) b;
            }
        } else {
            for (int64_t b = 0; b < n_bid; ++b) {
                out.bid_rows[b] = (int32_t) kb_row(kv_off + bcell(b));
            }
        }
        std::fill(out.bid_rows.begin() + n_bid, out.bid_rows.end(), scratch);
        // strixllama: TEST-ONLY fault injection (mirrors STRIXLLAMA_LB_FAULT). STRIXLLAMA_KB_FAULT=N makes the
        // LAST N completed blocks gather the SCRATCH row's key instead of their own, so their indexer score is
        // wrong and the block selection - and the text - MUST change. The last blocks are the most recent, which
        // a "summarise the context" query almost always attends to, so the corruption is guaranteed to be in the
        // attended set (corrupting the first blocks can miss the top-k entirely and read as a false blind gate).
        // This is the NEGATIVE CONTROL for the bit-identity gate (_idx_gate.ps1 -Mode negative): every shipped
        // indexer knob is bit-preserving by construction, so none can prove the gate watches the gathered keys.
        // If the text is identical under this injection the gate is blind and cannot certify anything. In-bounds:
        // scratch is a real row. Never set in production.
        static const int kb_fault = getenv("STRIXLLAMA_KB_FAULT") ? atoi(getenv("STRIXLLAMA_KB_FAULT")) : 0;
        if (kb_fault > 0) {
            const int64_t nf = std::min<int64_t>(kb_fault, n_bid);
            for (int64_t b = n_bid - nf; b < n_bid; ++b) {
                out.bid_rows[b] = scratch;
            }
            static bool once1 = false;
            if (!once1) { once1 = true; fprintf(stderr, "KB_FAULT site1 (out.bid_rows): nf=%lld n_bid=%lld n_blocks=%lld\n", (long long) nf, (long long) n_bid, (long long) n_blocks); }
        }
    }
    if (kb && kb->dirty_dst && kb->dirty_dst->data) {
        const int64_t dirty_max = kb->dirty_dst->ne[0];
        if (kb->dirty_cells->ne[0] != r*dirty_max || kb->dirty_pos->ne[0] != 4*dirty_max || !kb->dirty_cells->data || !kb->dirty_pos->data) {
            return false;
        }
        const int32_t scratch = (int32_t) kb_scratch_row();
        auto block_of = [&](size_t k, int64_t pb) -> int32_t {
            const auto & R = runs[k];
            return pb >= R.first && pb <= R.last ? run_blk[k][pb - R.first] : -1;
        };
        // written by the ubatch: each token's whole bucket in its own sequence; pending (kb_from): a sequence's whole buckets
        // holding a position from its pending start on, those not written
        std::vector<int32_t> written;
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            for (size_t k = 0; k < runs.size(); ++k) {
                if (runs[k].s == s) {
                    const int32_t b = block_of(k, ubatch->pos[i]/r);
                    if (b >= 0) {
                        written.push_back(b);
                    }
                }
            }
        }
        std::sort(written.begin(), written.end());
        written.erase(std::unique(written.begin(), written.end()), written.end());
        constexpr llama_pos none = std::numeric_limits<llama_pos>::max();
        std::vector<int32_t> pending;
        for (size_t k = 0; k < runs.size(); ++k) {
            const auto & R = runs[k];
            const llama_pos from = kb_from[R.s];
            if (from == none) {
                continue;
            }
            out.clear_from.push_back(R.s);
            for (int64_t pb = std::max<int64_t>(R.first, from/r - 1); pb <= R.last; ++pb) {
                const int32_t b = block_of(k, pb);
                if (b >= 0 && pb*r + r - 1 >= from && !std::binary_search(written.begin(), written.end(), b)) {
                    pending.push_back(b);
                }
            }
        }
        std::sort(pending.begin(), pending.end());
        out.dirty_cells.assign(r*dirty_max, 0);
        out.dirty_pos.assign(4*dirty_max, 0);
        out.dirty_dst.assign(dirty_max, 0);
        if ((int64_t) written.size() > dirty_max) {
            return false;
        }
        int64_t n_dirty = 0;
        bool    left    = false;
        for (int pass = 1; pass <= 2; ++pass) {
            for (const int32_t b : pass == 1 ? written : pending) {
                if (n_dirty >= dirty_max) {
                    left |= pass == 2;
                    continue;
                }
                for (int64_t slot = 0; slot < r; ++slot) {
                    out.dirty_cells[n_dirty*r + slot] = cell(runs[blk_run[b]], bpos(b) + slot);
                }
                for (int64_t sec = 0; sec < 4; ++sec) {
                    out.dirty_pos[sec*dirty_max + n_dirty] = bpos(b);
                }
                out.dirty_dst[n_dirty] = (int32_t) kb_row(kv_off + bcell(b));
                ++n_dirty;
            }
        }
        out.set_stale = !out.clear_from.empty() && left;
        for (int64_t d = n_dirty; d < dirty_max; ++d) {
            for (int64_t slot = 0; slot < r; ++slot) {
                out.dirty_cells[d*r + slot] = (int32_t) std::min<int64_t>(slot, n_kv - 1);
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                out.dirty_pos[sec*dirty_max + d] = 0;
            }
            out.dirty_dst[d] = scratch;
        }
    }
    return true;
}

void llama_memory_hybrid_idx::set_input_qsa_impl(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const qsa_kb_inputs * kb,
        const qsa_mixed_inputs * mixed,
        bool active_only,
        uint32_t kv_off) const {
    // strixllama: STRIX_QSA_RUN=0 scans every time; 2 takes both ways and aborts on any difference
    static const int run_mode = getenv("STRIX_QSA_RUN") ? atoi(getenv("STRIX_QSA_RUN")) : 1;
    qsa_run_inputs & ri = run_ri;
    ri.clear_from.clear();
    ri.set_stale = false;
    if (run_mode != 0 && blk_bias &&
            set_input_qsa_run(ri, cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kb, mixed, active_only, kv_off)) {
        constexpr llama_pos none = std::numeric_limits<llama_pos>::max();
        if (run_mode == 2) {
            std::vector<llama_pos> from0;
            std::vector<bool>      stale0;
            for (const llama_seq_id s : ri.clear_from) {
                from0.push_back(kb_from[s]);
                stale0.push_back(kb_stale.test(s));
            }
            const auto from_all  = kb_from;
            const auto stale_all = kb_stale;
            set_input_qsa_scan(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, blk_bias, kb, mixed, active_only, kv_off);
            auto same = [](const ggml_tensor * t, const void * v, size_t n) {
                return t == nullptr || t->data == nullptr || n == 0 || (ggml_nbytes(t) == n && memcmp(t->data, v, n) == 0);
            };
            auto same_i = [&](const ggml_tensor * t, const std::vector<int32_t> & v) { return same(t, v.data(), v.size()*sizeof(int32_t)); };
            auto same_f = [&](const ggml_tensor * t, const std::vector<float> & v) { return same(t, v.data(), v.size()*sizeof(float)); };
            // the scan's side effects: kb_from cleared for the sequences it had pending, kb_stale set for them when it left some
            auto expect_from = from_all;
            auto expect_stale = stale_all;
            for (const llama_seq_id s : ri.clear_from) {
                expect_from[s] = none;
                if (ri.set_stale) {
                    expect_stale.set(s);
                }
            }
            const bool ok = same_i(blk_cells, ri.blk_cells) && same_i(blk_pos, ri.blk_pos) && same_i(bias, ri.limits) &&
                same_i(tail_idxs, ri.tail) && (!kb || same_i(kb->bid_rows, ri.bid_rows)) &&
                (!kb || !kb->dirty_dst || !kb->dirty_dst->data ||
                    (same_i(kb->dirty_cells, ri.dirty_cells) && same_i(kb->dirty_pos, ri.dirty_pos) && same_i(kb->dirty_dst, ri.dirty_dst))) &&
                (!mixed || !mixed->seq_blk || (same_f(mixed->seq_blk, ri.seq_blk) && same_f(mixed->seq_tok, ri.seq_tok))) &&
                kb_from == expect_from && kb_stale == expect_stale;
            static int64_t n_checked = 0;
            if (!ok) {
                LLAMA_LOG_ERROR("%s: QSA_RUN check: the run's inputs differ from the scan's (ubatch of %u tokens, %u sequences)\n",
                        __func__, ubatch->n_tokens, ubatch->n_seqs_unq);
                GGML_ABORT("QSA_RUN check failed");
            }
            if (++n_checked % 32 == 0) {
                LLAMA_LOG_WARN("%s: QSA_RUN check: %lld ubatches, the run's inputs equal the scan's\n", __func__, (long long) n_checked);
            }
            return;
        }
        auto put = [](ggml_tensor * t, const void * v, size_t n) {
            if (t != nullptr && t->data != nullptr && n > 0) {
                GGML_ASSERT(ggml_nbytes(t) == n);
                memcpy(t->data, v, n);
            }
        };
        auto put_i = [&](ggml_tensor * t, const std::vector<int32_t> & v) { put(t, v.data(), v.size()*sizeof(int32_t)); };
        auto put_f = [&](ggml_tensor * t, const std::vector<float> & v) { put(t, v.data(), v.size()*sizeof(float)); };
        put_i(blk_cells, ri.blk_cells);
        put_i(blk_pos,   ri.blk_pos);
        put_i(bias,      ri.limits);
        put_i(tail_idxs, ri.tail);
        if (mixed && mixed->seq_blk) {
            put_f(mixed->seq_blk, ri.seq_blk);
            put_f(mixed->seq_tok, ri.seq_tok);
        }
        if (kb) {
            put_i(kb->bid_rows, ri.bid_rows);
            if (kb->dirty_dst && kb->dirty_dst->data) {
                put_i(kb->dirty_cells, ri.dirty_cells);
                put_i(kb->dirty_pos,   ri.dirty_pos);
                put_i(kb->dirty_dst,   ri.dirty_dst);
            }
        }
        for (const llama_seq_id s : ri.clear_from) {
            kb_from[s] = none;
            if (ri.set_stale) {
                kb_stale.set(s);
            }
        }
        return;
    }
    set_input_qsa_scan(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, blk_bias, kb, mixed, active_only, kv_off);
}

void llama_memory_hybrid_idx::set_input_qsa_scan(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const qsa_kb_inputs * kb,
        const qsa_mixed_inputs * mixed,
        bool active_only,
        uint32_t kv_off) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(tail_idxs || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    // strixllama: n_blocks is the length of the block list the graph was built for; n_pb is the window of
    // position buckets the cells are grouped by. They are one number unless the list covers only the
    // ubatch's own sequences (active_only, llama_memory_hybrid_idx_context::qsa_active_blocks): then the
    // list is shorter than the window, and every cell of another sequence is left out of the grouping -
    // its blocks were enumerated before only to be marked invisible to every query of the ubatch.
    GGML_ASSERT(!active_only || (tail_idxs && bias->type == GGML_TYPE_I32 && n_ns == 1));
    const int64_t n_pb = active_only ? std::max(n_blocks, (n_kv + r - 1)/r) : n_blocks;
    std::bitset<LLAMA_MAX_SEQ> active_seqs;
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) { active_seqs.set(ubatch->seq_id[i][k]); }
    }
    // strixllama: a cell of no sequence of this ubatch is invisible to every query in it, and its block key is
    // rebuilt by its own sequence's graph when it goes stale (kb_needs_full), so it is left out of the blocks.
    // That also keeps a window's foreign cells - the neighbours of the run a window views, whose positions have
    // nothing to do with it - out of the position buckets.

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    // strixllama: with the block-key cache an incremental graph builds its keys from the dirty list, so no
    // node reads blk_cells / blk_pos and ggml leaves them unallocated. They are still needed here (the
    // dirty-block scan reads them), so fall back to local scratch instead of writing through null.
    std::vector<int32_t> blk_cells_local, blk_pos_local;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    if (dst_blk_cells == nullptr) {
        blk_cells_local.assign((size_t) blk_cells->ne[0]*blk_cells->ne[1], 0);
        dst_blk_cells = blk_cells_local.data();
    }
    int32_t * dst_blk_pos = (int32_t *) blk_pos->data;
    if (dst_blk_pos == nullptr) {
        blk_pos_local.assign((size_t) blk_pos->ne[0], 0);
        dst_blk_pos = blk_pos_local.data();
    }
    const bool compact = bias->type == GGML_TYPE_I32;
    float * dst_bias = compact ? nullptr : (float *) bias->data;
    int32_t * limits = compact ? (int32_t *) bias->data : nullptr;
    // strixllama: several sequences in one compact ubatch - the membership inputs carry the sequence half
    const bool mixed_seqs = compact && mixed != nullptr && mixed->seq_blk != nullptr && mixed->seq_tok != nullptr;
    if (compact) {
        GGML_ASSERT(tail_idxs && blk_bias && n_ns == 1 && ggml_nelements(bias) == n_blocks+n_tps);
        if (mixed_seqs) {
            GGML_ASSERT(ubatch->seq_idx && ubatch->seq_id_unq && ubatch->n_seqs_unq >= 1);
            GGML_ASSERT(mixed->seq_blk->type == GGML_TYPE_F32 && mixed->seq_tok->type == GGML_TYPE_F32);
            GGML_ASSERT(mixed->seq_blk->ne[0] == ubatch->n_seqs_unq && mixed->seq_blk->ne[1] == n_blocks);
            GGML_ASSERT(mixed->seq_tok->ne[0] == ubatch->n_seqs_unq && mixed->seq_tok->ne[1] == n_tps);
            GGML_ASSERT(mixed->seq_blk->data && mixed->seq_tok->data);
        } else {
            for (int64_t i=0;i<n_tokens;++i) { GGML_ASSERT(ubatch->seq_id[i][0] == ubatch->seq_id[0][0]); }
        }
    }
    int32_t * dst_tail = tail_idxs ? (int32_t *) tail_idxs->data : nullptr;
    if (tail_idxs) {
        GGML_ASSERT(blk_bias && r > 1 && ggml_backend_buffer_is_host(tail_idxs->buffer));
        GGML_ASSERT(tail_idxs->ne[0] == r-1 && tail_idxs->ne[1] == n_tps && tail_idxs->ne[2] == n_ns);
        std::fill(dst_tail, dst_tail + ggml_nelements(tail_idxs), -1);
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_pb);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;
    // strixllama: with several sequences resident, the cells of each sequence of the ubatch, ranked among themselves
    std::map<llama_seq_id, std::vector<int32_t>> seq_order;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        // strixllama: every index below is relative to the view's first cell
        GGML_ASSERT(kv_off == 0 || n_ns == 1);
        const hybrid_idx_cells_view cells = { get_mem_idx()->get_cells(seq_of_stream), kv_off };

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }
                if ((cells.seq_get_all(j) & active_seqs).none()) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_pb) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position.
        // strixllama: `oor` needs the same treatment. M-RoPE also advances the position past an image by
        // its grid extent rather than by its token count, so positions run ahead of the cells holding
        // them and a cell can land outside the block window - the rank is dense by construction, so
        // ranking fixes that too. Without it the assert below aborts the server on the first long
        // enough conversation that contains an image.
        //
        // strixllama: with several sequences resident, each sequence of the ubatch is ranked among its own cells: the
        // cells of the others are invisible to its queries and left out of its blocks, so a cell's rank is its place
        // in its own conversation, as with one sequence resident. The ranking used to need one sequence in the whole
        // cache, and an image in a second resident conversation aborted the server below (group_members): unranked,
        // the image's repeated positions put two cells in one block slot. A cell two sequences of the ubatch share has
        // two places, so such a ubatch is not ranked (seq_cp; not what the slots do).
        if ((dup || oor) && ubatch->is_pos_2d()) {
            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            auto by_pos = [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            };

            order.clear();
            seq_order.clear();
            bool shared = false;

            for (int64_t j = 0; j < n_kv && !shared; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }
                if (one_seq) {
                    order.push_back((int32_t) j);
                    continue;
                }
                const auto own = cells.seq_get_all(j) & active_seqs;
                if (own.count() > 1) {
                    shared = true;
                } else if (own.any()) {
                    for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
                        if (own.test(sq)) {
                            seq_order[sq].push_back((int32_t) j);
                            break;
                        }
                    }
                }
            }

            if (!shared) {
                rank.assign(n_kv, -1);
                if (one_seq) {
                    std::sort(order.begin(), order.end(), by_pos);
                    for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                        rank[order[k]] = (int32_t) k;
                    }
                } else {
                    for (auto & [sq, ord] : seq_order) {
                        std::sort(ord.begin(), ord.end(), by_pos);
                        for (int64_t k = 0; k < (int64_t) ord.size(); ++k) {
                            rank[ord[k]] = (int32_t) k;
                        }
                    }
                }

                ranked = true;

                group_cells();
            }
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_pb; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk) { cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j]; }
        }

        std::vector<int32_t> group_members;
        if (dst_tail) {
            group_members.assign(grp_first.size()*r, -1);
            for (int64_t j=0;j<n_kv;++j) {
                const int32_t g=cell_grp[j];
                if (g<0) { continue; }
                const int64_t idx=ranked ? rank[j] : cells.pos_get(j);
                const int64_t slot=g*r+idx%r;
                GGML_ASSERT(group_members[slot]<0 || group_members[slot]==j);
                group_members[slot]=(int32_t)j;
            }
        }

        if (compact) {
            const llama_seq_id seq = ubatch->seq_id[0][0];
            for (int64_t b=0;b<n_blocks;++b) {
                // with several sequences the position half stays per block and the ownership goes to seq_blk
                limits[b] = b<n_bid && (mixed_seqs || cells.seq_has((uint32_t)bid_cell[b],seq)) ? bid_idx[b] : INT32_MAX;
            }
            if (mixed_seqs) {
                const int64_t n_slots = mixed->seq_blk->ne[0];
                float * sb = (float *) mixed->seq_blk->data;
                float * st = (float *) mixed->seq_tok->data;
                std::fill(sb, sb + n_slots*n_blocks, 0.0f);
                std::fill(st, st + n_slots*n_tps,    0.0f);
                for (int32_t b = 0; b < n_bid; ++b) {
                    const auto & owners = cells.seq_get_all((uint32_t) bid_cell[b]);
                    for (int64_t sl = 0; sl < n_slots; ++sl) {
                        if (owners.test(ubatch->seq_id_unq[sl])) { sb[b*n_slots + sl] = 1.0f; }
                    }
                }
                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const int32_t sl = ubatch->seq_idx[ubatch->seq_id[ii][0]];
                    GGML_ASSERT(sl >= 0 && sl < n_slots);
                    st[ii*n_slots + sl] = 1.0f;
                }
            }
        }

        // strixllama: block-key cache inputs. A cached block key goes stale only when one of the block's cells
        // is written, and a write happens exactly when a ubatch token lands in the cell, so the blocks this
        // ubatch completes are the full groups holding a cell whose (seq, pos) is a ubatch token.
        if (kb) {
            GGML_ASSERT(n_ns == 1 && !ranked && "qsa block-key cache: single stream, 1-D positions");
            GGML_ASSERT(kb->bid_rows && kb->bid_rows->ne[0] == n_blocks && kb->bid_rows->data);
            const int32_t scratch = (int32_t) kb_scratch_row();
            int32_t * br = (int32_t *) kb->bid_rows->data;
            for (int64_t b = 0; b < n_blocks; ++b) {
                br[b] = b < n_bid ? (int32_t) kb_row(kv_off + bid_cell[b]) : scratch;
            }
            // strixllama: TEST-ONLY fault injection, same as site1. This is the fill path the DECODE graph
            // actually uses (writes kb->bid_rows->data directly), which is why site1 alone read blind.
            static const int kb_fault2 = getenv("STRIXLLAMA_KB_FAULT") ? atoi(getenv("STRIXLLAMA_KB_FAULT")) : 0;
            if (kb_fault2 > 0) {
                const int64_t nf = std::min<int64_t>(kb_fault2, n_bid);
                for (int64_t b = n_bid - nf; b < n_bid; ++b) {
                    br[b] = scratch;
                }
                static bool once2 = false;
                if (!once2) { once2 = true; fprintf(stderr, "KB_FAULT site2 (bid_rows->data): nf=%lld n_bid=%lld n_blocks=%lld\n", (long long) nf, (long long) n_bid, (long long) n_blocks); }
            }
        }
        // the dirty list exists only in incremental graphs (a full-rebuild graph reads no such input)
        if (kb && kb->dirty_dst && kb->dirty_dst->data) {
            const int64_t dirty_max = kb->dirty_dst->ne[0];
            GGML_ASSERT(kb->dirty_cells->ne[0] == r*dirty_max && kb->dirty_pos->ne[0] == 4*dirty_max);
            GGML_ASSERT(kb->dirty_cells->data && kb->dirty_pos->data);
            const int32_t scratch = (int32_t) kb_scratch_row();

            llama_pos pmin = ubatch->pos[0], pmax = ubatch->pos[0];
            for (int64_t i = 1; i < n_tokens; ++i) {
                pmin = std::min(pmin, ubatch->pos[i]);
                pmax = std::max(pmax, ubatch->pos[i]);
            }
            // strixllama: and the blocks graphs without the indexer left pending (kb_from, kb_note_unrefreshed): a group
            // is 1 when the ubatch writes one of its cells, 2 when it only holds a pending cell of a ubatch sequence
            constexpr llama_pos none = std::numeric_limits<llama_pos>::max();
            std::vector<std::pair<llama_seq_id, llama_pos>> pend;
            llama_pos pend_min = none;
            for (int sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
                if (active_seqs.test(sq) && kb_from[sq] != none) {
                    pend.emplace_back((llama_seq_id) sq, kb_from[sq]);
                    pend_min = std::min(pend_min, kb_from[sq]);
                }
            }
            std::vector<uint8_t> grp_dirty(grp_first.size(), 0);
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];
                if (g < 0 || grp_bid[g] < 0 || grp_dirty[g] == 1) { continue; }
                const llama_pos p = cells.pos_get(j);
                if (p >= pmin && p <= pmax) {
                    for (int64_t i = 0; i < n_tokens; ++i) {
                        if (ubatch->pos[i] == p && cells.seq_has(j, ubatch->seq_id[i][0])) { grp_dirty[g] = 1; break; }
                    }
                }
                if (grp_dirty[g] == 0 && p >= pend_min) {
                    for (const auto & [sq, from] : pend) {
                        if (p >= from && cells.seq_has(j, sq)) { grp_dirty[g] = 2; break; }
                    }
                }
            }

            int32_t * dc = (int32_t *) kb->dirty_cells->data;
            int32_t * dp = (int32_t *) kb->dirty_pos->data;
            int32_t * dd = (int32_t *) kb->dirty_dst->data;
            int64_t n_dirty = 0;
            int64_t n_dirty_total = 0;   // the blocks the ubatch writes, which must fit
            bool    left = false;        // a pending block the list had no room for
            for (uint8_t pass = 1; pass <= 2; ++pass) {
                for (int32_t b = 0; b < n_bid; ++b) {
                    uint8_t dirty = 0;
                    for (int64_t slot = 0; slot < r; ++slot) {
                        const int32_t g = cell_grp[cur_blk_cells[b*r + slot]];
                        if (g >= 0 && grp_dirty[g]) { dirty = grp_dirty[g]; break; }
                    }
                    if (dirty != pass) { continue; }
                    n_dirty_total += pass == 1;
                    if (n_dirty >= dirty_max) { left |= pass == 2; continue; }
                    for (int64_t slot = 0; slot < r; ++slot) { dc[n_dirty*r + slot] = cur_blk_cells[b*r + slot]; }
                    for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + n_dirty] = dst_blk_pos[sec*n_blocks + b]; }
                    dd[n_dirty] = (int32_t) kb_row(kv_off + bid_cell[b]);
                    ++n_dirty;
                }
            }
            GGML_ASSERT(n_dirty_total <= dirty_max && "qsa block-key cache: more blocks completed than the graph can refresh");
            // nothing of these sequences is pending any more: refreshed by this list, or - what did not fit - by a full
            // rebuild on their next graph. qwen4exp gives every indexer layer of a graph this one input set (one ratio),
            // so the list reaches every layer's cache
            for (const auto & [sq, from] : pend) {
                kb_from[sq] = none;
                if (left) {
                    kb_stale.set(sq);
                }
            }
            for (int64_t d = n_dirty; d < dirty_max; ++d) {
                for (int64_t slot = 0; slot < r; ++slot) { dc[d*r + slot] = (int32_t) std::min<int64_t>(slot, n_kv - 1); }
                for (int64_t sec = 0; sec < 4; ++sec) { dp[sec*dirty_max + d] = 0; }
                dd[d] = scratch;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                // the query's place among its own sequence's cells
                const std::vector<int32_t> & ord = one_seq ? order : seq_order[seq_id];

                int64_t lo = 0;
                int64_t hi = (int64_t) ord.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = ord[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;
            if (dst_tail && q+1>tail_start) {
                const int64_t pb=tail_start/r;
                if (pb>=0 && pb<n_pb) {
                    int32_t * tail=dst_tail+(s*n_tps+ii)*(r-1);
                    for (int32_t g=grp_head[pb];g>=0;g=grp_next[g]) {
                        if (!cells.seq_has((uint32_t)grp_first[g],seq_id)) { continue; }
                        for (int64_t slot=0;slot<q+1-tail_start;++slot) {
                            const int32_t cell=group_members[g*r+slot];
                            if (cell<0 || !cells.seq_has((uint32_t)cell,seq_id)) { continue; }
                            GGML_ASSERT(tail[slot]<0 || tail[slot]==cell);
                            tail[slot]=cell;
                        }
                    }
                }
            }

            if (compact) {
                GGML_ASSERT(tail_start >= 0 && tail_start <= 16777216);
                limits[n_blocks+ii] = (int32_t)tail_start;
                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? (dst_tail ? -INFINITY : 1e9f) : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = dst_tail ? -INFINITY : 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, kb, get_idx()->get_kv_off());
}

ggml_tensor * llama_memory_hybrid_idx_context::get_kb(int32_t il) const {
    return mem ? mem->get_kb(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::kb_scratch_row() const {
    return mem ? mem->kb_scratch_row() : 0;
}

bool llama_memory_hybrid_idx_context::kb_needs_full(const llama_ubatch & ubatch) const {
    return mem ? mem->kb_needs_full(ubatch) : false;
}

bool llama_memory_hybrid_idx_context::kb_pos_dup() const {
    return mem ? mem->kb_pos_dup() : false;
}

bool llama_memory_hybrid_idx_context::kb_seq_unusable(const llama_ubatch & ubatch) const {
    return mem ? mem->kb_seq_unusable(ubatch) : false;
}

void llama_memory_hybrid_idx_context::kb_mark_full(const llama_ubatch & ubatch) const {
    if (mem) { mem->kb_mark_full(ubatch); }
}

void llama_memory_hybrid_idx_context::kb_note_unrefreshed(const llama_ubatch & ubatch) const {
    if (mem) { mem->kb_note_unrefreshed(ubatch); }
}

void llama_memory_hybrid_idx_context::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio,
        const llama_memory_hybrid_idx::qsa_kb_inputs * kb,
        const llama_memory_hybrid_idx::qsa_mixed_inputs * mixed,
        bool active_only) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, kb, mixed, active_only,
            get_idx()->get_kv_off());
}

bool llama_memory_hybrid_idx_context::qsa_position_prefix(const llama_ubatch & ubatch, bool active_only) const {
    if (!qsa_scalar_visibility(ubatch)) { return false; }
    const llama_seq_id seq=ubatch.seq_id[0][0];
    // a list of the ubatch's own sequences is one sequence's position prefix only if the ubatch has one sequence
    if (active_only) {
        for (uint32_t i=0;i<ubatch.n_tokens;++i) {
            if (ubatch.n_seq_id[i]!=1 || ubatch.seq_id[i][0]!=seq) { return false; }
        }
    }
    // strixllama: set_input_qsa_impl leaves other sequences' cells out of the blocks, so they cannot break the
    // prefix; the window starts at get_kv_off()
    return qsa_single_sequence_prefix(mem->get_mem_idx()->get_cells(seq),get_idx()->get_n_kv(),seq,true,
            get_idx()->get_kv_off());
}

// strixllama: in a unified cache every slot's conversation sits in one pool of cells and the graph spans the
// pool (n_kv), so the block list of a compact ubatch used to hold every conversation's blocks, the idle ones
// marked invisible: a decoded token paid for the block keys, the scores and the enumeration of conversations
// that were not running. The list can hold the ubatch's own sequences alone: a block's cells share one
// sequence set, every block of another set was invisible to every query here, and the visible ones keep
// their order, so the same blocks are selected in the same order. A sequence with at least the selection's
// budget of visible blocks gets bitwise the same result; a shorter one fills the budget with invisible
// entries, which now sit after its own blocks rather than among them, and that can move the last bits of its
// attention (towards what a single-slot server computes, which has no other conversation to fill it with).
// The length is an upper bound: at most one cell per position per sequence while no image has been written
// (kb_pos_dup), so a sequence holds at most pos_max - pos_min + 1 cells; rounded up to a multiple of 64
// blocks (256 cells at ratio 4, the step of n_kv) so a graph is rebuilt as rarely as before, and never below
// the budget. The reserve graph (positions all equal) keeps the whole pool: it sizes the buffers for the widest
// graph. LLAMA_QSA_ACTIVE_BLOCKS=0 turns this off.
int64_t llama_memory_hybrid_idx_context::qsa_active_blocks(const llama_ubatch & ubatch, uint32_t ratio, int64_t min_blocks) const {
    static const bool on = !getenv("LLAMA_QSA_ACTIVE_BLOCKS") || atoi(getenv("LLAMA_QSA_ACTIVE_BLOCKS")) != 0;
    if (!on || mem == nullptr || get_idx() == nullptr || ratio == 0 || get_n_stream() != 1 ||
            mem->kb_pos_dup() || !qsa_scalar_visibility(ubatch)) {
        return 0;
    }
    bool same_pos = ubatch.n_tokens > 1;
    for (uint32_t i = 1; same_pos && i < ubatch.n_tokens; ++i) {
        if (ubatch.pos[i] != ubatch.pos[0]) { same_pos = false; }
    }
    if (same_pos) {
        return 0;
    }
    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) { seqs.set(ubatch.seq_id[i][k]); }
    }
    const auto & cells = mem->get_mem_idx()->get_cells(ubatch.seq_id[0][0]);
    int64_t n_cells = 0;
    for (int32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (!seqs.test(s)) {
            continue;
        }
        const llama_pos p0 = cells.seq_pos_min(s);
        const llama_pos p1 = cells.seq_pos_max(s);
        if (p0 < 0 || p1 < p0) {
            return 0;
        }
        n_cells += (int64_t) p1 - p0 + 1;
    }
    const int64_t full   = ((int64_t) get_idx()->get_n_kv() + ratio - 1)/ratio;
    const int64_t blocks = std::max(min_blocks, ((n_cells + ratio - 1)/ratio + 63)/64*64);
    return blocks < full ? blocks : 0;
}

bool llama_memory_hybrid_idx_context::qsa_scalar_visibility(const llama_ubatch & ubatch) const {
    // see qwen4exp_qsa_embd_ok: a batch with tokens AND embeddings is the MTP draft head, not a vision batch
    const char * embd_env = getenv("LLAMA_QSA_TOKEN_EMBD");
    const bool embd_ok = !ubatch.embd || !embd_env || atoi(embd_env) != 0;
    if (get_n_stream()!=1 || !get_idx() || !ubatch.token || !embd_ok || !ubatch.pos || !ubatch.n_tokens ||
            !ubatch.n_pos || !ubatch.seq_id || !ubatch.n_seq_id) { return false; }
    if (ubatch.n_seq_id[0]<1 || !ubatch.seq_id[0]) { return false; }
    // strixllama: several sequences in one ubatch are fine (set_input_qsa_impl's membership inputs carry
    // which sequence owns a block) as long as every token names exactly one and no image or gap has
    // been written: those cells need the ranked enumeration, which only a single-sequence ubatch gets.
    const llama_seq_id seq=ubatch.seq_id[0][0];
    std::bitset<LLAMA_MAX_SEQ> seqs;
    for (uint32_t i=0;i<ubatch.n_tokens;++i) {
        if (ubatch.n_seq_id[i]<1 || !ubatch.seq_id[i] || ubatch.seq_id[i][0]<0 || ubatch.seq_id[i][0]>=LLAMA_MAX_SEQ ||
                ubatch.pos[i]<0 || ubatch.pos[i]>=16777216) { return false; }
        if (ubatch.seq_id[i][0]!=seq && ubatch.n_seq_id[i]!=1) { return false; }
        seqs.set(ubatch.seq_id[i][0]);
        for (uint32_t axis=1;axis<std::min<uint32_t>(ubatch.n_pos,3);++axis) {   // [p, p, p, 0]: see hybrid_idx_ubatch_pos_dup
            if (ubatch.pos[i+axis*ubatch.n_tokens]!=ubatch.pos[i]) { return false; }
        }
    }
    if (seqs.count()>1) {
        if (mem->kb_pos_dup() || !ubatch.seq_idx || !ubatch.seq_id_unq || ubatch.n_seqs_unq!=seqs.count()) { return false; }
        for (uint32_t i=0;i<ubatch.n_tokens;++i) {
            if (ubatch.n_seq_id[i]!=1 || ubatch.seq_idx[ubatch.seq_id[i][0]]<0) { return false; }
        }
    }
    if (ubatch.is_pos_2d()) {
        const auto & cells=mem->get_mem_idx()->get_cells(seq);
        // strixllama: a sequence that is one run of cells (llama_kv_cells::seq_run) holds no image cell - an image repeats a
        // position - so the window is scanned only for one that is not: 0.2 ms at 110K cells, two or three times a step
        bool runs=true;
        for (int s=0;s<LLAMA_MAX_SEQ && runs;++s) {
            uint32_t c0=0, c1=0;
            llama_pos p0=0;
            runs=!seqs.test(s) || cells.seq_run(s,c0,c1,p0);
        }
        const uint32_t off=get_idx()->get_kv_off();   // strixllama: the window's first cell
        // strixllama (issue #13): an image cell before every query of this text ubatch is visible to all of them by its
        // position alone, and set_input_qsa ranks the cells for the compact metadata; only one at or after the
        // ubatch's first position needs the dense inputs (see hybrid_idx_image_text_compact)
        const bool text_compact=hybrid_idx_image_text_compact();
        llama_pos pmin=ubatch.pos[0];
        for (uint32_t i=1;text_compact && i<ubatch.n_tokens;++i) { pmin=std::min(pmin,ubatch.pos[i]); }
        for (uint32_t j=off;!runs && j<off+get_idx()->get_n_kv();++j) {
            if (!cells.is_empty(j) && (cells.seq_get_all(j) & seqs).any() && cells.ext_get(j).is_2d_gt(cells.pos_get(j),cells.pos_get(j)) &&
                    (!text_compact || cells.pos_get(j)>=pmin)) { return false; }
        }
    }
    return true;
}

//
// k-pool DSA indexer (upstream: glm5-next, qwen4exp)
//

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(ggml_nelements(new_pool_idxs) == (int64_t) kpool*n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}
