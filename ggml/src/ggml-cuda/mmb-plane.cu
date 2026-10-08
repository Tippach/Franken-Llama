// mmb-plane.cu -- SLICE 16: routed-expert weight REPACK round-trip, TRUE 4-BIT K-MAJOR plane.
// Whole file is inert unless GGML_MMB_PLANE is defined (default OFF => live dispatch UNCHANGED).
//
// THIS FILE IS THE A-ROUTE (resident, byte-preserving) RE-CUT OF SLICE 14.
// Slice 14 stored codes_plane = bf16[code] = 2 B/weight = 66 B/block vs 18 B compact = 3.67x
// EXPANSION.  That shape cannot be swept in place (60 GB -> 218 GB > 108 GB ceiling).  Slice 16
// replaces it with a plane whose bytes/block EQUAL the compact IQ4_NL bytes/block, so the later
// in-place sweep (Slice 17) is genuinely self-evident.  Slice 14's bf16-code kernels are NOT lost:
// they live on as the B-route (per-layer turbo scratch) in mmb-plane-bf16codes.cu.
//
// ============================================================================================
// THE FORMAT (byte-preserving: 18 B/block == compact IQ4_NL)
// ============================================================================================
// Original IQ4_NL block (18 bytes):
//   bytes  0..1  : f16 block scale  d
//   bytes  2..17 : 16 bytes = 32 packed nibbles.  Source byte w (w = 0..15) holds
//                  weight n = 2w in its LOW nibble and weight n = 2w+1 in its HIGH nibble.
//
// Plane = TWO buffers, both indexed by the same block index i:
//   codes_plane : uint8_t[nblk * 16]   the 32 NIBBLES, K-major re-permuted, packed 2/byte.
//   scales_plane: __half [nblk]        the raw f16 scale d, hoisted (2 B/block).
//   => 16 + 2 = 18 B/block == compact.  BYTE-PRESERVING.  (G-BYTES asserts this.)
//
// CRITICAL: the IQ4_NL code range is -127..113, which does NOT fit in 4 bits, so the plane stores
// the NIBBLE (inherently 4-bit), NOT the code.  The GEMM (Slice 18) therefore STILL applies the
// codebook nibble->code->bf16 in the inner loop.  Per VRAM_STAGING §7c this plane removes ONLY the
// ~2% nibble address-gen term, NOT the ~98% 4-bit->bf16 expansion.  Do NOT make it bf16 again --
// that is the B route, a scratch-only architecture.
//
// ============================================================================================
// THE K-MAJOR SWIZZLE (explicit, strict bijection on the 32-nibble block)
// ============================================================================================
// A 16x16 WMMA tile steps K by 16, so a tile's K-slice is exactly one K-HALF of one block:
//   K-half 0 = weights n in [0,16)   (source bytes w in [0,8))
//   K-half 1 = weights n in [16,32)  (source bytes w in [8,16))
// Inside a half we separate the two nibble-parity classes so the two halves of a fragment's K-run
// sit at a FIXED BYTE OFFSET instead of requiring a mask/shift pairing across the row:
//
//   plane bytes  0..3  = LOW  nibbles of source bytes  0..7   (weights  0,2,...,14)
//   plane bytes  4..7  = HIGH nibbles of source bytes  0..7   (weights  1,3,...,15)
//   plane bytes  8..11 = LOW  nibbles of source bytes  8..15  (weights 16,18,...,30)
//   plane bytes 12..15 = HIGH nibbles of source bytes  8..15  (weights 17,19,...,31)
//
// FORWARD mapping (weight n -> plane position), n in [0,32):
//   half = n >> 4                       // K-half, 0 or 1
//   r    = n & 15                       // index inside the half
//   sb   = r >> 1                       // source byte inside the half (actual = half*8 + sb)
//   p    = r & 1                        // 0 = low nibble of that source byte, 1 = high nibble
//   q    = sb >> 1                      // pair group, 0..3
//   slot = sb & 1                       // 0 = low slot of the plane byte, 1 = high slot
//   PLANE BYTE  P = half*8 + p*4 + q
//   PLANE SLOT within P = slot
//
// INVERSE mapping (plane position -> source byte), P in [0,16), slot s in {0,1}:
//   half = P >> 3
//   g    = (P >> 2) & 1                 // 0 = low-nibble group, 1 = high-nibble group
//   q    = P & 3
//   w    = half*8 + q*2 + s             // source byte index in the block
//   nibble(w) LOW  <- nibble(plane[half*8     + q], s)
//   nibble(w) HIGH <- nibble(plane[half*8 + 4 + q], s)
//
// EQUIVALENT GATHER FORMS USED BY THE KERNELS (no read-modify-write, all indices compile-time):
//   forward, per plane byte P:   half=P>>3, g=(P>>2)&1, q=P&3, w0=half*8+q*2
//       plane[P] = nib_g(nib[w0]) | (nib_g(nib[w0+1]) << 4)
//   inverse, per source byte w:  half=w>>3, q=(w&7)>>1, s=w&1
//       nib[w]   = nib_s(plane[half*8+q]) | (nib_s(plane[half*8+4+q]) << 4)
//   where nib_g/nib_s pick the low (index 0) or high (index 1) nibble of a byte.
//
// BIJECTION PROOF: for a fixed half, the 16 weights split into 8 even (low nibbles of w0..w7) and
// 8 odd (high nibbles).  Lows fill plane bytes {0,1,2,3} x {slot0,slot1} exactly once (8 slots);
// highs fill {4,5,6,7} x {slot0,slot1} exactly once (8 slots) => 16 distinct positions per half,
// 32 across both halves, and the inverse formula is the exact functional inverse above.  The
// harness asserts this on-device before any data is moved (mmb_plane_validate_swizzle).
//
// CONTIGUITY: K-half 0 reads plane bytes [0,8) and K-half 1 reads plane bytes [8,16) -- 8
// contiguous bytes per 16-weight K-slice, with the parity classes already byte-separated.  Block
// order is unchanged (== GGUF file order), and codes_plane has a 16-byte stride, so every block's
// 16 bytes are 16-byte aligned off a 256-byte-aligned allocation and move as one uint4.
//
// ROUND-TRIP: because nibbles are stored VERBATIM (only reordered) and the scale is copied verbatim,
// inverse(forward(x)) == x is a permutation composed with its inverse => EXACT by construction,
// with NO codebook required in the inverse.
#ifdef GGML_MMB_PLANE

#ifdef PLANE_STANDALONE
// Standalone-harness build (Slice 16 test): no ggml headers, minimal stubs so the EXACT SAME
// kernels that ship in ggml-hip are what the round-trip gate exercises.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define CUDA_CHECK(x)                                                                                  \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            printf("HIP err %s @%d\n", hipGetErrorString(e_), __LINE__);                             \
            exit(2);                                                                                 \
        }                                                                                            \
    } while (0)
#define GGML_ASSERT(x)                                                                               \
    do {                                                                                             \
        if (!(x)) {                                                                                  \
            printf("ASSERT FAILED: %s @%d\n", #x, __LINE__);                                          \
            exit(3);                                                                                 \
        }                                                                                            \
    } while (0)
typedef hipStream_t cudaStream_t;
#else
#include "ggml-cuda/common.cuh"
#endif // PLANE_STANDALONE

// ---- format constants (G-BYTES) -------------------------------------------------------------
#define PLANE_BLOCK_BYTES_COMPACT 18   // IQ4_NL: 2 B f16 scale + 16 B of 32 packed nibbles
#define PLANE_CODES_BYTES_PER_BLK 16   // 32 nibbles packed 2/byte
#define PLANE_SCALES_BYTES_PER_BLK 2   // one f16 block scale
static_assert(PLANE_CODES_BYTES_PER_BLK + PLANE_SCALES_BYTES_PER_BLK == PLANE_BLOCK_BYTES_COMPACT,
              "4-bit K-major plane must be byte-preserving vs compact IQ4_NL");

// nibble helper: k = 0 -> low nibble, k = 1 -> high nibble.
__host__ __device__ __forceinline__ int plane_nib(uint8_t b, int k) {
    return (b >> (4 * k)) & 0x0F;
}

// ---- FORWARD gather index (plane byte -> the two contributing source bytes) -----------------
__host__ __device__ __forceinline__ void plane_fwd_index(int P, int &w0, int &g) {
    const int half = P >> 3;
    g = (P >> 2) & 1;
    const int q = P & 3;
    w0 = half * 8 + q * 2;
}

// ---- INVERSE gather index (source byte -> the two contributing plane bytes) ---------------
__host__ __device__ __forceinline__ void plane_inv_index(int w, int &Pl, int &Ph, int &s) {
    const int half = w >> 3;
    const int q = (w & 7) >> 1;
    s = w & 1;
    Pl = half * 8 + q;        // supplies the LOW nibble of source byte w
    Ph = half * 8 + 4 + q;    // supplies the HIGH nibble of source byte w
}

// Sanity: the two index helpers must be mutual inverses (host side of the bijection proof).
inline bool plane_swizzle_is_bijection() {
    int seen[32];
    for (int k = 0; k < 32; ++k) seen[k] = 0;
    for (int P = 0; P < 16; ++P) {
        int w0, g;
        plane_fwd_index(P, w0, g);
        for (int s = 0; s < 2; ++s) {
            const int w = w0 + s;
            int Pl, Ph, s2;
            plane_inv_index(w, Pl, Ph, s2);
            const int backP = (g == 0) ? Pl : Ph;
            if (backP != P || s2 != s) return false;
            const int key = P * 2 + s;
            if (key < 0 || key >= 32 || seen[key]) return false;
            seen[key] = 1;
        }
    }
    for (int k = 0; k < 32; ++k) {
        if (!seen[k]) return false;
    }
    return true;
}

// Device-side bijection check: walk every weight n forward, then back, and require the pair to agree.
__global__ void mmb_plane_swizzle_check_kernel(int * ok) {
    const int n = threadIdx.x; // 0..31 == weight index within the block
    if (n >= 32) return;
    const int half = n >> 4;
    const int r = n & 15;
    const int sb = r >> 1;
    const int p = r & 1;
    const int q = sb >> 1;
    const int slot = sb & 1;
    const int P = half * 8 + p * 4 + q;
    // now walk it back from the source byte this weight lives in
    const int w = half * 8 + sb;
    int Pl, Ph, s2;
    plane_inv_index(w, Pl, Ph, s2);
    const int backP = (p == 0) ? Pl : Ph;
    if (backP != P || s2 != slot) *ok = 0;
}

void mmb_plane_validate_swizzle() {
    GGML_ASSERT(plane_swizzle_is_bijection());
    int * d_ok;
    CUDA_CHECK(hipMalloc(&d_ok, sizeof(int)));
    const int one = 1;
    CUDA_CHECK(hipMemcpy(d_ok, &one, sizeof(int), hipMemcpyHostToDevice));
    mmb_plane_swizzle_check_kernel<<<1, 32>>>(d_ok);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());
    int host = 0;
    CUDA_CHECK(hipMemcpy(&host, d_ok, sizeof(int), hipMemcpyDeviceToHost));
    CUDA_CHECK(hipFree(d_ok));
    GGML_ASSERT(host == 1 && "4-bit K-major swizzle is not a strict bijection on the 32-nibble block");
}

// FORWARD repack kernel: one thread per 18-byte IQ4_NL block.
// 18 B read -> 16 B codes_plane + 2 B scales_plane.  Pure nibble permutation + scale copy.
__global__ void mmb_plane_repack_forward_kernel(
        const uint8_t * __restrict__ src,
        uint8_t * __restrict__ codes,
        __half * __restrict__ scales,
        const int nblk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint8_t * b = src + (size_t)i * PLANE_BLOCK_BYTES_COMPACT;

    // scales_plane: the f16 block scale, copied verbatim (bytes 0..1).
    scales[i] = *reinterpret_cast<const __half *>(b);

    const uint8_t * nib = b + 2;                 // 16 source bytes = 32 nibbles
    alignas(16) uint8_t plane[16];
#pragma unroll
    for (int P = 0; P < 16; ++P) {
        int w0, g;
        plane_fwd_index(P, w0, g);
        const uint8_t v0 = (uint8_t)plane_nib(nib[w0], g);
        const uint8_t v1 = (uint8_t)plane_nib(nib[w0 + 1], g);
        plane[P] = (uint8_t)(v0 | (v1 << 4));
    }
    // 16-byte stride off a 256-byte-aligned allocation => aligned uint4 store.
    *reinterpret_cast<uint4 *>(codes + (size_t)i * PLANE_CODES_BYTES_PER_BLK) =
        *reinterpret_cast<const uint4 *>(plane);
}

// INVERSE repack kernel: codes_plane + scales_plane -> EXACT original 18 bytes.
// Un-permute the nibbles and restore the f16 scale.  NO codebook: nibbles are stored verbatim.
__global__ void mmb_plane_repack_inverse_kernel(
        const uint8_t * __restrict__ codes,
        const __half * __restrict__ scales,
        uint8_t * __restrict__ dst,
        const int nblk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint4 plane4 =
        *reinterpret_cast<const uint4 *>(codes + (size_t)i * PLANE_CODES_BYTES_PER_BLK);
    const uint8_t * plane = reinterpret_cast<const uint8_t *>(&plane4);

    uint8_t * b = dst + (size_t)i * PLANE_BLOCK_BYTES_COMPACT;
    *reinterpret_cast<__half *>(b) = scales[i];   // restore f16 scale verbatim
    uint8_t * nib = b + 2;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        int Pl, Ph, s;
        plane_inv_index(w, Pl, Ph, s);
        const uint8_t lo = (uint8_t)plane_nib(plane[Pl], s);
        const uint8_t hi = (uint8_t)plane_nib(plane[Ph], s);
        nib[w] = (uint8_t)(lo | (hi << 4));
    }
}

// Host entry points (Slice 16 harness / later loader-stager).  NOT wired into the routed GEMM.
void mmb_plane_repack_forward(const uint8_t * src, uint8_t * codes, __half * scales,
                             int nblk, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (nblk + threads - 1) / threads;
    mmb_plane_repack_forward_kernel<<<blocks, threads, 0, stream>>>(src, codes, scales, nblk);
}

void mmb_plane_repack_inverse(const uint8_t * codes, const __half * scales,
                             uint8_t * dst, int nblk, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (nblk + threads - 1) / threads;
    mmb_plane_repack_inverse_kernel<<<blocks, threads, 0, stream>>>(codes, scales, dst, nblk);
}

// ============================================================================================
// SLICE 17 -- IN-PLACE CHUNKED SWEEP (compact <-> plane, SAME buffer footprint).
// ============================================================================================
// WHY IN-PLACE IS POSSIBLE: the 4-bit K-major plane is BYTE-PRESERVING (16 codes + 2 scale == 18 ==
// compact IQ4_NL).  And the Slice 16 nibble swizzle is STRICTLY INTRA-BLOCK: plane_fwd_index /
// plane_inv_index take a single block-local index, so no nibble ever crosses an 18-byte block.  The
// ONLY cross-position move is the scale HOIST (interleaved AoS <-> separate SoA array).
//
// CHUNK-LOCAL SoA.  A chunk of C blocks spans exactly 18*C bytes.  In COMPACT layout those bytes are
// C interleaved 18-byte blocks; in PLANE layout the SAME 18*C bytes are re-cut as
//   [ codes_plane : 16*C bytes ][ scales_plane : 2*C bytes ].
// Because the re-cut stays inside the chunk's own 18*C byte window, the sweep is IN-PLACE: we only
// need a scratch big enough to hold ONE chunk (18*C bytes) to snapshot before rewriting.  There is NO
// second full-tensor allocation.  Peak EXTRA device memory == one chunk, NOT the tensor.
//
// (A GLOBAL SoA plane -- codes for ALL blocks then scales for ALL blocks -- is NOT in-place with a
//  single chunk, because the global scales tail would overlap the not-yet-read high blocks.  Chunk-local
//  SoA is the direction-safe frame.  The GEMM (Slice 18) reads the same chunk-local layout.)
//
// ALIGNMENT: codes are written as uint4 (16 B).  A chunk starts at global byte c0*18, which is
// 16-byte aligned iff the starting block index c0 is a multiple of 8 (8*18 = 144 = 9*16).  The host
// wrapper enforces blocks_per_chunk % 8 == 0 so every chunk start (c0 = k*blocks_per_chunk) is a
// multiple of 8.  The final (partial) chunk keeps the same start alignment; its size may be < chunk.
//
// RACE SAFETY (one CTA processes one chunk):
//   step 1  snapshot the whole chunk (18*C bytes) main->scratch;  __syncthreads()
//   step 2  read scratch, write the codes region (thread i -> [i*16, i*16+16))
//           (writes disjoint from the scales region; all main reads already done via the barrier)
//   step 3  read scratch, write the scales region / compact scale bytes; disjoint from step 2.
// FORWARD and BACKWARD are exact inverses because each is (a permutation of nibbles within a block)
// composed with (a bijective AoS<->SoA scale move), and the two directions use mutual-inverse indices.
//
// The scratch pointer must point to >= 18*blocks_per_chunk bytes of device memory.
// ============================================================================================

// FORWARD in-place sweep of ONE chunk: compact interleaved -> chunk-local plane, same bytes.
__global__ void mmb_plane_sweep_fwd_kernel(
        uint8_t * __restrict__ chunk,     // 18*C bytes, 16-byte aligned
        uint8_t * __restrict__ scratch,   // >= 18*C bytes
        const long long C) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nthr = (long long)gridDim.x * blockDim.x;
    const long long chunk_bytes = C * PLANE_BLOCK_BYTES_COMPACT;
    // step 1: flat snapshot chunk -> scratch (byte copy: correct for any C, no alignment assumption).
    for (long long k = tid; k < chunk_bytes; k += nthr) scratch[k] = chunk[k];
    __syncthreads();
    // step 2: codes region from scratch (intra-block nibble permutation), uint4 store.
    for (long long i = tid; i < C; i += blockDim.x) {
        const uint8_t * sb = scratch + i * PLANE_BLOCK_BYTES_COMPACT;
        const uint8_t * snib = sb + 2;
        alignas(16) uint8_t plane[16];
#pragma unroll
        for (int P = 0; P < 16; ++P) {
            int w0, g;
            plane_fwd_index(P, w0, g);
            plane[P] = (uint8_t)(plane_nib(snib[w0], g) | (plane_nib(snib[w0 + 1], g) << 4));
        }
        *reinterpret_cast<uint4 *>(chunk + i * PLANE_CODES_BYTES_PER_BLK) =
            *reinterpret_cast<const uint4 *>(plane);
    }
    __syncthreads();
    // step 3: scales region (hoist AoS->SoA) at the chunk tail, from scratch.
    for (long long i = tid; i < C; i += blockDim.x) {
        const __half d = *reinterpret_cast<const __half *>(scratch + i * PLANE_BLOCK_BYTES_COMPACT);
        *reinterpret_cast<__half *>(chunk + C * PLANE_CODES_BYTES_PER_BLK + i * 2) = d;
    }
}

// BACKWARD in-place sweep of ONE chunk: chunk-local plane -> compact interleaved, same bytes.
__global__ void mmb_plane_sweep_bwd_kernel(
        uint8_t * __restrict__ chunk,     // 18*C bytes plane, 16-byte aligned
        uint8_t * __restrict__ scratch,   // >= 18*C bytes
        const long long C) {
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long nthr = (long long)gridDim.x * blockDim.x;
    const long long chunk_bytes = C * PLANE_BLOCK_BYTES_COMPACT;
    // step 1: flat snapshot plane -> scratch.
    for (long long k = tid; k < chunk_bytes; k += nthr) scratch[k] = chunk[k];
    __syncthreads();
    // step 2: un-permute nibbles into the compact block (bytes +2..+18), byte stores (unaligned target).
    for (long long i = tid; i < C; i += blockDim.x) {
        const uint8_t * scodes = scratch + i * PLANE_CODES_BYTES_PER_BLK;
        uint8_t * outnib = chunk + i * PLANE_BLOCK_BYTES_COMPACT + 2;
#pragma unroll
        for (int w = 0; w < 16; ++w) {
            int Pl, Ph, s;
            plane_inv_index(w, Pl, Ph, s);
            outnib[w] = (uint8_t)(plane_nib(scodes[Pl], s) | (plane_nib(scodes[Ph], s) << 4));
        }
    }
    __syncthreads();
    // step 3: restore the f16 scale at the compact block offset 0, from the scratch scales region.
    for (long long i = tid; i < C; i += blockDim.x) {
        const __half d = *reinterpret_cast<const __half *>(
            scratch + C * PLANE_CODES_BYTES_PER_BLK + i * 2);
        *reinterpret_cast<__half *>(chunk + i * PLANE_BLOCK_BYTES_COMPACT) = d;
    }
}

// Host entry points: sweep [0, nblk) blocks in chunks of blocks_per_chunk (MUST be a multiple of 8).
// scratch must be >= 18*blocks_per_chunk bytes of device memory.  Operates purely in place on buf.
void mmb_plane_sweep_forward(uint8_t * buf, uint8_t * scratch, long long nblk,
                            long long blocks_per_chunk, cudaStream_t stream) {
    GGML_ASSERT(blocks_per_chunk % 8 == 0 && "blocks_per_chunk must be a multiple of 8 (alignment)");
    // ONE CTA PER CHUNK (blocks = 1). The kernel's step-1 whole-chunk snapshot is followed by a
    // __syncthreads(), which is a barrier ONLY WITHIN a block. If we launched multiple CTAs per chunk,
    // CTA B could read scratch (step 2) before CTA A finished writing it (step 1) -> the in-place codes
    // region would clobber not-yet-read compact input. The design is explicitly 'one CTA == one chunk'.
    const int threads = 512;
    const int blocks = 1;
    for (long long c0 = 0; c0 < nblk; c0 += blocks_per_chunk) {
        long long C = nblk - c0;
        if (C > blocks_per_chunk) C = blocks_per_chunk;
        uint8_t * chunk = buf + c0 * PLANE_BLOCK_BYTES_COMPACT;
        mmb_plane_sweep_fwd_kernel<<<blocks, threads, 0, stream>>>(chunk, scratch, C);
    }
}

void mmb_plane_sweep_backward(uint8_t * buf, uint8_t * scratch, long long nblk,
                             long long blocks_per_chunk, cudaStream_t stream) {
    GGML_ASSERT(blocks_per_chunk % 8 == 0 && "blocks_per_chunk must be a multiple of 8 (alignment)");
    // ONE CTA PER CHUNK -- see forward wrapper for the __syncthreads()-is-block-local rationale.
    const int threads = 512;
    const int blocks = 1;
    for (long long c0 = 0; c0 < nblk; c0 += blocks_per_chunk) {
        long long C = nblk - c0;
        if (C > blocks_per_chunk) C = blocks_per_chunk;
        uint8_t * chunk = buf + c0 * PLANE_BLOCK_BYTES_COMPACT;
        mmb_plane_sweep_bwd_kernel<<<blocks, threads, 0, stream>>>(chunk, scratch, C);
    }
}

#endif // GGML_MMB_PLANE
