// mmb-plane-bf16codes.cu -- SLICE 14 PRESERVED (B ROUTE, scratch-only, NOT wired live).
//
// This is the ORIGINAL Slice 14 bf16-CODE plane, copied verbatim out of mmb-plane.cu before the
// Slice 16 re-cut, with every symbol renamed under a `bf16c_` prefix so it can coexist with the
// 4-bit K-major plane in the same build. Guarded by GGML_MMB_PLANE_BF16 (default OFF) => the
// file contributes NOTHING to the live binary unless explicitly turned on.
//
// WHY IT IS KEPT (user decision 2026-09-28, SLICE16 brief §PRESERVE THE B ROUTE):
//   codes_plane = bf16[code]  =>  2 bytes/weight  =>  66 B/block vs 18 B compact = 3.67x EXPANSION.
//   That shape CANNOT be resident / swept in place (60 GB -> 218 GB > 108 GB ceiling; bf16-resident
//   is the Slice 9 dead-end). BUT because the code is already materialised as bf16, a GEMM reading it
//   needs NO codebook in the inner loop: one fmaf(code, d, acc) covers the whole dequant class (~12%).
//   => B = PER-LAYER TURBO SCRATCH: build ONE layer's bf16 plane (~4.5 GB) into scratch during turbo
//      prefill, consume with a dequant-free GEMM, discard, next layer.  Bigger per-layer win than the
//      4-bit resident plane, but NOT resident and turbo-only.  Revisit AFTER the A-route in-place sweep
//      (Slice 17) + plane-reading GEMM (Slice 18) are working.
//
// PROPERTIES (unchanged from Slice 14, proven byte-exact 2026-09-28):
//   FORWARD : 18-byte IQ4_NL block -> codes_plane bf16[kvalue[nibble]] + scales_plane f16 d.
//   INVERSE : codes_plane + scales_plane -> EXACT original 18 bytes via the 256-entry inverse
//             codebook code->nibble (the 16-entry IQ4_NL codebook is a strict bijection;
//             lrintf(bf16(code)) == code for |code| <= 127, so no rounding).
//   Round-trip proven diff=0 on one expert (921,600 B) AND one full layer (471,859,200 B).
//
// DO NOT: make the A-route (resident) plane bf16.  That is this file's job only, as scratch.
#ifdef GGML_MMB_PLANE_BF16

#include "ggml-cuda/common.cuh"

// Authoritative IQ4_NL codebook (mirrors ggml-common.h kvalues_iq4nl).
static const int8_t bf16c_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Device constant tables: forward codebook + inverse codebook (code->nibble).
__constant__ int8_t  bf16c_kvalues[16];
__constant__ uint8_t bf16c_inv_code[256];

// Build + upload the constant tables once (idempotent).
static void bf16c_upload_tables() {
    static bool bf16c_tables_ready = false;
    if (bf16c_tables_ready) return;
    uint8_t inv[256];
    for (int k = 0; k < 256; ++k) inv[k] = 0xFF;
    for (int nib = 0; nib < 16; ++nib) {
        const int code = (int)bf16c_kvalues_iq4nl[nib];
        inv[(uint8_t)code] = (uint8_t)nib;
    }
    // bijection sanity: every nibble must map back to itself
    for (int nib = 0; nib < 16; ++nib) {
        const int code = (int)bf16c_kvalues_iq4nl[nib];
        GGML_ASSERT(inv[(uint8_t)code] == (uint8_t)nib);
    }
    CUDA_CHECK(hipMemcpyToSymbol(bf16c_kvalues, bf16c_kvalues_iq4nl, 16));
    CUDA_CHECK(hipMemcpyToSymbol(bf16c_inv_code, inv, 256));
    bf16c_tables_ready = true;
}

// FORWARD repack kernel: one thread per 18-byte IQ4_NL block.
__global__ void mmb_plane_bf16_repack_forward_kernel(
        const uint8_t * __restrict__ src,
        __hip_bfloat16 * __restrict__ codes,
        __half * __restrict__ scales,
        const int nblk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const uint8_t * b = src + (size_t)i * 18;
    const __half d = *reinterpret_cast<const __half *>(b);   // bytes 0..1 = f16 scale
    scales[i] = d;
    const uint8_t * nib = b + 2;                             // bytes 2..17 = 32 packed nibbles
    __hip_bfloat16 * c = codes + (size_t)i * 32;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const uint8_t byte = nib[w];
        const int code_lo = (int)bf16c_kvalues[byte & 0x0F];
        const int code_hi = (int)bf16c_kvalues[(byte >> 4) & 0x0F];
        c[2*w]   = __float2bfloat16((float)code_lo);
        c[2*w+1] = __float2bfloat16((float)code_hi);
    }
}

// INVERSE repack kernel: rebuild the EXACT original 18 bytes from the bf16-code plane.
__global__ void mmb_plane_bf16_repack_inverse_kernel(
        const __hip_bfloat16 * __restrict__ codes,
        const __half * __restrict__ scales,
        uint8_t * __restrict__ dst,
        const int nblk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    const __hip_bfloat16 * c = codes + (size_t)i * 32;
    uint8_t * b = dst + (size_t)i * 18;
    *reinterpret_cast<__half *>(b) = scales[i];               // restore f16 scale verbatim
    uint8_t * nib = b + 2;
#pragma unroll
    for (int w = 0; w < 16; ++w) {
        const int code_lo = (int)lrintf(__bfloat162float(c[2*w]));
        const int code_hi = (int)lrintf(__bfloat162float(c[2*w+1]));
        const uint8_t nib_lo = bf16c_inv_code[(uint8_t)code_lo];
        const uint8_t nib_hi = bf16c_inv_code[(uint8_t)code_hi];
        nib[w] = (uint8_t)((nib_lo & 0x0F) | ((nib_hi & 0x0F) << 4));
    }
}

// Host entry points (B-route scratch builder; NOT called from the live dispatch).
void mmb_plane_bf16_repack_forward(const uint8_t * src, __hip_bfloat16 * codes, __half * scales,
                                  int nblk, cudaStream_t stream) {
    bf16c_upload_tables();
    const int threads = 256;
    const int blocks = (nblk + threads - 1) / threads;
    mmb_plane_bf16_repack_forward_kernel<<<blocks, threads, 0, stream>>>(src, codes, scales, nblk);
}

void mmb_plane_bf16_repack_inverse(const __hip_bfloat16 * codes, const __half * scales,
                                  uint8_t * dst, int nblk, cudaStream_t stream) {
    bf16c_upload_tables();
    const int threads = 256;
    const int blocks = (nblk + threads - 1) / threads;
    mmb_plane_bf16_repack_inverse_kernel<<<blocks, threads, 0, stream>>>(codes, scales, dst, nblk);
}

#endif // GGML_MMB_PLANE_BF16
