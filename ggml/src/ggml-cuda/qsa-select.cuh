#pragma once

#include "common.cuh"

// strixllama SLICE 3: fused QSA indexer-select kernel.
//
// Replaces the I13+I14 chain (qwen4exp_apply_compact_visibility -> the TOP_K of
// qwen4exp_select_complete_blocks) with a single kernel that produces two buffers:
//   masked  = the visibility ADD output  (eff = visible ? score : -inf)
//   blocks  = the TOP_K output           (budget selected block indices, tie-break ascending)
// The downstream cell expansion (argsort -> get_rows -> concat tail) is already fused in the
// qsa_expand kernel and reads these buffers unchanged.
//
// See docs/SLICE2_SELECT_KERNEL_SPEC.md §D for the contract.
struct ggml_cuda_qsa_select_args {
    const ggml_tensor * score  = nullptr;  // [n_blocks, n_query] f32 contiguous (raw, pre-mask)
    const ggml_tensor * limits = nullptr;  // i32 input tensor the starts/tails views come from
    ggml_tensor       * masked = nullptr;  // [n_blocks, n_query] f32 OUT (visibility ADD output)
    ggml_tensor       * blocks = nullptr;  // [budget, n_query] i32 OUT (TOP_K output)
    int64_t starts_off = 0;              // element offset of starts within limits (int32)
    int64_t tails_off  = 0;              // element offset of tails within limits (int32)
    int n_blocks = 0;
    int n_query  = 0;
    int budget   = 0;   // block_budget = indexer_top_k / r
};

bool ggml_cuda_qsa_select_enabled();
void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_select_args & args);

// Matcher: anchored at a GGML_OP_REPEAT node, recognises the compact single-seq visibility chain
//   REPEAT -> SUB -> STEP -> LOG -> ADD -> TOP_K
// with correct producer edges, and fills `args` (score/starts/tails/masked/blocks). Returns the
// number of nodes to skip (5) on a match, else 0. Multi-seq (a MUL between STEP and LOG) and the
// non-compact path (ADD of a bias, not of a LOG) both fail the sequence and fall back to the chain.
int ggml_cuda_match_qsa_select(const ggml_cgraph * g, int i, ggml_cuda_qsa_select_args & a);
