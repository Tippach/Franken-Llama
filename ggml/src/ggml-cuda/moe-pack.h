#pragma once
// strixllama: MoE cross-expert N-column packing - Phase 1 reference (host, pure C++, no ggml, no GPU).
//
// WHY. The routed MoE (MUL_MAT_ID) tiles the N dimension (the routed rows) per expert: a BN-wide tile
// draws all its columns from ONE expert, so an expert with cnt < BN rows pads the tile. The Phase-0
// census (harness/_moe_census.ps1, 2026-10-15) measured 26.5% of the tile-columns are padding at the
// shipping ubatch - the gate for this rewrite. Cross-expert packing fills each BN-wide tile with
// consecutive routed rows across expert boundaries, so only the final partial tile pads: ceil(N/BN)
// tiles instead of sum_e ceil(cnt_e/BN) tiles.
//
// THE CRUX (why it is not free). A WMMA tile has ONE A (weight) matrix for all its N columns. A packed
// tile spanning several experts must load A_e for the column subrange belonging to e, compute, then
// load A_{e'} for the next subrange - one A-load per expert that touches the tile. This Phase-1 file
// only produces the LAYOUT + the per-tile expert subranges; Phase 2 ports the builder to a __global__
// and makes the routed kernel loop the subranges. Results must be bit-identical: every routed row is
// computed by exactly the same WMMA sequence as today, only the tile it sits in changes.
//
// Rows are assumed already grouped by expert (bounds[e]..bounds[e+1] is expert e's contiguous run), which
// is exactly what ggml_cuda_launch_mm_ids_bounded produces. The packed layout is then fully determined
// by bounds[] and BN. This reference is the oracle the device version and the test both check against.

#include <cstdint>

// The core is written once and used BOTH on the host (the test, compiled by cl.exe) and on the device
// (mmb.cu's __global__, compiled by clang/HIP). Under a CUDA/HIP compiler the qualifier makes it callable
// from BOTH host and device code, so a single -x hip TU can run the identical source on the host (the oracle)
// and inside a __global__ and memcmp the two - this is what tests/test-moe-pack-dev.cu checks. It uses only
// integer arithmetic and raw pointers - no CUDA intrinsics and no STL - so the source compiles in both worlds.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MOEPACK_HD __host__ __device__ __forceinline__
#else
#include <algorithm>
#include <vector>
#define MOEPACK_HD inline
#endif

struct moe_subrange {
    int32_t expert;     // which expert's rows this subrange covers
    int32_t col_start;  // first column (within the tile) of this subrange
    int32_t col_count;  // number of columns = number of this expert's rows in this tile
};

// Raw-pointer destination for the packing core, with capacities so it never overruns. The caller sizes
// these to the worst case (see moe_pack_worst_* below) and reads back the cursors.
struct moe_pack_buf {
    int32_t *     tile_sub_offset;   // cap n_tiles_max+1
    moe_subrange *subranges;         // cap subranges_max
    int32_t *     tile_bn;           // cap n_tiles_max
    int32_t *     tile_col_base;     // cap n_tiles_max
    int32_t *     col2row;           // cap cols_max
    int  n_tiles_max, subranges_max, cols_max;
    // out cursors
    int  n_tiles, n_sub, n_cols;
};

// Worst-case sizes for a given expert count E and routed-row count N:
//   tiles   <= N/BN_SMALL + E            (each expert can open a new tile; small tiles are BN_SMALL wide)
//   subranges <= tiles + E               (each tile holds >=1 subrange; +E for experts that split across tiles)
//   cols    <= N + E*BN_BIG              (real cols = N, plus at most one partial pad per class per expert)
// Worst-case sizes for a given expert count E and routed-row count N:
//   blocks/subranges <= N/16 + E            (each expert's rows split into <=16-row blocks; +E for the
//                                            last partial block of each expert)
//   tiles   <= blocks + 3                   (each tile holds >=1 block; +3 for a partial tile per class)
//   cols    <= N + E*16 + 3*BN_BIG          (real cols = N, +<=15 pad per expert's last block, +<=BN per class tail)
// The 16 is the WMMA fragment width: a subrange's col_start is always a multiple of 16 and it occupies 16
// columns (real rows then padding), so blocks - not raw rows - bound the counts. BN_BIG is the widest tile.
MOEPACK_HD int moe_pack_worst_tiles(int N, int E, int BN_SMALL) { (void) BN_SMALL; return N / 16 + E + 6; }
MOEPACK_HD int moe_pack_worst_subs (int N, int E, int BN_SMALL) { (void) BN_SMALL; return N / 16 + E + 6; }
MOEPACK_HD int moe_pack_worst_cols (int N, int E, int BN_BIG)   { return N + E * 16 + 3 * BN_BIG + 16; }

// The packing core. bounds has E+1 entries, bounds[E] = total routed rows N. Rows are grouped by expert
// (bounds[e]..bounds[e+1] is expert e's contiguous run), exactly what the device ids-prep yields.
//
// THE WMMA 16-COLUMN CONSTRAINT (this is what shapes the whole layout). mmb_tile_gemm runs 16x16x16 WMMAs:
// each accumulator fragment acc[i][j] covers 16 consecutive N-columns (routed rows) and multiplies them
// against ONE A sub-tile, i.e. ONE expert's weights (loaded once per wave). A fragment whose 16 rows came
// from two different experts would apply one expert's weights to the other's activations -> not bit-exact.
// Therefore every expert boundary MUST fall on a 16-column boundary. Concretely: each expert's rows are cut
// into blocks of <=16 rows; each block starts at a multiple of 16 and occupies 16 columns (real rows then
// padding). A subrange is one such block: col_start % 16 == 0, 0 < col_count <= 16. Packing then fills a
// tile's BN columns with whole 16-row blocks from consecutive same-class experts.
//
// CLASS-AWARE packing (the naive single-BN version FAILED the "must not add columns" gate): experts are
// bucketed into the SAME three classes mmb_build_desc3 uses (cnt>BN_MID -> big/BN_BIG; BN_SMALL<cnt<=BN_MID
// -> mid/BN_MID; 0<cnt<=BN_SMALL -> small/BN_SMALL), and cross-expert packing happens WITHIN a class. Because
// BN is a multiple of 16 in every class, blocks never straddle a tile boundary, so a class's column count is
// ceil(sum_class_blocks*16 / BN)*BN <= the current per-expert tiling: the win comes from experts whose
// per-expert remainder wasted columns (the census's 48..192 shape). col2row records the global row behind
// each column (padding = -1) so the permutation check is trivial and the kernel gathers activations by row.
//
// NOTE: column-waste is NOT the e2e prize. A wave straddling k experts loads A k times per K-step; the net
// prefill effect (B-load + launches saved vs A-reload added) is priced separately before the kernel rewrite.
MOEPACK_HD void moe_pack_core(const int32_t * bounds, int E, int BN_BIG, int BN_MID, int BN_SMALL, moe_pack_buf & b) {
    int n_tiles = 0, n_sub = 0, n_cols = 0;
    b.tile_sub_offset[0] = 0;
    // three passes, one per class, big -> mid -> small (matches desc3's tile ordering intent).
    for (int pass = 0; pass < 3; ++pass) {
        const int BN = pass == 0 ? BN_BIG : pass == 1 ? BN_MID : BN_SMALL;
        int cur_tile_cols = -1;   // -1 = no tile open
        for (int ee = 0; ee < E; ++ee) {
            const int cnt = bounds[ee + 1] - bounds[ee];
            if (cnt <= 0) { continue; }
            const bool in_class = pass == 0 ? (cnt > BN_MID) : pass == 1 ? (cnt > BN_SMALL && cnt <= BN_MID) : (cnt <= BN_SMALL);
            if (!in_class) { continue; }
            int remaining = cnt;
            int row_off = bounds[ee];
            // cut this expert's rows into <=16-row blocks; each block is one subrange occupying 16 columns.
            while (remaining > 0) {
                const int blk = remaining < 16 ? remaining : 16;   // real rows in this block
                if (cur_tile_cols < 0) {
                    b.tile_bn[n_tiles] = BN;
                    b.tile_col_base[n_tiles] = n_cols;
                    b.tile_sub_offset[n_tiles] = n_sub;
                    cur_tile_cols = 0;
                }
                // a block always starts at a multiple of 16 within the tile: BN % 16 == 0 and every prior
                // block consumed exactly 16 columns, so cur_tile_cols is already a multiple of 16 here.
                b.subranges[n_sub].expert = ee;
                b.subranges[n_sub].col_start = cur_tile_cols;
                b.subranges[n_sub].col_count = blk;
                n_sub++;
                for (int c = 0; c < blk; ++c) { b.col2row[n_cols++] = row_off + c; }
                for (int c = blk; c < 16; ++c) { b.col2row[n_cols++] = -1; }   // block pad to 16
                cur_tile_cols += 16; row_off += blk; remaining -= blk;
                if (cur_tile_cols == BN) { n_tiles++; cur_tile_cols = -1; }
            }
        }
        if (cur_tile_cols >= 0) {
            for (int c = cur_tile_cols; c < BN; ++c) { b.col2row[n_cols++] = -1; }  // tail pad to BN
            n_tiles++;
        }
    }
    b.tile_sub_offset[n_tiles] = n_sub;
    b.n_tiles = n_tiles; b.n_sub = n_sub; b.n_cols = n_cols;
}

// The CURRENT tiling's column count, replicating mmb_build_desc3's three classes (BN_BIG/BN_MID/BN_SMALL)
// so a test can compare packed vs current and report the recovered fraction. cnt>BN_MID -> ceil/BN_BIG
// big tiles; BN_SMALL<cnt<=BN_MID -> one mid; 0<cnt<=BN_SMALL -> one small. Pure integer, so it is available
// on host and device (the -x hip parity test uses it on the host path to report the recovered %).
MOEPACK_HD int moe_current_cols_desc3(const int32_t * bounds, int E, int BN_BIG, int BN_MID, int BN_SMALL) {
    int cols = 0;
    for (int ee = 0; ee < E; ++ee) {
        const int cnt = bounds[ee + 1] - bounds[ee];
        if (cnt <= 0) { continue; }
        if (cnt > BN_MID) { cols += ((cnt + BN_BIG - 1) / BN_BIG) * BN_BIG; }
        else if (cnt > BN_SMALL) { cols += BN_MID; }
        else { cols += BN_SMALL; }
    }
    return cols;
}

#ifndef __CUDACC__
#ifndef __HIPCC__
// Host-only convenience wrapper owning the storage, so the existing test keeps its vector API.
struct moe_pack {
    int n_tiles = 0;
    std::vector<int32_t>     tile_sub_offset;
    std::vector<moe_subrange> subranges;
    std::vector<int32_t>     tile_bn;
    std::vector<int32_t>     tile_col_base;
    std::vector<int32_t>     col2row;
};

inline void moe_pack_build_classed(const int32_t * bounds, int E, int BN_BIG, int BN_MID, int BN_SMALL, moe_pack & out) {
    const int N = bounds[E];
    const int tmax = moe_pack_worst_tiles(N, E, BN_SMALL);
    const int smax = moe_pack_worst_subs(N, E, BN_SMALL);
    const int cmax = moe_pack_worst_cols(N, E, BN_BIG);
    out.tile_sub_offset.assign(tmax + 1, 0);
    out.subranges.assign(smax, moe_subrange{ 0, 0, 0 });
    out.tile_bn.assign(tmax, 0);
    out.tile_col_base.assign(tmax, 0);
    out.col2row.assign(cmax, 0);
    moe_pack_buf b;
    b.tile_sub_offset = out.tile_sub_offset.data();
    b.subranges = out.subranges.data();
    b.tile_bn = out.tile_bn.data();
    b.tile_col_base = out.tile_col_base.data();
    b.col2row = out.col2row.data();
    b.n_tiles_max = tmax; b.subranges_max = smax; b.cols_max = cmax;
    moe_pack_core(bounds, E, BN_BIG, BN_MID, BN_SMALL, b);
    out.n_tiles = b.n_tiles;
    out.tile_sub_offset.resize(b.n_tiles + 1);
    out.subranges.resize(b.n_sub);
    out.tile_bn.resize(b.n_tiles);
    out.tile_col_base.resize(b.n_tiles);
    out.col2row.resize(b.n_cols);
}

inline int moe_pack_columns(const moe_pack & p) {
    int cols = 0;
    for (size_t t = 0; t < p.tile_bn.size(); ++t) { cols += p.tile_bn[t]; }
    return cols;
}
#endif
#endif
