#pragma once

#include "llama.h"
#include "common.h"

#include <ggml-backend.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// needed for HANDLE / OVERLAPPED (the one in-flight chunk write). matched the
// guard style used elsewhere in the server so min/max stay macros-free.
#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#endif

namespace fs = std::filesystem;

// one matched chunk, in chain order. holds only the file path, not blob data:
// the caller replays the chunks one at a time (read -> set_data -> free),
// so the whole chain is never in RAM at once.
struct kv_chain_chunk {
    fs::path     file;     // this chunk's .kvcache (all three blobs live in this one file)
    llama_tokens tokens;   // the chunk's token IDs, verbatim from the file header
    llama_pos    pos_lo;   // KV position window start (stored: after an image the window is
    llama_pos    pos_hi;   //   NOT k*ubs, so it must be recorded, never derived)
    uint64_t     exit_fp;  // cumulative block-chain fingerprint of the state at this chunk's
                           //   end: H(entry_fp, this chunk's attn+comp+tail bytes). The next
                           //   cached chunk's entry_fp == this exit_fp (threaded by caller);
                           //   glue recomputed at restore folds its live state in between.
};

// which blob of a chunk file to read out (see the file layout note below)
enum class kv_chunk_slice {
    attn, // ATTN_ONLY rows for this chunk's window
    comp, // COMP_ONLY comp rows for this chunk's window (dsv4 only)
    tail, // the recurrent/tail snapshot taken at this chunk boundary
};

// disk-backed, content-addressed KV state store. ONE file per chunk:
//   <cache_dir>/<chunk_hash_hex>.kvcache
// header: u32 magic KVC1, u32 version, u32 hash32, u32 n_tokens,
//         llama_token[n_tokens], u32 attn_size, u32 comp_size, u32 tail_size
// body:   attn[attn_size] comp[comp_size] tail[tail_size]
// the three blobs are dumped straight into one pinned staging buffer at their
// own offsets, so a chunk is published with a single overlapped WriteFile and a
// single rename: either the whole chunk is on disk or none of it is, which is
// why the restore side needs no per-slice presence bookkeeping at all.
// tail_size is 0 on off-stride chunks (tail_stride): the tail is last-write-wins
// and constant-size, so only every Nth chunk (plus each prompt's last) carries it.
// no trailing checksum: verifying one costs a full pass over every (multi-MiB)
// recr file, which dominated restore time.
// the blob is a self-contained seq-state blob (its own io_magic + module
// header, no source seq_id), feedable straight to the state_seq_set API.
// root_hash = FNV-1a64 over a canonical metadata blob; everything in the blob
// must affect the numeric content or layout of cached KV values, so any change
// is a clean miss, never a garbage read.
struct kv_chain_metadata {
    int32_t  format_version;              // KV_CHAIN_VERSION (single version: file layout + this blob)
    uint32_t chunk_size;                  // == n_ubatch at store construction
    int64_t  model_file_size;             // stat() of the model file (-1 if stat fails)
    int64_t  model_file_mtime;            // stat() mtime seconds
    std::string arch;                     // model architecture string
    std::string ftype;                    // model quantization string
    uint32_t type_k;                      // ggml_type of the K cache
    uint32_t type_v;                      // ggml_type of the V cache
    int32_t  rope_scaling_type;           // llama_rope_scaling_type
    uint32_t rope_freq_base_bits;         // float bits (0.0f = "from model")
    uint32_t rope_freq_scale_bits;        // float bits (0.0f = "from model")
    uint32_t n_seq_max;                   // --parallel: the attn blob's n_stream scales with it, so a
                                          // different --parallel changes the root hash (clean miss)
};

// a free-list of pinned host buffers. the per-chunk dump size is fixed (it
// depends only on the ubatch window and the model geometry), so allocating a
// ~200 MiB pinned buffer on every save() and freeing it once the write lands is
// pure waste - and on WDDM cudaMallocHost/cudaFreeHost are synchronous and
// serialize the device, which stalls the compute-bound prefill pipeline. Instead
// the pool is warmed up over the first few chunks and then recycles the same
// buffers forever: acquire() pops a free one big enough, release() hands it back.
// Only the very first chunks ever touch the allocator.
class pinned_pool {
public:
    pinned_pool() = default;
    // a pinned buffer with capacity >= size, or nullptr if pinning is unavailable
    // (caller falls back to pageable).
    ggml_backend_buffer_t acquire(size_t size);
    void                  release(ggml_backend_buffer_t buf);
    ~pinned_pool();

    pinned_pool(const pinned_pool &)            = delete;
    pinned_pool & operator=(const pinned_pool &) = delete;

private:
    std::mutex                          mtx;
    std::vector<ggml_backend_buffer_t>  free_;
    ggml_backend_buffer_type_t          buft      = nullptr;
    bool                                buft_init = false;
    static constexpr size_t             cap       = 8;  // > in-flight buffers (exactly 1) + slack
};

// a host staging buffer for one dumped chunk. on an UMA box the GPU->host copy is
// 2-4x faster when the destination is PINNED (pageable copies are staged through
// the runtime's internal ring and even degrade with size), so we take the buffer
// from a pinned_pool when one is supplied and fall back to a plain pageable vector
// if pinning is unavailable. move-only RAII; on destruction it returns its buffer
// to the pool (or frees it if there is none), which is exactly when the write of
// that region has completed.
class dump_blob {
public:
    dump_blob() = default;
    dump_blob(size_t size, pinned_pool * pool) { reset(size, pool); }
    ~dump_blob() { free_buf(); }

    dump_blob(dump_blob && o) noexcept { *this = std::move(o); }
    dump_blob & operator=(dump_blob && o) noexcept {
        if (this != &o) {
            free_buf();
            m_pinned = o.m_pinned; o.m_pinned = nullptr;
            m_buf    = o.m_buf;    o.m_buf    = nullptr;
            m_pool   = o.m_pool;   o.m_pool   = nullptr;
            m_page   = std::move(o.m_page);
            m_size   = o.m_size;   o.m_size   = 0;
        }
        return *this;
    }
    dump_blob(const dump_blob &)            = delete;
    dump_blob & operator=(const dump_blob &) = delete;

    void reset(size_t size, pinned_pool * pool) {
        free_buf();
        m_pool = pool;
        m_size = size;
        if (size == 0) {
            return;
        }
        // prefer a recycled/alloc'd pinned buffer; the D2H into it is a direct DMA.
        if (m_pool != nullptr) {
            ggml_backend_buffer_t buf = m_pool->acquire(size);
            if (buf != nullptr) {
                void * base = ggml_backend_buffer_get_base(buf);
                if (base != nullptr) {
                    m_buf    = buf;
                    m_pinned = base;
                    return;
                }
                m_pool->release(buf);
            }
        }
        // fallback: pageable
        m_page.resize(size);
    }

    uint8_t * data()       { return m_pinned ? (uint8_t *) m_pinned : m_page.data(); }
    const uint8_t * data() const { return m_pinned ? (const uint8_t *) m_pinned : m_page.data(); }
    size_t    size() const { return m_size; }
    bool      pinned() const { return m_pinned != nullptr; }

private:
    void free_buf() {
        if (m_buf) {
            if (m_pool) {
                m_pool->release(m_buf);   // recycle: no device-serializing free on the hot path
            } else {
                ggml_backend_buffer_free(m_buf);
            }
            m_buf = nullptr;
        }
        m_pinned = nullptr;
        m_pool   = nullptr;
        m_page.clear();
        m_page.shrink_to_fit();
        m_size = 0;
    }

    void                    * m_pinned = nullptr; // pinned base ptr, or null
    ggml_backend_buffer_t     m_buf    = nullptr; // owning pinned buffer (null if pageable)
    pinned_pool *             m_pool   = nullptr; // pool to return m_buf to, or null to free it
    std::vector<uint8_t>      m_page;             // pageable fallback storage
    size_t                    m_size   = 0;
};

class kv_chain_store {
public:
    // ubatch_size: the chunk stride == n_ubatch (-ub). chunk boundaries are the
    // ubatch boundaries (where the state snapshots fire), not the n_batch size.
    // tail_stride: write the (last-write-wins, constant-size) tail slice only every
    // tail_stride-th chunk (plus the last chunk of each prompt), instead of at every
    // boundary. 1 = every chunk. >1 trades fork-mid-chain granularity for fewer
    // (~113 MiB on the hybrid model) tail dumps + writes.
    kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t ubatch_size,
                   const common_params & params, const llama_model * model,
                   int32_t tail_stride = 1);
    // waits for every in-flight chunk write to land on disk before tearing down.
    ~kv_chain_store();

    // ---- the running block-chain fingerprint -------------------------------------
    // at every prefill commit point the store holds a RUNNING fingerprint (running_fp_) that
    // uniquely identifies the KV state produced by everything prefilled so far. it is a plain
    // block-chain fold over each block's state bytes (attn||comp||tail), advanced whether that
    // block was loaded from disk, freshly computed, or is uncached "glue" (an image / a partial).
    // a chunk file is named by f(chunk_hash, running_fp): the input hash alone is not enough,
    // because the same text after a DIFFERENT image is a different KV state -- the running fp
    // differs, so the name differs, so a stale chunk is a clean miss (no header read needed to
    // reject it). the server never sees a fingerprint: it just reports commit points and whether
    // each is cacheable; the store owns running_fp_ and threads it internally.
    //
    // call begin_prompt() once at prompt arrival (before load_prefix / the first save), which
    // flushes any in-flight write and resets running_fp_ to root_hash_ (the model/config identity
    // that seeds chunk 0).
    void begin_prompt();

    // saves one cacheable chunk covering the KV window [pos_lo, pos_hi) of seq_id's state.
    // chunk_tokens (the block's identity: text token ids) are stored verbatim in the header for
    // validation. the file name is f(chunk_hash, running_fp_); if that file already exists the
    // chunk is a no-op (idempotent) and running_fp_ advances to its stored exit_fp. otherwise the
    // state is dumped async and, at publish time, its bytes are folded into running_fp_ to form
    // exit_fp (written into the header) -- so running_fp_ is left correct for the next block.
    // is_last: final chunk of the prompt -> force-write the tail slice regardless of tail_stride.
    // returns false on failure (store disabled, io error, ...). called from the server worker
    // thread only (like load_prefix/flush), which is why there is no locking.
    bool save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
              uint64_t chunk_hash, const llama_tokens & chunk_tokens,
              bool is_last = false);

    // advances running_fp_ over an UNCACHED block's live state (image / text-partial "glue"):
    // the block has just been decoded into [pos_lo, pos_hi), so its attn/comp/tail bytes are
    // folded into running_fp_ (synchronously -- glue is small and rare). no file is written.
    // this is what certifies cached text AFTER an image: a different image folds to a different
    // running_fp_, so the next cached block's name differs -> clean miss -> recompute.
    void fold_glue(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi);

    // hash for one chunk: FNV-1a over the chunk's token ids, seeded by prev_hash.
    uint64_t hash_chunk(const llama_tokens & chunk_tokens, uint64_t prev_hash) const;

    // fold a block's dumped state bytes into the running fingerprint:
    //   exit_fp = H(entry_fp, attn || comp || tail). Deterministic given identical bytes
    //   (prefill is bit-reproducible), so a legit match reproduces it exactly; a different
    //   earlier image changes the bytes and thus every downstream fp -> chain breaks there.
    static uint64_t fold_state_fp(uint64_t entry_fp, const uint8_t * attn, size_t attn_size,
                                  const uint8_t * comp, size_t comp_size,
                                  const uint8_t * tail, size_t tail_size);

    // the full hash chain for a prompt, computed once at prompt arrival:
    // hashes[k] = hash for chunk k (tokens [k*bs, (k+1)*bs)). the restore walk
    // and the per-ubatch save hook both index this vector, so save and restore
    // can never disagree about a chunk's name.
    std::vector<uint64_t> hash_chain(const llama_tokens & tokens) const;

    // walks the caller-provided chain hashes (the server owns the block schedule, since only
    // it sees the media layout), stopping at the first missing file -- that is the chain
    // break, and the caller prefills from there. chunk_identity[k] is the k-th cached chunk's
    // token ids (used to validate the matched file's header). NOT const: it advances running_fp_
    // as it goes -- chunk k's file name is f(chain_hashes[k], running_fp); a hit reads that file's
    // exit_fp and adopts it as the next running_fp, a miss stops the walk. so the store is left
    // holding the fingerprint of the restored prefix, and the first computed chunk after the
    // break is named consistently with how it was named when it was first saved. returns the
    // matched chunks in order (each carrying its stored pos_lo/pos_hi/exit_fp) and sets *n_cells
    // to the total cell count covered (= sum of matched chunk_identity sizes).
    std::vector<kv_chain_chunk> load_prefix(const std::vector<uint64_t> & chain_hashes,
                                            const std::vector<llama_tokens> & chunk_identity,
                                            size_t * n_cells);

    // continue the restore walk from a cursor, using the store's CURRENT running_fp_ (left by
    // load_prefix / save / fold_glue). this is what lets cached text AFTER an image be restored:
    // the image is prefilled (glue), fold_glue() folds its live state into running_fp_, and the
    // next cached chunk's name -- f(hash, running_fp_) -- then matches what was saved. the caller
    // invokes this once per prefill iteration until it returns no chunks. same tail-stride rule
    // as load_prefix: the returned run always ends on a chunk that carries a tail slice.
    std::vector<kv_chain_chunk> load_next(const std::vector<uint64_t> & chain_hashes,
                                          const std::vector<llama_tokens> & chunk_identity,
                                          size_t start_index, size_t * n_cells);

private:
    // shared forward walk behind load_prefix()/load_next(): starts at start_index from the
    // store's current running_fp_, stops at the first missing name, truncates to the last
    // tail-carrying chunk, and leaves running_fp_ at that chunk's exit_fp.
    std::vector<kv_chain_chunk> walk_chain(const std::vector<uint64_t> & chain_hashes,
                                           const std::vector<llama_tokens> & chunk_identity,
                                           size_t start_index, size_t * n_cells);

public:

    // reads + validates one chunk file (header magic/version + the token IDs must
    // match expected_tokens) and returns just the requested slice. returns false
    // if missing/corrupt/token-mismatch. The optional out params read back the chunk's
    // stored KV window and cumulative fingerprint.
    bool read_chunk_slice(const fs::path & file, std::vector<uint8_t> & out_blob,
                          const llama_tokens & expected_tokens, kv_chunk_slice slice,
                          llama_pos * out_pos_lo = nullptr, llama_pos * out_pos_hi = nullptr,
                          uint64_t * out_exit_fp = nullptr) const;

    // block until every in-flight chunk write has hit the disk (at most one write
    // plus one pending dump). called before shutdown, before a same-session restore
    // (a chunk written moments ago must not be seen as missing), and by tests that
    // read a chunk back right after saving it.
    void flush();

    size_t total_bytes() const { return total_bytes_cur; }
    bool   enabled() const { return !root_dir.empty(); }
    int32_t ubatch_size() const { return ubatch_size_; }
    uint64_t root_hash() const { return root_hash_; }
    // true when the model has compressed K caches (arch deepseek4): then each
    // chunk also gets a .cmcache file (the chunk's COMP_ONLY comp rows) and the
    // .rscache is rings-only. non-dsv4 archs: false, two files, no cm loop.
    bool has_comp() const { return has_comp_; }

    static uint64_t fnv1a64(const uint8_t * data, size_t len);
    static uint64_t fnv1a64(uint64_t h, const uint8_t * data, size_t len);

private:
    static std::string hash_str(uint64_t h);

    // a chunk file's base name (no extension): f(chunk_hash, running_fp). see save()/load_prefix().
    static std::string chunk_key_str(uint64_t chunk_hash, uint64_t running_fp);

    // computes the root hash from the metadata blob (see struct above).
    // the model file is stat()ed only, never opened.
    void compute_root_hash(const common_params & params, const llama_model * model);

    void evict_oldest(uint64_t need_bytes);

    // ---- the chunk pipeline: async dump -> overlapped write -> reap ----------------
    // three stages, at most one chunk in each, all advanced by save()/flush() on the
    // single server worker thread (no locks, no threads, no backpressure):
    //   1. copies: the state dump is ENQUEUED as async GPU->host copies into a pinned
    //      staging blob (llama_state_seq_get_data_*_async - NO ctx->synchronize(), so
    //      the GPU pipeline is never drained mid-prefill) and one event is recorded.
    //      the copies are stream-ordered right after the ubatch's kernels, so they
    //      capture exactly the post-ubatch state (recr tail included).
    //   2. write:  at the NEXT save() (or flush()) the event is awaited - instant in
    //      steady state, the copies finished a full chunk period ago - and the blob
    //      is published with one overlapped WriteFile, which the kernel reads while
    //      the GPU computes the current chunk. prefill emits a chunk every ~10 s and
    //      a ~200 MiB NVMe write takes ~0.2 s, so both waits are pure safety margins.
    //   3. reap:   at the save() after that, GetOverlappedResult + rename .tmp->final
    //      + the blob returns to the pool (the only ordering the design needs: never
    //      overwrite a buffer the kernel is still reading).
    struct pending_copies {
        bool      pending = false;
        bool      async   = false; // copies enqueued on the GPU stream; wait copy_event
        dump_blob blob;            // staging the chunk was dumped into (owned until published)
        fs::path  tmp;
        fs::path  file;
        uint64_t  bytes = 0;
        uint64_t  entry_fp = 0;    // running block-chain fp at this chunk's start (folded at publish)
        size_t    off_attn = 0;    // offsets/sizes of the state blobs within the staging region,
        size_t    off_comp = 0;    //   so publish_pending_copies can hash exactly those bytes
        size_t    off_tail = 0;
        size_t    sz_attn  = 0;
        size_t    sz_comp  = 0;
        size_t    sz_tail  = 0;
    };
    struct chunk_write {
        bool      pending = false;
        dump_blob blob;   // the staging the kernel is reading (owned until reaped)
        fs::path  tmp;
        fs::path  file;
        uint64_t  bytes = 0;
#ifdef _WIN32
        HANDLE     handle = INVALID_HANDLE_VALUE;
        OVERLAPPED ov     = {};
#endif
    };

    // wait for the in-flight write (if any), close its handle and rename .tmp ->
    // final, and release its staging blob to the pool. returns false if the write
    // failed (the .tmp file is then deleted).
    bool reap_pending_write();

    // wait the pending copies' event (instant in steady state) and publish the chunk
    // with one overlapped WriteFile of the whole staging region. no-op when nothing
    // is pending. returns false if the write could not be issued (chunk dropped).
    bool publish_pending_copies();

    // lazily create the copy event from ctx's KV device (first save). returns false
    // when events are unavailable -> save() uses the synchronous dump path instead.
    bool writer_ensure_event(llama_context * ctx);

    // serialises the chunk header into dst (which must hold kv_chunk_header_size()
    // bytes) and returns the same size.
    size_t write_header(uint8_t * dst, uint64_t chunk_hash, const llama_tokens & tokens,
                        llama_pos pos_lo, llama_pos pos_hi, uint64_t exit_fp,
                        uint32_t attn_size, uint32_t comp_size, uint32_t tail_size) const;

    // non-Windows fallback: synchronous write of the staging buffer + rename.
    bool write_staging_sync(const fs::path & tmp, const fs::path & file, const dump_blob & blob, size_t bytes);

    // pinned staging blobs, recycled across chunks (see pinned_pool). with the dump
    // and the write one stage apart there are exactly two live buffers in steady
    // state - the one the GPU is filling and the one the kernel is reading - and
    // because the per-chunk size is fixed by the ubatch window + model geometry they
    // are allocated once and reused forever. declared first so it is destroyed LAST.
    pinned_pool            kb_pool;
    pending_copies         copies;
    chunk_write            write_in_flight;
    ggml_backend_event_t   copy_event       = nullptr; // recorded after each chunk's async copies
    bool                   copy_event_tried = false;   // lazy init attempted (null event = sync fallback)

    std::string root_dir;
    uint64_t    root_hash_ = 0; // identity of this model/config; parent for chunk 0's hash
                                //   AND the seed of running_fp_ at the start of every prompt
    // the running block-chain fingerprint of the KV state at the current prefill commit point
    // (see begin_prompt/save/fold_glue). reset to root_hash_ per prompt, advanced by save() and
    // fold_glue(), and threaded forward by load_prefix() over the restored prefix.
    uint64_t    running_fp_ = 0;
    // monotonic commit-point counter for the current prompt; keys tail-stride (chunks are no
    // longer uniform-width once an image is present, so pos_lo/ubs can't be used for that).
    size_t      boundary_index_ = 0;
    uint64_t    limit_bytes;
    int32_t     ubatch_size_; // chunk stride == n_ubatch
    int32_t     tail_stride_; // write the tail slice every N chunks (+ last chunk of a prompt)
    std::atomic<uint64_t> total_bytes_cur { 0 }; // bytes published on disk (incl. in-flight)
    bool        has_comp_ = false; // model has compressed K caches (deepseek4)
};
