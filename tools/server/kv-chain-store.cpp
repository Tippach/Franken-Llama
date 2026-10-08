#include "kv-chain-store.h"

#include "server-common.h"

#include "../../src/llama-ext.h" // llama_model_arch_name

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifdef __linux__
#include <fcntl.h>    // AT_FDCWD
#include <sys/stat.h> // utimensat
#endif

namespace fs = std::filesystem;

static constexpr uint32_t KV_CHAIN_MAGIC   = 0x4b564331; // "KVC1"
// single version number for the kv-chain format: the chunk file layout AND the
// root-hash metadata blob. bump on any change to either: a stale file is never
// read (version check) and the root hash changes, so old chunks are a clean miss.
// v7: one file per chunk instead of two/three (.kvcache + .cmcache + .rscache
// merged into a single .kvcache holding attn+comp+tail), which is what lets a
// chunk be published with one overlapped WriteFile and one rename. v6 and older
// chunk files are a clean miss.
// v8: the chunk header carries its KV position window (pos_lo/pos_hi) and the cumulative
// block-chain fingerprint (exit_fp), so chunks are no longer assumed uniform-width (k*ubs)
// -- that assumption desynced permanently once an image entered the prompt. old files miss
// cleanly (version check + root hash).
static constexpr uint32_t KV_CHAIN_VERSION = 8;

// true when the model has compressed K caches (arch deepseek4): then each chunk
// gets a third, .cmcache, file holding the chunk's comp rows (COMP_ONLY blob).
// non-dsv4 archs stay at two files and their restore is byte-identical to v5.
static bool kv_chain_has_comp(const llama_model * model) {
    return model != nullptr && std::string(llama_model_arch_name(model)) == "deepseek4";
}

// everything load_prefix / save() need from a chunk file WITHOUT touching the
// (multi-MiB) blob bodies: the three slice sizes (a 0 tail means the chunk was
// written on an off-stride boundary), the stored KV position window, and the
// cumulative block-chain fingerprint recorded at publish time. defined below;
// prototyped here so save()'s hit path (adopt exit_fp) can call it.
struct kv_chunk_header_info {
    uint32_t  sizes[3];   // attn, comp, tail
    llama_pos pos_lo;
    llama_pos pos_hi;
    uint64_t  exit_fp;
    uint32_t  n_tokens;
};
static bool probe_chunk_header(const fs::path & file, kv_chunk_header_info & info);

// resolve the device's pinned host buffer type once (null if pinning is off/unavailable).
static ggml_backend_buffer_type_t pinned_host_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    return dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
}

ggml_backend_buffer_t pinned_pool::acquire(size_t size) {
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (!buft_init) {
            buft      = pinned_host_buft();
            buft_init = true;
        }
        if (buft == nullptr) {
            return nullptr;
        }
        // reuse any free buffer that still fits (they are all the same fixed chunk
        // size in practice, so this is just the front of the list).
        for (size_t i = 0; i < free_.size(); ++i) {
            if (ggml_backend_buffer_get_size(free_[i]) >= size) {
                ggml_backend_buffer_t buf = free_[i];
                free_[i] = free_.back();
                free_.pop_back();
                return buf;
            }
        }
    }
    // none free: allocate a fresh pinned buffer (only during warmup).
    return ggml_backend_buft_alloc_buffer(buft, size);
}

void pinned_pool::release(ggml_backend_buffer_t buf) {
    if (buf == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(mtx);
    if (free_.size() < cap) {
        free_.push_back(buf);
    } else {
        ggml_backend_buffer_free(buf);
    }
}

pinned_pool::~pinned_pool() {
    for (auto & buf : free_) {
        ggml_backend_buffer_free(buf);
    }
    free_.clear();
}

kv_chain_store::kv_chain_store(std::string root_dir, uint64_t limit_bytes, int32_t ubatch_size,
                               const common_params & params, const llama_model * model,
                               int32_t tail_stride) :
    root_dir(std::move(root_dir)), limit_bytes(limit_bytes), ubatch_size_(ubatch_size > 0 ? ubatch_size : 512),
    tail_stride_(tail_stride > 0 ? tail_stride : 1),
    has_comp_(kv_chain_has_comp(model)) {
    if (!enabled()) {
        return;
    }

    compute_root_hash(params, model);

    std::error_code ec;
    const fs::path cache_dir = fs::path(this->root_dir);
    fs::create_directories(cache_dir, ec);
    if (ec) {
        SRV_ERR("kv-chain[storage]: failed to create cache dir '%s': %s (ec=%d)\n",
                cache_dir.string().c_str(), ec.message().c_str(), ec.value());
        this->root_dir.clear();
        this->root_hash_ = 0;
        return;
    }
    // remove stray .tmp files from an aborted previous run (a chunk is only ever
    // published by renaming .tmp -> final, so any .tmp left over is garbage), then
    // index existing chunks (eviction is global over the flat dir, LRU by mtime).
    for (const auto & entry : fs::directory_iterator(cache_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".tmp") {
            fs::remove(entry.path(), ec);
            continue;
        }
        const auto size = static_cast<uint64_t>(entry.file_size());
        total_bytes_cur += size;
    }
    // seed the running block-chain fingerprint: every prompt resets it to root_hash_
    // (begin_prompt), and root_hash_ is also the parent of chunk 0's content hash.
    running_fp_ = root_hash_;
    SRV_INF("kv-chain[storage]: cache dir '%s', root=%s, %zu bytes on disk, %.1f GiB (limit %.1f GiB), chunk ub=%d\n",
            cache_dir.string().c_str(), hash_str(root_hash_).c_str(), (size_t) total_bytes_cur.load(),
            (double) total_bytes_cur.load() / (1024.0*1024.0*1024.0),
            (double) limit_bytes / (1024.0*1024.0*1024.0),
            ubatch_size);
    SRV_INF("kv-chain[storage]: tail stride = %d (tail slice every %d-th chunk + last-of-prompt), dump = %s\n",
            tail_stride_, tail_stride_, "async (event)");
}

kv_chain_store::~kv_chain_store() {
    // make sure the last chunk of the run is on disk before the store disappears.
    flush();
    if (copy_event != nullptr) {
        ggml_backend_event_free(copy_event);
        copy_event = nullptr;
    }
}

void kv_chain_store::flush() {
    // drain both stages: finish the older write, publish the pending copies (event
    // wait + issue its write), then finish that write too. when flush() returns,
    // every saved chunk is fully on disk.
    (void) reap_pending_write();
    (void) publish_pending_copies();
    (void) reap_pending_write();
}

uint64_t kv_chain_store::fnv1a64(const uint8_t * data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t kv_chain_store::fnv1a64(uint64_t h, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t kv_chain_store::hash_chunk(const llama_tokens & chunk_tokens, uint64_t prev_hash) const {
    uint64_t h = 1469598103934665603ULL ^ prev_hash;
    for (auto t : chunk_tokens) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(&t);
        for (int i = 0; i < (int) sizeof(t); ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    return h;
}

// fold a block's dumped state bytes into the running fingerprint. the three slices are fed
// in a fixed order with their sizes, so a change in any of them (or a different layout)
// changes the result. deterministic given identical bytes.
uint64_t kv_chain_store::fold_state_fp(uint64_t entry_fp, const uint8_t * attn, size_t attn_size,
                                       const uint8_t * comp, size_t comp_size,
                                       const uint8_t * tail, size_t tail_size) {
    uint64_t h = 1469598103934665603ULL ^ entry_fp;
    auto feed = [&](const uint8_t * p, size_t n) {
        // length-prefix the size so slice boundaries are unambiguous
        for (int i = 0; i < 8; ++i) { h ^= (uint8_t) ((uint64_t) n >> (8*i)); h *= 1099511628211ULL; }
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    };
    feed(attn ? attn : (const uint8_t *) "", attn_size);
    feed(comp ? comp : (const uint8_t *) "", comp_size);
    feed(tail ? tail : (const uint8_t *) "", tail_size);
    return h;
}

std::vector<uint64_t> kv_chain_store::hash_chain(const llama_tokens & tokens) const {
    std::vector<uint64_t> hashes;
    const size_t bs = (size_t) ubatch_size_;
    if (bs == 0 || tokens.empty()) {
        return hashes;
    }
    const size_t n_chunks = tokens.size() / bs;
    hashes.reserve(n_chunks);
    // chain chunk 0 off the root hash: keeps different models/configs in disjoint
    // hash namespaces even for identical leading tokens.
    uint64_t prev = root_hash_;
    for (size_t k = 0; k < n_chunks; ++k) {
        const llama_tokens block(tokens.begin() + k * bs, tokens.begin() + (k + 1) * bs);
        prev = hash_chunk(block, prev);
        hashes.push_back(prev);
    }
    return hashes;
}

std::string kv_chain_store::hash_str(uint64_t h) {
    static const char * hex = "0123456789abcdef";
    std::string s(16, '0');
    for (int i = 15; i >= 0; --i) {
        s[i] = hex[h & 0xf];
        h >>= 4;
    }
    return s;
}

// a chunk file's name is f(chunk_hash, running_fp): the input-content hash AND the
// running block-chain fingerprint of the KV state that precedes the chunk. the fp is
// folded in so the same text after a different image (a different KV state) yields a
// different name -> a clean miss, with no need to open a non-matching file. both sides
// (save and load) build the name the same way, so they can never disagree.
std::string kv_chain_store::chunk_key_str(uint64_t chunk_hash, uint64_t running_fp) {
    uint64_t mixed = running_fp;
    for (int i = 0; i < 8; ++i) { mixed ^= (uint8_t) (chunk_hash >> (8*i)); mixed *= 1099511628211ULL; }
    for (int i = 0; i < 8; ++i) { mixed ^= (uint8_t) (running_fp >> (8*i)); mixed *= 1099511628211ULL; }
    return hash_str(mixed);
}

// feeds a field into a running FNV-1a hash, length-prefixed (u64 LE) so field
// boundaries are unambiguous.
static uint64_t hash_field(uint64_t h, const void * data, size_t len) {
    uint8_t lbuf[8];
    for (int i = 0; i < 8; ++i) {
        lbuf[i] = static_cast<uint8_t>((len >> (8 * i)) & 0xff);
    }
    h = kv_chain_store::fnv1a64(h, lbuf, 8);
    h = kv_chain_store::fnv1a64(h, reinterpret_cast<const uint8_t *>(data), len);
    return h;
}
static uint64_t hash_le(uint64_t h, uint64_t v, size_t nbytes) {
    uint8_t b[8] = {0};
    for (size_t i = 0; i < nbytes && i < 8; ++i) {
        b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
    }
    return hash_field(h, b, nbytes);
}
static uint64_t hash_str_field(uint64_t h, const std::string & s) {
    return hash_field(h, s.data(), s.size());
}

void kv_chain_store::compute_root_hash(const common_params & params, const llama_model * model) {
    kv_chain_metadata md;
    md.format_version    = KV_CHAIN_VERSION;
    md.chunk_size        = ubatch_size_;
    md.model_file_size   = -1;
    md.model_file_mtime  = -1;
    md.arch              = "";
    md.ftype             = "";
    md.type_k            = params.cache_type_k;
    md.type_v            = params.cache_type_v;
    md.rope_scaling_type = params.rope_scaling_type;
    // the attn blob's n_stream scales with --parallel, so a different value
    // changes the blob layout -> it must change the root hash (clean miss).
    md.n_seq_max           = static_cast<uint32_t>(params.n_parallel > 0 ? params.n_parallel : 1);
    {
        // copy the float bits portably (memcpy, not a type-punned pointer cast)
        const float rope_freq_base  = params.rope_freq_base;  // 0.0f = "from model"
        const float rope_freq_scale = params.rope_freq_scale; // 0.0f = "from model"
        std::memcpy(&md.rope_freq_base_bits,  &rope_freq_base,  sizeof(uint32_t));
        std::memcpy(&md.rope_freq_scale_bits, &rope_freq_scale, sizeof(uint32_t));
    }

    std::string model_path;
    if (model != nullptr) {
        md.arch  = llama_model_arch_name(model);
        md.ftype = llama_ftype_name(llama_model_ftype(model));
        model_path = params.model.path;
        if (!model_path.empty()) {
            // stat only - never open the (multi-GB) model file
            std::error_code ec;
            const auto st = fs::status(model_path, ec);
            if (!ec && st.type() == fs::file_type::regular) {
                md.model_file_size  = static_cast<int64_t>(fs::file_size(model_path, ec));
                if (!ec) {
                    md.model_file_mtime = std::chrono::duration_cast<std::chrono::seconds>(
                        fs::last_write_time(model_path, ec).time_since_epoch()).count();
                    if (ec) {
                        md.model_file_mtime = -1;
                    }
                }
            }
        }
    }
    if (model == nullptr) {
        SRV_WRN("%s", "kv-chain[storage]: no model handle, root hash will not include arch/ftype/file identity");
    }

    // canonical serialization: every field, length-prefixed, in struct order.
    uint64_t h = 1469598103934665603ULL;
    h = hash_le(h, static_cast<uint64_t>(md.format_version), 4);
    h = hash_le(h, static_cast<uint64_t>(md.chunk_size),       4);
    h = hash_le(h, static_cast<uint64_t>(md.model_file_size),  8);
    h = hash_le(h, static_cast<uint64_t>(md.model_file_mtime), 8);
    h = hash_str_field(h, md.arch);
    h = hash_str_field(h, md.ftype);
    h = hash_le(h, static_cast<uint64_t>(md.type_k),           4);
    h = hash_le(h, static_cast<uint64_t>(md.type_v),           4);
    h = hash_le(h, static_cast<uint64_t>(md.rope_scaling_type),4);
    h = hash_le(h, md.rope_freq_base_bits,  4);
    h = hash_le(h, md.rope_freq_scale_bits, 4);
    h = hash_le(h, md.n_seq_max,            4);
    h = hash_str_field(h, model_path);

    root_hash_ = h;
    {
        float rope_freq_base  = 0.0f;
        float rope_freq_scale = 0.0f;
        std::memcpy(&rope_freq_base,  &md.rope_freq_base_bits,  sizeof(float));
        std::memcpy(&rope_freq_scale, &md.rope_freq_scale_bits, sizeof(float));
        SRV_DBG("kv-chain[storage]: metadata: version=%d chunk_size=%d model='%s' size=%lld mtime=%lld arch='%s' ftype='%s' type_k=%d type_v=%d rope=(%d,%.6g,%.6g) n_seq_max=%d\n",
                md.format_version, md.chunk_size, model_path.c_str(),
                (long long) md.model_file_size, (long long) md.model_file_mtime,
                md.arch.c_str(), md.ftype.c_str(),
                (int) md.type_k, (int) md.type_v,
                md.rope_scaling_type, rope_freq_base, rope_freq_scale,
                (int) md.n_seq_max);
    }
}

// byte size of a chunk file header holding n_tokens token ids: magic, version, hash32,
// n_tokens, pos_lo, pos_hi, exit_fp_hi, exit_fp_lo, the ids, then the three blob sizes
// (attn/comp/tail). pos_lo/pos_hi are the chunk's KV position window (once an image is in
// the prompt chunks are no longer uniform-width, so the window is stored per chunk rather
// than derived as k*ubs -- that derivation was the source of the desync). exit_fp is the
// cumulative block-chain fingerprint of the state at this chunk's end.
static size_t kv_chunk_header_size(size_t n_tokens) {
    return 8 * sizeof(uint32_t) + sizeof(llama_token) * n_tokens + 3 * sizeof(uint32_t);
}

// save one chunk covering [pos_lo, pos_hi) into ONE file.
//
// the three blobs are dumped straight into one pinned staging buffer at their own
// offsets (the state API takes a caller-supplied dst pointer, so no intermediate
// copy is needed), then the whole region goes to disk with a single overlapped
// WriteFile + rename - issued one stage later by publish_pending_copies(), once
// the async copies' event has fired. that makes a chunk all-or-nothing: a chunk
// file either has all three slices or does not exist, so the restore side needs
// no per-slice presence bookkeeping.
//
// the dump itself is async (llama_state_seq_get_data_*_async): no
// ctx->synchronize(), the GPU pipeline keeps running while the KV bytes stream
// into the pinned staging, and the copies are stream-ordered right after this
// ubatch's kernels, so they capture exactly the post-ubatch state.
//
// the tail blob is TAIL_ONLY, not FULL (flags=0): on a DSV4 cache a FULL-mode blob
// starts with kv_raw and its state_read clears kv_raw first, which would wipe the
// per-token rows restored from the attn slice (loaded after this). on DSV4
// TAIL_ONLY is the compressor RING states ONLY (no kv_raw, no comp): the comp K
// caches are the per-chunk COMP_ONLY slice, additive across chunks like the attn
// rows, so the tail stays constant-size. see llama-kv-cache-dsv4.cpp
// DSV4_STATE_MODE_TAIL / DSV4_STATE_MODE_COMP.
//
// reset the per-prompt block-chain fingerprint state. call once at prompt arrival,
// before load_prefix() and before the first save()/fold_glue(). flush() first so a
// chunk written moments ago (same session) is on disk and seen as a hit, not a miss.
void kv_chain_store::begin_prompt() {
    flush();
    running_fp_     = root_hash_;
    boundary_index_ = 0;
}

// tail-stride: the tail is last-write-wins and constant-size, so on off-stride
// chunks (write_tail == false) it is skipped entirely (tail_size = 0 in the
// header); load_prefix restores up to the last chunk that carries one.
bool kv_chain_store::save(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi,
                          uint64_t chunk_hash, const llama_tokens & chunk_tokens,
                          bool is_last /* = false */) {
    if (!enabled() || ctx == nullptr || chunk_tokens.empty() || pos_hi <= pos_lo) {
        return false;
    }

    // advance the pipeline oldest-first BEFORE forming the name: publishing the
    // previous chunk folds its bytes into running_fp_, so this chunk's name (and
    // tail-stride index) see the fully-updated fingerprint. with a chunk every ~10 s
    // and sub-second dumps/writes both waits are already satisfied; they are safety
    // margins (never overwrite a buffer the kernel/GPU is still using), not the bottleneck.
    if (!reap_pending_write()) {
        return false;
    }
    if (!publish_pending_copies()) {
        return false;
    }

    // the file name binds the input hash AND the running fp: a chunk whose content
    // matches but whose KV state differs (a different preceding image) has a
    // different name, so it is a clean miss -- no header read to reject it.
    const fs::path dir  = fs::path(root_dir);
    const fs::path file = dir / (chunk_key_str(chunk_hash, running_fp_) + ".kvcache");
    if (fs::exists(file)) {
        // hit: adopt the stored exit_fp so running_fp_ advances exactly as it did when
        // this chunk was first saved -- the next block's name stays consistent.
        kv_chunk_header_info info{};
        info.sizes[0] = info.sizes[1] = info.sizes[2] = 0;
        if (probe_chunk_header(file, info)) {
            running_fp_ = info.exit_fp;
        }
        boundary_index_++;
        return false; // already saved (idempotent): nothing to dump or write
    }

    // tail-stride: the tail slice is last-write-wins and constant-size (~113 MiB on
    // the hybrid model), so dumping + writing it at every boundary is pure overhead
    // for a linear prompt. write it every tail_stride-th chunk, plus the last chunk
    // of each prompt (a restore always needs a tail).
    const size_t chunk_index = boundary_index_;
    const bool write_tail = is_last || tail_stride_ <= 1 || (chunk_index % (size_t) tail_stride_ == 0);

    const size_t attn_size =
        llama_state_seq_get_size_window_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY, pos_lo, pos_hi);
    // the tail snapshot taken at this boundary (skipped on off-stride chunks)
    const size_t tail_size = write_tail ?
        llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY) : 0;
    // this window's comp rows (dsv4 only), additive across chunks like the attn rows
    const size_t comp_size = has_comp_
        ? llama_state_seq_get_size_window_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_COMP_ONLY, pos_lo, pos_hi)
        : 0;
    if (attn_size == 0 || (write_tail && tail_size == 0) || (has_comp_ && comp_size == 0)) {
        SRV_WRN("kv-chain[storage]: failed to size chunk window [%d, %d) (attn=%zu comp=%zu tail=%zu)\n",
                (int) pos_lo, (int) pos_hi, attn_size, comp_size, tail_size);
        return false;
    }

    const size_t hdr_size = kv_chunk_header_size(chunk_tokens.size());
    const size_t off_attn = hdr_size;
    const size_t off_comp = off_attn + attn_size;
    const size_t off_tail = off_comp + comp_size;
    const size_t total    = off_tail + tail_size;
    // one WriteFile takes a DWORD length; a chunk is ~200 MiB, so this is a
    // sanity check on the geometry rather than a real limit.
    if (total > 0xFFFFffffULL) {
        SRV_WRN("kv-chain[storage]: chunk too large to write in one call (%zu bytes)\n", total);
        return false;
    }

    // one pinned region for the whole chunk: header + attn + comp + tail.
    copies.blob.reset(total, &kb_pool);
    if (copies.blob.data() == nullptr || copies.blob.size() != total) {
        SRV_WRN("kv-chain[storage]: failed to allocate %zu bytes of staging\n", total);
        copies.blob.reset(0, nullptr);
        return false;
    }
    uint8_t * base = copies.blob.data();

    // exit_fp is a placeholder here: the state bytes are dumped async and only final at
    // publish time, so publish_pending_copies() folds them and patches the header in place.
    write_header(base, chunk_hash, chunk_tokens, pos_lo, pos_hi, /* exit_fp */ 0,
                 (uint32_t) attn_size, (uint32_t) comp_size, (uint32_t) tail_size);

    // the state API takes a caller-supplied dst pointer, so each slice is dumped
    // straight into its own offset of the staging region - no intermediate copy.
    // ATTN_ONLY (not FULL_ONLY, which would also emit the compressor rings that
    // belong in the tail slice), then the tail, then comp.
    if (writer_ensure_event(ctx)) {
        // enqueue the dumps WITHOUT any ctx->synchronize(): the copies are issued
        // on the GPU stream, stream-ordered right after this ubatch's kernels, so
        // they capture exactly the post-ubatch state (the recr tail included - the
        // next ubatch's kernels are queued behind them in the same stream), and
        // one event is recorded at the end. the GPU pipeline never drains here.
        if (llama_state_seq_get_data_window_async(ctx, base + off_attn, attn_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY, pos_lo, pos_hi, copy_event) != attn_size) {
            SRV_WRN("kv-chain[storage]: failed to dump attn window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
            copies.blob.reset(0, nullptr);
            return false;
        }
        if (write_tail && llama_state_seq_get_data_async(ctx, base + off_tail, tail_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY, copy_event) != tail_size) {
            SRV_WRN("%s", "kv-chain[storage]: failed to dump tail state");
            copies.blob.reset(0, nullptr);
            return false;
        }
        if (has_comp_ && llama_state_seq_get_data_window_async(ctx, base + off_comp, comp_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_COMP_ONLY, pos_lo, pos_hi, copy_event) != comp_size) {
            SRV_WRN("kv-chain[storage]: failed to dump comp window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
            copies.blob.reset(0, nullptr);
            return false;
        }
        copies.async = true;
    } else {
        // synchronous fallback (no event support on the KV device): the old
        // blocking dump; publish_pending_copies() then skips the event wait.
        if (llama_state_seq_get_data_window_ext(ctx, base + off_attn, attn_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY, pos_lo, pos_hi) != attn_size) {
            SRV_WRN("kv-chain[storage]: failed to dump attn window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
            copies.blob.reset(0, nullptr);
            return false;
        }
        if (write_tail && llama_state_seq_get_data_ext(ctx, base + off_tail, tail_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_TAIL_ONLY) != tail_size) {
            SRV_WRN("%s", "kv-chain[storage]: failed to dump tail state");
            copies.blob.reset(0, nullptr);
            return false;
        }
        if (has_comp_ && llama_state_seq_get_data_window_ext(ctx, base + off_comp, comp_size, seq_id,
                LLAMA_STATE_SEQ_FLAGS_COMP_ONLY, pos_lo, pos_hi) != comp_size) {
            SRV_WRN("kv-chain[storage]: failed to dump comp window [%d, %d)\n", (int) pos_lo, (int) pos_hi);
            copies.blob.reset(0, nullptr);
            return false;
        }
        copies.async = false;
    }

    if (limit_bytes > 0 && total_bytes_cur + total > limit_bytes) {
        evict_oldest(total);
    }

    static std::atomic<int> s_pin_logged { 0 };
    if (s_pin_logged.fetch_add(1) < 2) {
        SRV_DBG("kv-chain[storage]: chunk staging %zu bytes staging=%s\n", total,
                copies.blob.pinned() ? "PINNED" : "pageable");
    }

    // park the chunk: the overlapped WriteFile is issued by publish_pending_copies()
    // at the NEXT save() (or flush()), once the copy event has fired - by then the
    // wait is instant, so neither the GPU dump nor the disk write ever stalls the
    // prefill thread.
    copies.tmp      = fs::path(file.string() + ".tmp");
    copies.file     = file;
    copies.bytes    = total;
    copies.pending  = true;
    copies.entry_fp = running_fp_; // fp at this chunk's start; publish folds the bytes -> exit_fp
    copies.off_attn = off_attn;
    copies.off_comp = off_comp;
    copies.off_tail = off_tail;
    copies.sz_attn  = attn_size;
    copies.sz_comp  = comp_size;
    copies.sz_tail  = tail_size;
    boundary_index_++;

    total_bytes_cur += total;
    SRV_DBG("kv-chain[storage]: staged chunk key=%s (content=%s entry_fp=%s) tokens=[%d..%d) "
            "(attn %.1f MiB, cm %.1f MiB, tail %.1f MiB, dump=%s)\n",
            chunk_key_str(chunk_hash, copies.entry_fp).c_str(), hash_str(chunk_hash).c_str(),
            hash_str(copies.entry_fp).c_str(), (int) pos_lo, (int) pos_hi,
            (double) attn_size / (1024.0*1024.0), (double) comp_size / (1024.0*1024.0),
            (double) tail_size / (1024.0*1024.0), copies.async ? "async" : "sync");
    return true;
}

// advances running_fp_ over an uncached block's live state (image / text-partial glue).
// the block was just decoded into [pos_lo, pos_hi); its attn/comp/tail bytes are folded in
// synchronously (glue is small and rare, so a blocking dump is fine). no file is written.
// a different image folds to a different running_fp_, so the next cached block's name
// differs -> a clean miss -> it is recomputed. this is what certifies cached text after an image.
void kv_chain_store::fold_glue(llama_context * ctx, llama_seq_id seq_id, llama_pos pos_lo, llama_pos pos_hi) {
    if (!enabled() || ctx == nullptr || pos_hi <= pos_lo) {
        return;
    }
    // publish any pending chunk first so running_fp_ is current before folding glue on top.
    (void) reap_pending_write();
    (void) publish_pending_copies();

    const size_t attn_size =
        llama_state_seq_get_size_window_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY, pos_lo, pos_hi);
    // NOTE: the tail (recurrent) snapshot is deliberately NOT folded for glue. The tail is a
    // GLOBAL, last-write-wins snapshot whose bytes depend on WHEN it is sampled -- and the async
    // image decode means the hook fires at a different position on the cold (first) run than on a
    // warm (restored) run (measured: cold folds the image tail at pos 22041, warm at 16921), so a
    // tail-inclusive glue fold diverged and the post-image chunk was a permanent miss. The attn
    // (and comp) window [pos_lo, pos_hi) is position-scoped and immutable once written, so folding
    // only those is time-invariant -- identical on cold and warm for identical inputs -- while still
    // certifying the glue: a different image yields different attn rows -> different running_fp ->
    // the next cached chunk's name differs -> clean miss. The prefix's cached chunks already carry
    // the recurrent tail (restored verbatim), so the glue needs no tail of its own.
    const size_t comp_size = has_comp_
        ? llama_state_seq_get_size_window_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_COMP_ONLY, pos_lo, pos_hi)
        : 0;
    if (attn_size == 0 || (has_comp_ && comp_size == 0)) {
        SRV_WRN("kv-chain[storage]: fold_glue: failed to size glue window [%d, %d) (attn=%zu comp=%zu)\n",
                (int) pos_lo, (int) pos_hi, attn_size, comp_size);
        return;
    }
    std::vector<uint8_t> attn(attn_size), comp(comp_size);
    if (llama_state_seq_get_data_window_ext(ctx, attn.data(), attn_size, seq_id,
            LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY, pos_lo, pos_hi) != attn_size) {
        SRV_WRN("%s", "kv-chain[storage]: fold_glue: attn dump failed");
        return;
    }
    if (has_comp_ && llama_state_seq_get_data_window_ext(ctx, comp.data(), comp_size, seq_id,
            LLAMA_STATE_SEQ_FLAGS_COMP_ONLY, pos_lo, pos_hi) != comp_size) {
        SRV_WRN("%s", "kv-chain[storage]: fold_glue: comp dump failed");
        return;
    }
    running_fp_ = fold_state_fp(running_fp_, attn.data(), attn_size,
                                comp_size ? comp.data() : nullptr, comp_size,
                                /* tail */ nullptr, /* tail_size */ 0);
    SRV_DBG("kv-chain[storage]: fold_glue [%d..%d) -> running_fp=%s\n",
            (int) pos_lo, (int) pos_hi, hash_str(running_fp_).c_str());
}

// lazily create the copy event on ctx's KV device (first save). false => the
// device has no event support and save() falls back to the synchronous dump.
bool kv_chain_store::writer_ensure_event(llama_context * ctx) {
    if (copy_event_tried) {
        return copy_event != nullptr;
    }
    copy_event_tried = true;
    ggml_backend_dev_t dev = llama_state_seq_get_device(ctx);
    if (dev == nullptr) {
        SRV_WRN("%s", "kv-chain[storage]: no KV device, async state dump disabled\n");
        return false;
    }
    copy_event = ggml_backend_event_new(dev); // nullptr when events are unsupported
    if (copy_event == nullptr) {
        SRV_WRN("%s", "kv-chain[storage]: KV device has no event support, async state dump disabled\n");
        return false;
    }
    SRV_INF("kv-chain[storage]: async state dump enabled (copy event on dev %s)\n", ggml_backend_dev_name(dev));
    return true;
}

// publish the chunk whose copies are pending: wait the copy event (instant in
// steady state - the copies were enqueued a full chunk period ago), then issue
// ONE overlapped WriteFile of the whole staging region. the kernel reads the
// blob asynchronously from here on; it stays owned (by write_in_flight) until
// the write is reaped at the next save()/flush().
bool kv_chain_store::publish_pending_copies() {
    if (!copies.pending) {
        return true;
    }
    // the write slot is empty by construction (callers reap first); defend anyway
    if (write_in_flight.pending && !reap_pending_write()) {
        return false;
    }
    if (copies.async && copy_event != nullptr) {
        // NOTE: waits ONLY on this chunk's copies, recorded before the current
        // ubatch's kernels were enqueued - it never waits on in-flight compute.
        ggml_backend_event_synchronize(copy_event);
    }

    const fs::path tmp   = copies.tmp;
    const fs::path file  = copies.file;
    const uint64_t bytes = copies.bytes;
    uint8_t *      base  = copies.blob.data();

    // the state bytes are final now (the copy event has fired), so fold them into the running
    // block-chain fingerprint and patch exit_fp into the header in place. exit_fp occupies
    // the two uint32 slots right after pos_hi (fields[6..7], byte offset 6*sizeof(uint32)).
    // the published chunk is the most recent one the pipeline held, so exit_fp becomes the
    // store's current running_fp_ -- the next block's name (save/fold_glue) builds on it.
    {
        const uint8_t * attn = base + copies.off_attn;
        const uint8_t * comp = copies.sz_comp ? base + copies.off_comp : nullptr;
        const uint8_t * tail = copies.sz_tail ? base + copies.off_tail : nullptr;
        const uint64_t exit_fp = fold_state_fp(copies.entry_fp, attn, copies.sz_attn,
                                               comp, copies.sz_comp, tail, copies.sz_tail);
        const uint32_t hi = static_cast<uint32_t>(exit_fp >> 32);
        const uint32_t lo = static_cast<uint32_t>(exit_fp & 0xFFFFFFFFu);
        std::memcpy(base + 6 * sizeof(uint32_t), &hi, sizeof(hi));
        std::memcpy(base + 7 * sizeof(uint32_t), &lo, sizeof(lo));
        running_fp_ = exit_fp;
    }

    // undoes the accounting done at park time when the publish fails
    const auto unpark = [&]() {
        copies.blob.reset(0, nullptr);
        copies = pending_copies{};
        total_bytes_cur = total_bytes_cur > bytes ? total_bytes_cur - bytes : 0;
    };

#ifdef _WIN32
    // one overlapped WriteFile of the whole region: the prefill thread does not
    // block on the disk, it just parks the handle and reaps it on the next chunk.
    HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        SRV_ERR("kv-chain[storage]: failed to open '%s' for writing (gle=%lu)\n",
                tmp.string().c_str(), GetLastError());
        unpark();
        return false;
    }
    write_in_flight = chunk_write{};
    write_in_flight.tmp    = tmp;
    write_in_flight.file   = file;
    write_in_flight.bytes  = bytes;
    write_in_flight.handle = h;
    write_in_flight.ov = {};
    write_in_flight.ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (write_in_flight.ov.hEvent == nullptr) {
        SRV_ERR("kv-chain[storage]: CreateEvent failed (gle=%lu)\n", GetLastError());
        CloseHandle(h);
        write_in_flight = chunk_write{};
        unpark();
        return false;
    }
    DWORD done = 0;
    const BOOL ok = WriteFile(h, base, (DWORD) bytes, &done, &write_in_flight.ov);
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
        SRV_ERR("kv-chain[storage]: overlapped write failed for '%s' (gle=%lu)\n",
                tmp.string().c_str(), GetLastError());
        CloseHandle(write_in_flight.ov.hEvent);
        CloseHandle(h);
        write_in_flight = chunk_write{};
        std::error_code ec;
        fs::remove(tmp, ec);
        unpark();
        return false;
    }
    write_in_flight.blob    = std::move(copies.blob); // the kernel reads it until reaped
    write_in_flight.pending = true;
    copies = pending_copies{};
#else
    if (!write_staging_sync(tmp, file, copies.blob, bytes)) {
        unpark();
        return false;
    }
    copies.blob.reset(0, nullptr);
    copies = pending_copies{};
#endif
    return true;
}

// serialise the chunk header into dst; returns the number of bytes written.
size_t kv_chain_store::write_header(uint8_t * dst, uint64_t chunk_hash, const llama_tokens & tokens,
                                    llama_pos pos_lo, llama_pos pos_hi, uint64_t exit_fp,
                                    uint32_t attn_size, uint32_t comp_size, uint32_t tail_size) const {
    size_t o = 0;
    const uint32_t fields[8] = { KV_CHAIN_MAGIC, KV_CHAIN_VERSION,
                                 static_cast<uint32_t>(chunk_hash), static_cast<uint32_t>(tokens.size()),
                                 static_cast<uint32_t>(pos_lo),     static_cast<uint32_t>(pos_hi),
                                 static_cast<uint32_t>(exit_fp >> 32), static_cast<uint32_t>(exit_fp & 0xFFFFFFFFu) };
    std::memcpy(dst + o, fields, sizeof(fields)); o += sizeof(fields);
    if (!tokens.empty()) {
        std::memcpy(dst + o, tokens.data(), sizeof(llama_token) * tokens.size());
        o += sizeof(llama_token) * tokens.size();
    }
    const uint32_t sizes[3] = { attn_size, comp_size, tail_size };
    std::memcpy(dst + o, sizes, sizeof(sizes)); o += sizeof(sizes);
    return o;
}

// non-Windows fallback: write the staging region with ofstream, then rename.
bool kv_chain_store::write_staging_sync(const fs::path & tmp, const fs::path & file, const dump_blob & blob, size_t bytes) {
    std::error_code ec;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            SRV_ERR("kv-chain[storage]: failed to open '%s' for writing\n", tmp.string().c_str());
            return false;
        }
        f.write(reinterpret_cast<const char *>(blob.data()), bytes);
        if (!f) {
            SRV_ERR("kv-chain[storage]: write failed for '%s'\n", tmp.string().c_str());
            fs::remove(tmp, ec);
            return false;
        }
        f.close();
    }
    fs::rename(tmp, file, ec);
    if (ec) {
        SRV_ERR("kv-chain[storage]: rename failed for '%s': %s\n", file.string().c_str(), ec.message().c_str());
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

// finish the chunk write issued by save(): wait for the overlapped WriteFile,
// close the handle and rename .tmp -> final. with one chunk every ~10 s and a
// sub-second write, the wait is already satisfied and returns immediately.
// always clears the in-flight slot (and releases staging) before returning.
bool kv_chain_store::reap_pending_write() {
    if (!write_in_flight.pending) {
        return true;
    }
    // NOTE: the OVERLAPPED and the handle are used IN PLACE - the kernel holds a
    // pointer to the OVERLAPPED and writes its completion status there, so the
    // struct must not be copied or moved while the I/O is in flight. only the
    // bookkeeping fields are copied out.
    const fs::path tmp   = write_in_flight.tmp;
    const fs::path file  = write_in_flight.file;
    const uint64_t bytes = write_in_flight.bytes;

    bool ok = false;
#ifdef _WIN32
    DWORD done = 0;
    // bWait=TRUE: in practice the transfer already finished, so this returns at
    // once; it only actually blocks if the disk is slower than prefill.
    ok = GetOverlappedResult(write_in_flight.handle, &write_in_flight.ov, &done, TRUE) != 0
         && done == bytes;
    if (!ok) {
        SRV_ERR("kv-chain[storage]: chunk write failed for '%s' (gle=%lu, %lu of %llu bytes)\n",
                tmp.string().c_str(), GetLastError(), done, (unsigned long long) bytes);
    }
    CloseHandle(write_in_flight.handle);
    CloseHandle(write_in_flight.ov.hEvent);
#else
    ok = true; // synchronous fallback: nothing pending
#endif
    write_in_flight = chunk_write{}; // clear the slot now that the I/O is finished

    std::error_code ec;
    if (ok) {
        fs::rename(tmp, file, ec);
        if (ec) {
            SRV_ERR("kv-chain[storage]: rename failed for '%s': %s\n",
                    file.string().c_str(), ec.message().c_str());
            ok = false;
        }
    }
    if (!ok) {
        fs::remove(tmp, ec); // never publish a partial chunk
        total_bytes_cur = total_bytes_cur > bytes ? total_bytes_cur - bytes : 0;
    } else {
        SRV_DBG("kv-chain[storage]: saved chunk %s (%.1f MiB)\n", file.filename().string().c_str(),
                (double) bytes / (1024.0*1024.0));
    }

    write_in_flight.blob.reset(0, nullptr); // release only now: the kernel was reading this buffer
    return ok;
}

// reads + validates one chunk file and returns ONE slice of it (attn/comp/tail).
// the header's token IDs must match expected_tokens verbatim (a collision/stale
// file is a clean miss, not a garbage restore). only the requested slice is read:
// the restore replays the attn slices of the whole chain first, so pulling the
// tail bytes on every chunk would re-read the same bytes over and over.
// returns false if missing/corrupt/token-mismatch.
bool kv_chain_store::read_chunk_slice(const fs::path & file, std::vector<uint8_t> & out_blob,
                                      const llama_tokens & expected_tokens, kv_chunk_slice slice,
                                      llama_pos * out_pos_lo /* = nullptr */, llama_pos * out_pos_hi /* = nullptr */,
                                      uint64_t * out_exit_fp /* = nullptr */) const {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    // magic/version/hash32/n_tokens/pos_lo/pos_hi/exit_fp_hi/exit_fp_lo + the three blob sizes
    const uint64_t min_size = kv_chunk_header_size(0);
    if (file_size < min_size) {
        return false;
    }
    f.seekg(0, std::ios::beg);

    uint32_t hdr[8];
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) {
        return false;
    }
    const uint32_t magic   = hdr[0];
    const uint32_t version = hdr[1];
    // hdr[2] = hash32 (the file name; not re-verified - the name IS the hash)
    const uint32_t n_tokens = hdr[3];
    const llama_pos pos_lo  = (llama_pos) hdr[4];
    const llama_pos pos_hi  = (llama_pos) hdr[5];
    const uint64_t  exit_fp = ((uint64_t) hdr[6] << 32) | (uint64_t) hdr[7];
    // no trailing checksum: verifying one costs a full pass over every (multi-MiB)
    // chunk file, which dominated restore time; we trust the storage device.
    if (magic != KV_CHAIN_MAGIC || version != KV_CHAIN_VERSION) {
        SRV_WRN("kv-chain[storage]: bad magic/version in %s (magic=%08x version=%u), ignoring\n",
                file.string().c_str(), magic, version);
        return false;
    }
    // bound n_tokens before allocating
    if (n_tokens == 0 || (uint64_t) n_tokens > (file_size - min_size) / sizeof(llama_token)) {
        return false;
    }
    std::vector<llama_token> file_tokens(n_tokens);
    f.read(reinterpret_cast<char *>(file_tokens.data()), sizeof(llama_token) * n_tokens);
    if (!f) {
        return false;
    }
    uint32_t sizes[3]; // attn, comp, tail
    f.read(reinterpret_cast<char *>(sizes), sizeof(sizes));
    if (!f) {
        return false;
    }
    const uint64_t hdr_len = kv_chunk_header_size(n_tokens);
    const uint64_t total   = (uint64_t) sizes[0] + sizes[1] + sizes[2];
    if (hdr_len + total != file_size) {
        SRV_WRN("kv-chain[storage]: size mismatch in %s (header %llu + blobs %u/%u/%u != file %llu), ignoring\n",
                file.string().c_str(), (unsigned long long) hdr_len,
                sizes[0], sizes[1], sizes[2], (unsigned long long) file_size);
        return false;
    }
    // the token IDs must match the prompt verbatim
    if (file_tokens.size() != expected_tokens.size() ||
        std::memcmp(file_tokens.data(), expected_tokens.data(), sizeof(llama_token) * n_tokens) != 0) {
        SRV_WRN("kv-chain[storage]: token mismatch in %s (n=%u vs %zu), treating as miss\n",
                file.string().c_str(), n_tokens, expected_tokens.size());
        return false;
    }
    if (out_pos_lo)  { *out_pos_lo  = pos_lo;  }
    if (out_pos_hi)  { *out_pos_hi  = pos_hi;  }
    if (out_exit_fp) { *out_exit_fp = exit_fp; }
    // the slice's offset within the body: attn | comp | tail, in that order
    const size_t idx = (slice == kv_chunk_slice::attn) ? 0
                     : (slice == kv_chunk_slice::comp) ? 1 : 2;
    uint64_t off = hdr_len;
    for (size_t i = 0; i < idx; ++i) {
        off += sizes[i];
    }
    const uint64_t blob_size = sizes[idx];
    if (blob_size == 0) {
        out_blob.clear();
        return false; // asking for a slice this file does not carry
    }
    f.seekg((std::streamoff) off, std::ios::beg);
    out_blob.resize(blob_size);
    f.read(reinterpret_cast<char *>(out_blob.data()), blob_size);
    if (!f) {
        return false;
    }
    return true;
}

// everything load_prefix / save() need from a chunk file without touching the
// (multi-MiB) blob bodies (see the struct declared near the top of the file).
static bool probe_chunk_header(const fs::path & file, kv_chunk_header_info & info) {
    std::error_code ec;
    const uint64_t file_size = fs::file_size(file, ec);
    if (ec || file_size < kv_chunk_header_size(0)) {
        return false;
    }
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    uint32_t hdr[8];
    f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
    if (!f) {
        return false;
    }
    if (hdr[0] != KV_CHAIN_MAGIC || hdr[1] != KV_CHAIN_VERSION) {
        return false;
    }
    info.n_tokens = hdr[3];
    info.pos_lo   = (llama_pos) hdr[4];
    info.pos_hi   = (llama_pos) hdr[5];
    info.exit_fp  = ((uint64_t) hdr[6] << 32) | (uint64_t) hdr[7];
    if (info.n_tokens == 0 || (uint64_t) info.n_tokens > (file_size - kv_chunk_header_size(0)) / sizeof(llama_token)) {
        return false;
    }
    // the blob sizes sit right after the token ids
    f.seekg((std::streamoff) (sizeof(hdr) + sizeof(llama_token) * info.n_tokens), std::ios::beg);
    f.read(reinterpret_cast<char *>(info.sizes), 3 * sizeof(uint32_t));
    if (!f) {
        return false;
    }
    return true;
}

void kv_chain_store::evict_oldest(uint64_t need_bytes) {
    std::error_code ec;
    // delete oldest (by mtime) until we have room. no tree-integrity checks:
    // an evicted file just shortens the chain on the next restore.
    const fs::path cache_dir = fs::path(root_dir);
    struct ent { fs::path p; uint64_t size; std::filesystem::file_time_type mtime; };
    std::vector<ent> all;
    for (const auto & e : fs::directory_iterator(cache_dir, ec)) {
        if (!e.is_regular_file()) {
            continue;
        }
        const auto ext = e.path().extension();
        if (ext == ".kvcache") {
            all.push_back({ e.path(), (uint64_t) e.file_size(), e.last_write_time(ec) });
        }
    }
    std::sort(all.begin(), all.end(), [](const ent & a, const ent & b) {
        return a.mtime < b.mtime;
    });
    for (auto & e : all) {
        if (limit_bytes == 0 || total_bytes_cur + need_bytes <= limit_bytes) {
            break;
        }
        if (fs::remove(e.p, ec)) {
            total_bytes_cur -= e.size;
            SRV_INF("kv-chain[storage]: evicted %s (%.1f MiB)\n", e.p.filename().string().c_str(), e.size / (1024.0*1024.0));
        }
    }
}

// sets atime+mtime of every file to "now"; a per-file failure is non-fatal.
static void touch_chain_files(const std::vector<fs::path> & files) {
    if (files.empty()) {
        return;
    }
#ifdef __linux__
    const struct timespec now[2] = { { 0, UTIME_NOW }, { 0, UTIME_NOW } };
    for (const auto & p : files) {
        (void) utimensat(AT_FDCWD, p.c_str(), now, 0);
    }
#else
    (void) files; // no-op on non-linux
#endif
}

// shared forward walk for load_prefix()/load_next(). starts at start_index using the store's
// current running_fp_ and advances it over every hit. stops at the first missing name (the chain
// break). tail-stride: the returned run is truncated to end on a chunk that carries a tail slice,
// because the recurrent state is last-write-wins -- attn rows past the last tail would be
// orphaned. on success running_fp_ is left at that last tail-carrying chunk's exit_fp.
std::vector<kv_chain_chunk> kv_chain_store::walk_chain(const std::vector<uint64_t> & chain_hashes,
                                                      const std::vector<llama_tokens> & chunk_identity,
                                                      size_t start_index, size_t * n_cells) {
    std::vector<kv_chain_chunk> chunks;
    *n_cells = 0;
    if (!enabled() || chain_hashes.empty() || chain_hashes.size() != chunk_identity.size()
        || start_index >= chain_hashes.size()) {
        return chunks;
    }

    const fs::path dir = fs::path(root_dir);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return chunks;
    }

    const size_t n_chunks = chain_hashes.size();

    std::vector<fs::path>             files;
    std::vector<kv_chunk_header_info> infos;
    files.reserve(n_chunks - start_index);
    infos.reserve(n_chunks - start_index);

    uint64_t fp = running_fp_;   // seeded by begin_prompt()/the previous walk segment
    size_t   last_tail = 0;      // 1-based count (within `files`) of the last chunk carrying a tail
    for (size_t k = start_index; k < n_chunks; ++k) {
        fs::path file = dir / (chunk_key_str(chain_hashes[k], fp) + ".kvcache");
        if (!fs::exists(file)) {
            // the break point is the interesting part of every short restore: name = f(content_hash,
            // running_fp). if the content hash is one that was never saved, the PROMPT diverged here
            // (not a cache fault); if it was saved under a different running_fp, the state diverged.
            SRV_INF("kv-chain[storage]: walk broke at chunk %zu: name=%s not on disk (content=%s running_fp=%s)\n",
                    k, chunk_key_str(chain_hashes[k], fp).c_str(), hash_str(chain_hashes[k]).c_str(),
                    hash_str(fp).c_str());
            break; // chain break
        }
        kv_chunk_header_info info{};
        info.sizes[0] = info.sizes[1] = info.sizes[2] = 0;
        if (!probe_chunk_header(file, info)) {
            SRV_WRN("kv-chain[storage]: walk_chain: unreadable chunk header at chunk %zu, discarding restore\n", k);
            return chunks;
        }
        fp = info.exit_fp;
        if (info.sizes[2] > 0) {
            last_tail = files.size() + 1;
        }
        files.push_back(std::move(file));
        infos.push_back(info);
    }

    const size_t n_found = files.size();
    if (n_found == 0 || last_tail == 0) {
        SRV_DBG("kv-chain[storage]: %zu cached chunks from index %zu, no usable chain (%zu found on disk)\n",
                n_chunks - start_index, start_index, n_found);
        return chunks;
    }

    // the restore ends at the last tail-carrying chunk; the running fp must be the fp at THAT
    // boundary (not the last found), so a computed chunk after the break is named consistently.
    const size_t usable = last_tail;
    running_fp_ = infos[usable - 1].exit_fp;
    files.resize(usable);
    infos.resize(usable);

    // validate the TAIL slice of the last chunk up front: it is a single fixed-size object with
    // no earlier fallback, so a bad tail discards the whole restore - and the corrupt file is
    // deleted so it is not re-read (and re-deleted) on every restore. the attn/comp slices are
    // validated by the caller as it replays them.
    const fs::path & file_tail = files[usable - 1];
    {
        std::vector<uint8_t> tail_blob;
        if (!read_chunk_slice(file_tail, tail_blob, chunk_identity[start_index + usable - 1],
                              kv_chunk_slice::tail)) {
            SRV_WRN("kv-chain[storage]: walk_chain: tail read failed/mismatch at chunk %zu, discarding entire restore\n",
                    start_index + usable - 1);
            std::error_code dec;
            if (fs::exists(file_tail, dec) && fs::remove(file_tail, dec)) {
                SRV_WRN("kv-chain[storage]: walk_chain: deleted corrupt chunk file %s\n",
                        file_tail.filename().string().c_str());
            }
            return chunks;
        }
    }

    size_t cells = 0;
    for (size_t k = 0; k < usable; ++k) {
        kv_chain_chunk chunk;
        chunk.file    = files[k];
        chunk.tokens  = chunk_identity[start_index + k];
        chunk.pos_lo  = infos[k].pos_lo;
        chunk.pos_hi  = infos[k].pos_hi;
        chunk.exit_fp = infos[k].exit_fp;
        cells += chunk_identity[start_index + k].size();
        chunks.push_back(std::move(chunk));
    }
    *n_cells = cells;

    // touch every replayed chunk file: a future prompt may fork at one of the tail-carrying
    // boundaries, so keeping the chain warm is what makes those forks restorable.
    {
        std::vector<fs::path> to_touch;
        to_touch.reserve(chunks.size());
        for (const auto & c : chunks) {
            to_touch.push_back(c.file);
        }
        touch_chain_files(to_touch);
    }

    SRV_DBG("kv-chain[storage]: walk from chunk %zu: %zu found, last tail at +%zu, replay %zu chunks (%zu cells), running_fp=%s\n",
            start_index, n_found, usable - 1, chunks.size(), cells, hash_str(running_fp_).c_str());
    return chunks;
}

std::vector<kv_chain_chunk> kv_chain_store::load_prefix(const std::vector<uint64_t> & chain_hashes,
                                                        const std::vector<llama_tokens> & chunk_identity,
                                                        size_t * n_cells) {
    return walk_chain(chain_hashes, chunk_identity, 0, n_cells);
}

std::vector<kv_chain_chunk> kv_chain_store::load_next(const std::vector<uint64_t> & chain_hashes,
                                                      const std::vector<llama_tokens> & chunk_identity,
                                                      size_t start_index, size_t * n_cells) {
    return walk_chain(chain_hashes, chunk_identity, start_index, n_cells);
}
