// strixllama diagnostic: capture the SIZE of large / failing C++ allocations.
//
// Rationale: the live server hits `std::bad_alloc` ("bad allocation") at deep context with MTP,
// but the caught exception's what() carries no size, so we cannot see WHAT was requested. The
// depth-dependence (~58-66k tokens) points at a size that scales with context and is mis-computed.
//
// This overrides the global operator new for the llama DLL. It is OFF unless STRIX_ALLOC_LOG_MB is
// set. When enabled:
//   - every request >= threshold logs its size (so we see the ramp), and
//   - a FAILED request logs the exact size + (if STRIX_ALLOC_TRACE=1) a stack trace, then throws.
//
// Env:
//   STRIX_ALLOC_LOG_MB = integer MiB threshold (0 or unset = disabled). e.g. 64 logs >= 64 MiB.
//   STRIX_ALLOC_TRACE  = 1 to capture a stack trace on failure (and on the first few large allocs).
//   STRIX_ALLOC_MAXLOG = max lines to emit (default 200) to bound spam.
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <new>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

static const long    g_thr_bytes = []() -> long {
    const char * v = getenv("STRIX_ALLOC_LOG_MB");
    return v ? (long) atoi(v) * 1024L * 1024L : 0L;   // 0 => disabled
}();
static const bool    g_trace     = []() -> bool {
    const char * v = getenv("STRIX_ALLOC_TRACE");
    return v && atoi(v) != 0;
}();
static const long    g_maxlog    = []() -> long {
    const char * v = getenv("STRIX_ALLOC_MAXLOG");
    return v ? atol(v) : 200L;
}();
static long g_logged = 0;

static inline bool g_enabled() { return g_thr_bytes > 0; }

static void trace_once(const char * tag, size_t sz) {
#if defined(_WIN32)
    if (!g_trace) return;
    void * frames[48];
    USHORT n = CaptureStackBackTrace(/*skip=*/2, /*max=*/48, frames, nullptr);
    fprintf(stderr, "ALLOC %s sz=%.3f MiB stack(%u):\n", tag, sz / 1048576.0, (unsigned) n);
    for (USHORT i = 0; i < n; ++i) {
        fprintf(stderr, "  #%-2u %p\n", (unsigned) i, frames[i]);
    }
#else
    (void) tag; (void) sz;
#endif
}

static void * alloc_checked(size_t sz, const char * kind) {
    if (!g_enabled()) {
        void * p = std::malloc(sz ? sz : 1);
        if (!p) throw std::bad_alloc();
        return p;
    }
    const bool big = (long) sz >= g_thr_bytes;
    if (big && g_logged < g_maxlog) {
        ++g_logged;
        fprintf(stderr, "ALLOC %s req=%.3f MiB (%zu bytes) [log %ld/%ld]\n",
                kind, sz / 1048576.0, sz, g_logged, g_maxlog);
    }
    void * p = std::malloc(sz ? sz : 1);
    if (!p) {
        // this is the money line: the exact size that FAILED to allocate
        fprintf(stderr, "ALLOC_FAILED %s size=%.3f MiB (%zu bytes)  <== THIS is the bad_alloc size\n",
                kind, sz / 1048576.0, sz);
        trace_once("FAILED", sz);
        throw std::bad_alloc();
    }
    if (big && g_trace && g_logged <= 8) {
        trace_once("BIG-OK", sz);
    }
    return p;
}

} // namespace

// Global operator new overrides for this DLL. These intercept C++ allocations made from code
// compiled into the llama library (where llama_decode throws std::bad_alloc).
void * operator new  (size_t sz) { return alloc_checked(sz, "new"); }
void * operator new[](size_t sz) { return alloc_checked(sz, "new[]"); }

void operator delete  (void * p) noexcept             { std::free(p); }
void operator delete[](void * p) noexcept             { std::free(p); }
void operator delete  (void * p, size_t) noexcept     { std::free(p); }
void operator delete[](void * p, size_t) noexcept     { std::free(p); }

// nothrow variants route through the same path but must return null instead of throwing.
void * operator new  (size_t sz, const std::nothrow_t &) noexcept { return std::malloc(sz ? sz : 1); }
void * operator new[](size_t sz, const std::nothrow_t &) noexcept { return std::malloc(sz ? sz : 1); }
void operator delete  (void * p, const std::nothrow_t &) noexcept { std::free(p); }
void operator delete[](void * p, const std::nothrow_t &) noexcept { std::free(p); }
