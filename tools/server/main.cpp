// strixllama: the validated kernel-gate configuration is baked into the process environment at
// entry, so the launcher no longer has to set it and a pre-loaded parent environment cannot
// disturb the deployment.
//
// The wipe is a catch-all: the environment block is REBUILT (everything except LLAMA_*/GGML_*/
// STRIX_* carried over verbatim, the baked set appended, the block re-sorted the way the UCRT
// sorts it, then _environ redirected), so a gate variable that exists in the parent but is not in
// the baked list still cannot leak through. Every module in the package links the same shared
// UCRT (api-ms-win-crt-environment), so the redirect is seen by ggml-hip.dll / llama.dll /
// llama-common.dll too, not just by this executable.
//
// STRIX_ENV_BAKE (read from the INHERITED environment, before the wipe):
//   unset or 1 -> bake (the shipping default)
//   0          -> leave the environment exactly as the parent left it; this is how the dev A/B
//                 harnesses under harness/ keep steering gates through the environment
//   2          -> bake, and always print the summary line
//
// Every gate is read lazily at first use (each sits in a function-local `static const`), so doing
// this in main() is early enough; nothing in the shipped code reads a gate at DLL-load time.
// The original block is deliberately not freed: we hand our table to the CRT, which frees it at
// exit. A few KB, once, at process start.
#ifdef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char * const g_strixllama_env_gates[] = {
    // MMB: fused routed GLU + HC gate-mix + F32SPLIT
    "LLAMA_MMB=1",
    "LLAMA_MMB_MIN_T=512",
    "LLAMA_MMB_BF16W=1",
    "LLAMA_MMB_GLU=1",
    "LLAMA_MMB_TALL=2",
    "LLAMA_MMB_CACHE=4",
    "LLAMA_MMB_F32SPLIT=2",
    "LLAMA_MMB_HC16=0",   // REQUIRED: HC16=1 floods output with '/' on Windows/TheRock/Clang
    "LLAMA_MMB_SHADOW=2",
    "LLAMA_MMB_DOWN16=1",
    // HC: combine_norm shape + gate-mix + block/res/pack
    "LLAMA_HC_CN_SHAPE=1",
    "LLAMA_HC_GATEMIX=1",
    "LLAMA_HC_MIX_FUSE=1",
    "LLAMA_HC_BLK16=1",
    "LLAMA_HC_RES16=1",
    "LLAMA_HC_PACK_DI=1",
    // NORM / INDEXER / CONV fusions
    "LLAMA_NORM_GATED=1",
    "LLAMA_NORM_ROWS=1",
    "LLAMA_IDX_RELU_SUM=1",
    "LLAMA_PLE_CONV=1",
    "LLAMA_GDN_CONV=1",
    "LLAMA_MTP_QSA=1",
    // LLAMA_MTP_INDEX_SHARE is DELIBERATELY ABSENT (measured 2026-10-03): it looked like a
    // +3.12% win at temp0/p_min0.80, but that was an artifact of the conservative operating
    // point. At the shipping settings (temp 1.0, p_min 0.30) it is -5.75% at ~62K depth and
    // -14.10% at 179K: the draft attends a selection refreshed every 32 positions, sampling
    // proposes many more drafts and the stale selection corrupts them (acceptance 0.605 -> 0.547
    // at 62K, 0.717 -> 0.567 at 179K), so the wasted verify work outweighs the cheaper draft.
    // rulith's "-6.5% @212K, same acceptance" did NOT reproduce on this tree. Do NOT add it back
    // without re-measuring at the shipping temp/p_min.
    // QSA: block-sparse attention
    "LLAMA_QSA_SPARSE=1",
    "LLAMA_QSA_BLOCK_SELECTION=1",
    "LLAMA_QSA_COMPACT_METADATA=1",
    "LLAMA_QSA_DENSE_SHORTCUT=1",
    "LLAMA_QSA_DIRECT_INDICES=1",
    "LLAMA_QSA_FA_V3=1",
    "LLAMA_QSA_FUSE_EXPAND=1",
    "LLAMA_QSA_NO_DENSE_MASK=1",
    "LLAMA_QSA_PACK_KEYS=1",
    "LLAMA_QSA_PACK_VALUES=1",
    "LLAMA_QSA_SCORE_BOUNDS=1",
    "LLAMA_QSA_WHOLE_ATTN=1",
    // Fused QSA indexer-select: one kernel does the visibility mask + top-512, replacing the
    // REPEAT/SUB/STEP/LOG/ADD + TOP_K chain. Bit-identical at full scale; pp131k +4-5%. The
    // kernel default is OFF, so it has to be set here.
    "LLAMA_QSA_SELECT_FUSE=1",
    // Fused lightning-indexer score: emits the reduced [n_blocks,n_query] score directly instead
    // of the raw [n_blocks,n_idx_h(4),n_query] MUL_MAT output. The raw tensor's element count is
    // n_blocks*4*n_query; at deep ctx x large ubatch it exceeds INT32_MAX (2^31) and a 32-bit
    // index wraps negative -> heap corruption (0xc0000374) surfacing later as a random "launch
    // failure"/"tensor" error. The reduced form is 4x smaller and stays under the limit. REQUIRED
    // for ctx>=524k at ub8192; safe margin at 262k. PPL-equivalent to unfused (6.1717 vs 6.1756).
    "LLAMA_QSA_SCORE_WMMA=1",
    // Decode-side sparse gather: at decode batch 1-4 gather the top_k cells instead of reading the
    // whole f16 KV dense. Halves the decode-vs-depth slope.
    "LLAMA_QSA_DECODE_GATHER=1",
    // Multi-slot MTP verify graph reuse: keep one (result, scheduler) pair per verify shape so the
    // ~8900-node target-verify graph builds once and reuses. @46k +15.6%, byte-identical at temp=0.
    "STRIX_SPEC_REUSE_MULTI=1",
    // Cap the MTP draft ubatch so its buffers fit next to the target.
    "STRIX_SPEC_DRAFT_UBATCH=2048",
};

static int strixllama_env_cmp(const void * a, const void * b) {
    return _stricmp(*(char * const *) a, *(char * const *) b);
}

// the same prefixes the launcher used to clear, so the wipe scope is unchanged
static bool strixllama_env_is_gate(const char * kv) {
    return _strnicmp(kv, "LLAMA_", 6) == 0
        || _strnicmp(kv, "GGML_",  5) == 0
        || _strnicmp(kv, "STRIX_", 6) == 0;
}

// STRIXLLAMA_* is the DEV/TEST namespace: it is deliberately NOT wiped, so test hooks
// (STRIXLLAMA_LB_FAULT, _KB_FAULT, _DT_SYNC, _KB_DBG, ...) and the A/B harnesses that toggle a
// gate through the environment survive the bake. The cost is that a couple of DEFAULT-ON SHIPPING
// gates also live in this namespace and are therefore NOT re-asserted by the binary -- the
// launcher never sets them, so any value present at a BAKED launch came from the parent shell and
// silently changes behaviour (a leaked STRIXLLAMA_DRAFT_LEN=0 from a dev session disables a
// default-ON +8.9%-CJK feature in production, unfalsifiable from the log). We cannot bake or wipe
// these names without breaking the harnesses that toggle them via env (STRIX_ENV_BAKE stays ON for
// the 39 canonical gates those harnesses rely on), so instead we make a leak LOUD: the banner
// names any of these that is present. Add a name here when a new default-ON gate is put in the
// STRIXLLAMA_ namespace.
static const char * const g_strixllama_leak_watch[] = {
    "STRIXLLAMA_DRAFT_LEN",   // draft-length controller: unset = ON, 0 = OFF (fixed --spec-draft-n-max)
    "STRIXLLAMA_KB_RESTORE",  // block-key-cache restore: unset = ON, 0 = OFF
};
static const size_t n_leak_watch = sizeof(g_strixllama_leak_watch) / sizeof(g_strixllama_leak_watch[0]);

// case-insensitive compare of the NAME part (before '=') of an "NAME=value" env entry against a
// bare NAME.
static bool strixllama_env_name_eq(const char * kv, const char * name) {
    size_t nlen = strlen(name);
    const char * eq = strchr(kv, '=');
    if (!eq || (size_t) (eq - kv) != nlen) {
        return false;
    }
    return _strnicmp(kv, name, nlen) == 0;
}

static void strixllama_bake_env(void) {
    static const size_t n_gates = sizeof(g_strixllama_env_gates) / sizeof(g_strixllama_env_gates[0]);

    const char * mode = getenv("STRIX_ENV_BAKE");
    if (mode && atoi(mode) == 0) {
        return;
    }

    size_t n_in = 0;
    while (_environ && _environ[n_in]) {
        n_in++;
    }

    char ** table = (char **) calloc(n_in + n_gates + 1, sizeof(char *));
    if (!table) {
        return;
    }

    size_t n    = 0;
    size_t drop = 0;
    // inherited STRIXLLAMA_* vars are carried over verbatim (the dev/test namespace is not owned).
    // flag the ones that are actually default-ON shipping gates so a leaked control value in a
    // production launch is visible rather than silently reconfiguring the server.
    const char * leaks[8];
    size_t       n_leaks = 0;
    for (size_t i = 0; i < n_in; i++) {
        if (strixllama_env_is_gate(_environ[i])) {
            drop++;
            continue;
        }
        for (size_t w = 0; w < n_leak_watch; w++) {
            if (strixllama_env_name_eq(_environ[i], g_strixllama_leak_watch[w]) && n_leaks < sizeof(leaks) / sizeof(leaks[0])) {
                leaks[n_leaks++] = _environ[i];
                break;
            }
        }
        if (char * dup = _strdup(_environ[i])) {
            table[n++] = dup;
        }
    }
    for (size_t i = 0; i < n_gates; i++) {
        if (char * dup = _strdup(g_strixllama_env_gates[i])) {
            table[n++] = dup;
        }
    }
    table[n] = nullptr;

    // the UCRT binary-searches its environment block, so it has to be sorted the way the CRT sorts
    qsort(table, n, sizeof(char *), strixllama_env_cmp);

    _environ  = table;
    _wenviron = nullptr;   // rebuild the wide view from the new block on first use

    if (drop > 0 || (mode && atoi(mode) >= 2)) {
        fprintf(stderr, "strixllama: environment baked, %zu gates set, %zu inherited gate vars dropped\n",
                n_gates, drop);
        fflush(stderr);
    }
    // Printed regardless of drop: a leaked shipping gate is exactly the case that needs to be loud,
    // and in a clean production launch drop is usually 0 so the banner above never fires.
    if (n_leaks > 0) {
        fprintf(stderr, "strixllama: WARNING %zu shipping gate(s) set in the STRIXLLAMA_ pass-through "
                        "namespace (NOT owned by the bake, left in effect):\n", n_leaks);
        for (size_t i = 0; i < n_leaks; i++) {
            fprintf(stderr, "strixllama:   %s\n", leaks[i]);
        }
        fprintf(stderr, "strixllama: if this is a production launch, unset them -- the launcher does "
                        "not set these, so they came from the parent environment.\n");
        fflush(stderr);
    }
}
#endif // _WIN32

int llama_server(int argc, char ** argv);

int main(int argc, char ** argv) {
#ifdef _WIN32
    strixllama_bake_env();
#endif
    return llama_server(argc, argv);
}
