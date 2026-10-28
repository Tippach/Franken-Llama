// strix_gui.cpp -- native Win32 launcher for the gfx1151 llama.cpp server.
//
// Peer of strix.bat / strix_gui.ps1, not a replacement: the .bat stays the scripted path.
// This is a plain Win32 window (no frameworks): pick a checkpoint, tick the quick-access knobs,
// launch llama-server.exe with NO console (CreateProcess + CREATE_NO_WINDOW, stderr -> a log
// file), and poll the live HTTP endpoints (/health /props /metrics /slots) on a worker thread
// so the UI never blocks. A server already on the port (started by strix.bat or a previous run)
// is adopted on open.
//
// Build: build_gui.cmd (calls vcvars64 then cl). Pure ASCII source.

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <commctrl.h>
#include <winhttp.h>
#include <shellapi.h>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <cwctype>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "ws2_32.lib")
// themed common controls without a separate .manifest / rc.exe
#pragma comment(linker, "\"/manifestdependency:type='win32' " \
    "name='Microsoft.Windows.Common-Controls' version='6.0.0.0' " \
    "processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

// ------------------------------------------------------------------ model list
// embeddedTpl: when true the launcher does NOT force the packaged chat_template.jinja on this
// checkpoint; it launches with an empty CTPL so the server uses the template embedded in the
// model's own GGUF (the same result as clearing CTPL in strix.bat). --jinja is already the server
// default, so no template flag is emitted on this path.
// noLatin: the checkpoint's MTP draft carries its own full-vocab LM head with no d2t mapping, so
// --spec-draft-vocab-latin must NOT be requested: common_speculative turns that into a hard startup
// error ("draft model has no d2t draft-vocab mapping") instead of silently running untrimmed, and
// the load-time trim is impossible here anyway - an EXL3 head is trellis-coded, its rows cannot be
// gathered as contiguous quant blocks. The Latin checkbox is disabled for such a checkpoint.
struct Model { const wchar_t* label; const wchar_t* repo; const wchar_t* draft; const wchar_t* draftFile;
               bool embeddedTpl = false; bool noLatin = false; };
static const Model MODELS[] = {
    { L"UD-IQ4_XS  (unsloth quant)", L"unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS",
      L"unsloth/Qwen3.8-Flash-Next-GGUF", L"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" },
    { L"ISTA IQ3_S  (GSQ-RCO)", L"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:IQ3_S",
      L"unsloth/Qwen3.8-Flash-Next-GGUF", L"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" },
    // Agent/reasoning fine-tune of the same base, from a different quantizer. Verified from the GGUF
    // headers of all three checkpoints (no model load): identical tensor-name set and dims (1224 each),
    // identical vocab (248320 tokens, bos 248044 / eos 248046, tokenizer pre "qwen35") and identical
    // PLE n-gram geometry -- which is what makes the shared MTP head below loadable. Its dtype set is a
    // subset of what UD and ISTA already ship, and the one header key it adds
    // (qwen4exp.attention.recurrent_layers) equals the pattern the loader derives when that key is
    // absent, so no hparam changes and no kernel path runs that is not already exercised.
    // Its embedded chat template is kept (embeddedTpl=true): the packaged template matches none of
    // the three models byte-for-byte, and this fine-tune ships its own, so it runs unmodified.
    { L"Swift 1.5 IQ4_XS  (agent FT)", L"ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF:IQ4_XS",
      L"unsloth/Qwen3.8-Flash-Next-GGUF", L"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf", true },
    // Abliterated (refusal-disabled) re-quant of the same UD-IQ4_XS base already shipped above, by
    // huihui-ai. Same arch, same quant, so it uses the packaged chat template and the shared MTP draft
    // exactly like the UD entry. The repo nests the quant under an "unsloth-UD-IQ4_XS" subdir, but the
    // :UD-IQ4_XS tag still resolves (the matcher regexes "UD-IQ4_XS[.-]" against the shard paths) and
    // the root base mmproj is picked over the repo's Swift mmproj (find_best_sibling, verified).
    { L"Huihui IQ4_XS  (abliterated)", L"huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF:UD-IQ4_XS",
      L"unsloth/Qwen3.8-Flash-Next-GGUF", L"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" },
    // EXL3 (turboderp trellis quants, repacked to GGUF by rulith for this engine; GGUF type ids 256+).
    // 4.05 bpw: KL to BF16 0.022 vs UD-IQ4_XS 0.048 - the upstream repo's own measurement (six domains,
    // ctx 1024), not ours; on this box it decoded at least as fast as UD (42.7 vs 40.9 t/s, MTP on).
    // The quality option, 104.3 GB across 6 shards (first shard is metadata only). The n-gram table
    // streams from disk (lazy on-direct), so resident weight pressure is what it is for any quant here.
    // Its MTP draft MUST be the lr512 own-head file from the same repo: an EXL3 target cannot lend
    // LM-head rows to a shared draft, which is why the other checkpoints' shared Q4_K_M head does not
    // apply - and with a full-vocab own head there is no d2t, so noLatin (see struct note).
    // The repo's root mmproj-F16.gguf is what --mmproj-auto resolves for vision (verified at promote).
    { L"EXL3 4.05bpw  (shipping default)", L"Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF:EXL3-4.05bpw",
      L"Rulith/Qwen3.8-Flash-Next-Rulith-EXL3-GGUF", L"mtp-Qwen3.8-Flash-Next-lr512-ownhead.gguf",
      false, true },
};
static const int N_MODELS = (int)(sizeof(MODELS) / sizeof(MODELS[0]));
// The checkpoint a fresh launch selects, so the default is a named constant instead of the accident of
// array order. --selftest asserts the index and that the default needs no Latin trim, which is what
// catches a MODELS[] reorder silently moving the default somewhere unintended.
static const int DEFAULT_MODEL = 4;   // EXL3 4.05bpw
static const int CTX_VALUES[] = { 131072, 262144, 524288 };

// ------------------------------------------------------------------ control ids
enum {
    IDC_MODEL = 100, IDC_RT, IDC_VISION, IDC_KVQ8, IDC_MTP, IDC_LATIN, IDC_YARN, IDC_SSL,
    IDC_CTX, IDC_PAR, IDC_SPIN, IDC_BIND, IDC_KEY, IDC_KEYHINT, IDC_CACHE, IDC_CACHEDIR,
    IDC_LAUNCH, IDC_STOP, IDC_OPEN, IDC_LIGHT, IDC_STATE, IDC_STAT, IDC_LOG,
};
#define WM_STATUS (WM_APP + 1)
#define WM_EXITED (WM_APP + 2)

// light colors: 0 gray(stopped) 1 goldenrod(starting) 2 limegreen(ready) 3 darkorange(sleeping)
static COLORREF LIGHT[4] = { RGB(128,128,128), RGB(218,165,32), RGB(50,205,50), RGB(255,140,0) };

// ------------------------------------------------------------------ globals
static HINSTANCE g_hInst = nullptr;
static HWND g_hwnd = nullptr, g_hLight = nullptr, g_hState = nullptr, g_hStat = nullptr, g_hLog = nullptr;
static HWND g_hModel = nullptr, g_hCtx = nullptr, g_hPar = nullptr, g_hSpin = nullptr, g_hBind = nullptr,
            g_hKey = nullptr, g_hKeyHint = nullptr, g_hCache = nullptr;
static HWND g_hKeyLabel = nullptr;
static HWND g_ckVision = nullptr, g_ckKvQ8 = nullptr, g_ckMtp = nullptr, g_ckLatin = nullptr,
            g_ckYarn = nullptr, g_ckSsl = nullptr;
static HWND g_btnLaunch = nullptr, g_btnStop = nullptr, g_btnOpen = nullptr, g_hRuntime = nullptr;
static HFONT g_fontUi = nullptr, g_fontMono = nullptr;
static COLORREF g_light = LIGHT[0];

// ------------------------------------------------------------------ DPI scaling
// The process is Per-Monitor-V2 aware, so on a 125%/150% panel the fonts (created with MulDiv over
// LOGPIXELSY) grow while hand-written pixel coordinates do not -- text then overflows its box and
// neighbouring controls appear to overwrite each other. Every coordinate is authored at 96 DPI and
// run through SX(). g_dpi is set from GetDpiForSystem() in wWinMain before any window is created.
static int g_dpi = 96;
static int SX(int v) { return MulDiv(v, g_dpi, 96); }

static std::wstring g_here, g_exe, g_logPath, g_base;
static int g_port = 9931;
static bool g_ssl = false;

static HANDLE g_hProc = nullptr;
static DWORD g_pid = 0;
static bool g_adopted = false;
static ULONGLONG g_launchTick = 0;

static std::thread g_worker;
static std::atomic<bool> g_poll{ false };
static CRITICAL_SECTION g_cs;
static bool g_uiAlive = true;

// result posted from the worker to the UI thread
struct Status {
    bool ok = false, exited = false;
    int exitCode = 0, light = 0;
    std::wstring state, stat, log;
};

// ------------------------------------------------------------------ small helpers
static std::wstring u8tow(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}
static std::string wtou8(const std::wstring& s) {
    if (s.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0, nullptr, nullptr);
    std::string a(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, s.data(), (int)s.size(), &a[0], n, nullptr, nullptr);
    return a;
}
static bool fileExists(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}
static std::wstring dirJoin(const std::wstring& a, const std::wstring& b) {
    if (!a.empty() && a.back() != L'\\') return a + L"\\" + b;
    return a + b;
}
static bool isChecked(HWND h) { return SendMessage(h, BM_GETCHECK, 0, 0) == BST_CHECKED; }
static int comboSel(HWND h) { return (int)SendMessage(h, CB_GETCURSEL, 0, 0); }
static std::wstring editText(HWND h) {
    int n = GetWindowTextLength(h);
    std::wstring s(n, 0);
    GetWindowTextW(h, &s[0], n + 1);
    return s;
}
static void addLine(std::wstring& a, const std::wstring& b) { a += L" "; a += b; }

// ------------------------------------------------------------------ JSON scrape (flat, known fields)
static bool jStr(const std::string& j, const std::string& key, std::string& out, size_t from = 0) {
    std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat, from); if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size()); if (p == std::string::npos) return false; p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t')) p++;
    if (p >= j.size() || j[p] != '"') return false;
    p++; std::string r;
    while (p < j.size() && j[p] != '"') { if (j[p] == '\\' && p + 1 < j.size()) p++; r += j[p++]; }
    out = r; return true;
}
static bool jNum(const std::string& j, const std::string& key, double& out, size_t from = 0) {
    std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat, from); if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size()); if (p == std::string::npos) return false; p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t')) p++;
    size_t s = p;
    while (p < j.size() && (iswdigit(j[p]) || j[p]=='.'||j[p]=='-'||j[p]=='+'||j[p]=='e'||j[p]=='E')) p++;
    if (p == s) return false;
    out = atof(j.substr(s, p - s).c_str()); return true;
}
static bool jBool(const std::string& j, const std::string& key, bool& out, size_t from = 0) {
    std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat, from); if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size()); if (p == std::string::npos) return false; p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t')) p++;
    if (j.compare(p, 4, "true") == 0) { out = true; return true; }
    if (j.compare(p, 5, "false") == 0) { out = false; return true; }
    return false;
}
// split a JSON array of objects into element substrings (nesting + strings aware)
static std::vector<std::string> jsonObjects(const std::string& s) {
    std::vector<std::string> out; size_t i = 0, n = s.size();
    while (i < n && s[i] != '[') i++; if (i >= n) return out; i++;
    while (i < n) {
        while (i < n && (s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t'||s[i]==','||s[i]=='[')) i++;
        if (i >= n || s[i] == ']') break;
        if (s[i] != '{') break;
        int depth = 0; size_t start = i; bool instr = false, esc = false;
        for (; i < n; i++) { char c = s[i];
            if (instr) { if (esc) esc = false; else if (c=='\\') esc = true; else if (c=='"') instr = false; continue; }
            if (c=='"') { instr = true; continue; }
            if (c=='{' || c=='[') depth++;
            else if (c=='}' || c==']') { depth--; if (depth == 0) { i++; break; } }
        }
        out.push_back(s.substr(start, i - start));
    }
    return out;
}
// parse a Prometheus line set: "llamacpp:name value"
struct Metrics {
    bool has_pp=false, has_tg=false, has_proc=false, has_def=false, has_max=false;
    double pp=0, tg=0, proc=0, def=0, max=0, draft=0, accepted=0;
    // cumulative since server start; NOT reset per scrape, unlike the *_seconds gauges
    bool has_cum=false;
    double prompt_total=0, prompt_secs=0, predict_total=0, predict_secs=0;
};
static Metrics parseMetrics(const std::string& m) {
    Metrics x; size_t pos = 0;
    while (pos < m.size()) {
        size_t e = m.find('\n', pos); if (e == std::string::npos) e = m.size();
        std::string line = m.substr(pos, e - pos); pos = e + 1;
        if (line.compare(0, 9, "llamacpp:") != 0) continue;
        size_t sp = line.find_first_of(" \t"); if (sp == std::string::npos) continue;
        std::string name = line.substr(9, sp - 9);
        double v = atof(line.c_str() + sp + 1);
        if      (name == "prompt_tokens_seconds")   { x.pp = v; x.has_pp = true; }
        else if (name == "predicted_tokens_seconds"){ x.tg = v; x.has_tg = true; }
        else if (name == "requests_processing")    { x.proc = v; x.has_proc = true; }
        else if (name == "requests_deferred")      { x.def = v; x.has_def = true; }
        else if (name == "n_tokens_max")           { x.max = v; x.has_max = true; }
        else if (name == "spec_decode_num_draft_tokens_total") x.draft = v;
        else if (name == "spec_decode_num_accepted_tokens_total") x.accepted = v;
        else if (name == "prompt_tokens_total")                { x.prompt_total = v; x.has_cum = true; }
        else if (name == "prompt_seconds_total")               x.prompt_secs = v;
        else if (name == "tokens_predicted_total")             x.predict_total = v;
        else if (name == "tokens_predicted_seconds_total")     x.predict_secs = v;
    }
    return x;
}

// ------------------------------------------------------------------ WinHTTP GET (loopback only)
static bool httpGet(HINTERNET s, bool https, INTERNET_PORT port, const wchar_t* path, std::string& out) {
    out.clear();
    HINTERNET c = WinHttpConnect(s, L"127.0.0.1", port, 0); if (!c) return false;
    HINTERNET r = WinHttpOpenRequest(c, L"GET", path, nullptr, WINHTTP_NO_REFERER,
                                     WINHTTP_DEFAULT_ACCEPT_TYPES, https ? WINHTTP_FLAG_SECURE : 0);
    if (!r) { WinHttpCloseHandle(c); return false; }
    if (https) {
        DWORD fl = SECURITY_FLAG_IGNORE_UNKNOWN_CA | SECURITY_FLAG_IGNORE_CERT_CN_INVALID
                 | SECURITY_FLAG_IGNORE_CERT_DATE_INVALID | SECURITY_FLAG_IGNORE_CERT_WRONG_USAGE;
        WinHttpSetOption(r, WINHTTP_OPTION_SECURITY_FLAGS, &fl, sizeof fl);
    }
    bool ok = false;
    if (WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
        && WinHttpReceiveResponse(r, nullptr)) {
        ok = true;
        for (;;) {
            DWORD avail = 0;
            if (!WinHttpQueryDataAvailable(r, &avail) || avail == 0) break;
            char buf[8192]; DWORD rd = 0;
            while (WinHttpReadData(r, buf, sizeof buf, &rd) && rd) out.append(buf, rd);
        }
    }
    WinHttpCloseHandle(r); WinHttpCloseHandle(c);
    return ok;
}

// ------------------------------------------------------------------ log tail
static std::wstring readLogTail() {
    HANDLE f = CreateFileW(g_logPath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return L"";
    LARGE_INTEGER li; GetFileSizeEx(f, &li);
    LARGE_INTEGER sp; sp.QuadPart = li.QuadPart > 8192 ? li.QuadPart - 8192 : 0;
    long long skip = (long long)sp.QuadPart;
    SetFilePointerEx(f, sp, nullptr, FILE_BEGIN);
    std::string raw; char buf[8192]; DWORD rd = 0;
    while (ReadFile(f, buf, sizeof buf, &rd, nullptr) && rd) raw.append(buf, rd);
    CloseHandle(f);
    if (skip > 0) { size_t nl = raw.find('\n'); if (nl != std::string::npos) raw = raw.substr(nl + 1); }
    return u8tow(raw);
}

// ------------------------------------------------------------------ command-line builder (mirrors strix.bat)
struct Opts {
    int model = DEFAULT_MODEL, ctx = 524288, port = 9931, parallel = 2, yarnScale = 2;
    int cache = 0;                 // 0 kvchain, 1 disk, 2 off
    bool loopback = true;
    // kvq8 defaults OFF: the shipping checkpoint fits the full 512k / 2-slot shape with an
    // UNQUANTIZED (f16) KV cache, so quality is not traded for headroom that is not needed.
    bool yarn = true, kvq8 = false, vision = true, mtp = true, latin = true, ssl = false;
    bool withEndpoints = true;
    std::wstring apiKey, cacheDir, tmpl, sslKey, sslCert;
    int kvchainGb = 100, pccacheGb = 64;
};
static std::wstring fmt(const wchar_t* f, ...);
static bool buildArgs(const Opts& o, std::wstring& args, std::wstring& err) {
    bool loopback = o.loopback;
    // The API key is OPTIONAL even on a non-loopback bind: the server runs fine without one, and
    // refusing to start was too heavy-handed. The GUI shows a warning instead. If a key is given it
    // is passed through below.
    const Model& m = MODELS[o.model];
    bool local = wcsstr(m.repo, L".gguf") != nullptr;
    args = local ? L"-m \"" : L"-hf \"";
    args += m.repo; args += L"\"";
    addLine(args, L"-ngl 99");
    addLine(args, fmt(L"--ctx-size %d", o.ctx));
    addLine(args, L"-b 8192 -ub 8192 --flash-attn on -t 4");
    addLine(args, fmt(L"--parallel %d", o.parallel));
    addLine(args, fmt(L"--port %d --host %s", o.port, loopback ? L"127.0.0.1" : L"0.0.0.0"));
    addLine(args, L"--temp 1.0 --presence-penalty 0.0 --repeat-penalty 1.0");

    if (o.cache == 0) {
        addLine(args, fmt(L"--kv-chain-dir \"%s\" --kv-chain-limit-gb %d", o.cacheDir.c_str(), o.kvchainGb));
    } else if (o.cache == 1) {
        addLine(args, fmt(L"--cache-ram 1024 --no-cache-idle-slots --cache-disk-dir \"%s\" --cache-disk-limit-mib %d",
                          o.cacheDir.c_str(), o.pccacheGb * 1024));
    } else {
        addLine(args, L"--cache-ram 0");
    }
    addLine(args, o.parallel <= 1 ? L"--no-kv-unified" : L"--kv-unified");
    addLine(args, o.vision ? L"--mmproj-auto" : L"--no-mmproj");
    addLine(args, L"--fit off --load-mode none --lazy-mode on-direct");
    if (o.kvq8) addLine(args, L"-ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0");
    if (o.yarn) {
        if (o.ctx <= 262144) {
            err = fmt(L"YaRN is on but CTX=%d is within the trained 262144 - turn YaRN off or raise CTX.", o.ctx);
            return false;
        }
        addLine(args, fmt(L"--rope-scaling yarn --rope-scale %d --yarn-orig-ctx 262144", o.yarnScale));
    }
    if (!o.tmpl.empty()) {
        addLine(args, fmt(L"--jinja --chat-template-file \"%s\"", o.tmpl.c_str()));
        addLine(args, L"--reasoning on --reasoning-format deepseek --reasoning-effort low --reasoning-preserve");
    }
    if (o.mtp && m.draft && m.draft[0]) {
        addLine(args, fmt(L"-hfd \"%s\" -md \"%s\"", m.draft, m.draftFile));
        addLine(args, L"--spec-type draft-mtp --spec-draft-ngl 99 --spec-draft-n-max 3 --spec-draft-p-min 0.30");
        // noLatin checkpoints throw at startup if the trim is required - never emit the flag for them
        if (o.latin && !m.noLatin) addLine(args, L"--spec-draft-vocab-latin");
    }
    if (o.ssl) {
        if (!fileExists(o.sslKey) || !fileExists(o.sslCert)) { err = L"HTTPS is on but key/cert files are missing."; return false; }
        addLine(args, fmt(L"--ssl-key-file \"%s\" --ssl-cert-file \"%s\"", o.sslKey.c_str(), o.sslCert.c_str()));
    }
    if (o.withEndpoints) addLine(args, L"--metrics --slots");
    if (!o.apiKey.empty()) addLine(args, fmt(L"--api-key \"%s\"", o.apiKey.c_str()));
    return true;
}
static std::wstring fmt(const wchar_t* f, ...) {
    wchar_t buf[2048]; va_list ap; va_start(ap, f);
    _vsnwprintf(buf, 2048, f, ap); va_end(ap);
    return std::wstring(buf);
}

// ------------------------------------------------------------------ CreateProcess, no console
static bool startServer(const std::wstring& args, std::wstring& err) {
    SECURITY_ATTRIBUTES sa = {}; sa.nLength = sizeof sa; sa.bInheritHandle = TRUE; sa.lpSecurityDescriptor = nullptr;
    HANDLE lf = CreateFileW(g_logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (lf == INVALID_HANDLE_VALUE) { err = L"cannot open log file"; return false; }
    STARTUPINFOW si = {}; si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES; si.hStdError = lf; si.hStdOutput = lf; si.hStdInput = nullptr;
    std::wstring cmd = L"\"" + g_exe + L"\" " + args;
    std::vector<wchar_t> cmdv(cmd.begin(), cmd.end()); cmdv.push_back(0);
    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessW(nullptr, cmdv.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW, nullptr, g_here.c_str(), &si, &pi);
    CloseHandle(lf);
    if (!ok) { err = fmt(L"CreateProcess failed, Win32 error %lu", GetLastError()); return false; }
    CloseHandle(pi.hThread);
    g_hProc = pi.hProcess; g_pid = pi.dwProcessId;
    return true;
}

// ------------------------------------------------------------------ find the PID owning the port
static DWORD findServerPid(int port) {
    // iphelperapi via dynamic load to avoid an extra import lib link
    HMODULE iph = LoadLibraryW(L"iphlpapi.dll"); if (!iph) return 0;
    typedef DWORD(WINAPI* GetTcpFn)(PVOID, PULONG, BOOL, ULONG, TCP_TABLE_CLASS, ULONG);
    GetTcpFn fn = (GetTcpFn)GetProcAddress(iph, "GetExtendedTcpTable");
    DWORD pid = 0;
    if (fn) {
        ULONG sz = 0;
        fn(nullptr, &sz, TRUE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0);
        std::vector<char> buf(sz);
        if (fn(buf.data(), &sz, TRUE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0) == NO_ERROR) {
            auto* t = (PMIB_TCPTABLE_OWNER_PID)buf.data();
            for (DWORD i = 0; i < t->dwNumEntries; i++) {
                // dwLocalPort is network-byte-order in the LOW 16 bits (verified against the table)
                DWORD lp = ntohs((u_short)(t->table[i].dwLocalPort & 0xFFFF));
                if (t->table[i].dwState == MIB_TCP_STATE_LISTEN && lp == (DWORD)port) { pid = t->table[i].dwOwningPid; break; }
            }
        }
    }
    FreeLibrary(iph);
    return pid;
}

// ------------------------------------------------------------------ worker: poll + build status
static void postStatus(Status* s) {
    EnterCriticalSection(&g_cs);
    if (g_uiAlive) PostMessageW(g_hwnd, WM_STATUS, 0, (LPARAM)s); else delete s;
    LeaveCriticalSection(&g_cs);
}
static std::wstring fmtNum(double v, bool has) {
    if (!has) return L"-";
    wchar_t b[64]; _snwprintf(b, 64, L"%.1f", v); return b;
}
// counts (slots in flight, deferred, sequence length) are integers -- printing them with %.1f gave
// nonsense like "processing 0.0" and "max seq 89816.0".
static std::wstring fmtInt(double v, bool has) {
    if (!has) return L"-";
    wchar_t b[64]; _snwprintf(b, 64, L"%.0f", v); return b;
}
// Live per-request throughput, derived the same way the WebUI gets its smooth numbers -- from the
// counters of the request in flight rather than from a server-wide gauge. /slots exposes, per slot,
// n_prompt_tokens_processed and next_token[].n_decoded, both of which climb for the duration of the
// current task, so the difference between two polls divided by the real elapsed time is a genuine
// instantaneous rate. The task id (id_task) is tracked so a new request re-baselines instead of
// producing a huge negative delta, and only slots that are actually processing contribute.
struct SlotBase { double idTask = -1; double promptProc = 0; double decoded = 0; };
static std::map<int, SlotBase> g_slotBase;
static ULONGLONG g_slotsTick = 0;
static void resetSlotRates() { g_slotBase.clear(); g_slotsTick = 0; }

// One poll cycle: hit /health, and if up, /props + /metrics + /slots, and build the status text.
// Shared by the GUI worker thread and the headless --smoke test so both exercise the same parse.
static Status pollOnce(HINTERNET s) {
    std::string health;
    Status st; st.log = readLogTail();
    if (!httpGet(s, g_ssl, (INTERNET_PORT)g_port, L"/health", health)) {
        st.ok = false; st.light = 1; st.state = L"Starting... (loading model)";
        st.stat = L"Waiting for the server to answer /health.\r\n"
                  L"Large checkpoints and the first HF download can take a while.";
        return st;
    }
    st.ok = true;
    std::string props, metrics, slots;
    bool sleeping = false, vision = false, metricsOn = false, slotsOn = false;
    std::wstring model, ftype, nctx, nslots, build;
    if (httpGet(s, g_ssl, (INTERNET_PORT)g_port, L"/props", props)) {
        std::string v; double d;
        if (jStr(props, "model_alias", v)) model = u8tow(v);
        if (jStr(props, "model_ftype", v)) ftype = u8tow(v);
        size_t dg = props.find("\"default_generation_settings\"");
        if (jNum(props, "n_ctx", d, dg)) { wchar_t b[32]; _snwprintf(b, 32, L"%.0f", d); nctx = b; }
        if (jNum(props, "total_slots", d)) { wchar_t b[32]; _snwprintf(b, 32, L"%.0f", d); nslots = b; }
        if (jStr(props, "build_info", v)) build = u8tow(v);
        size_t mm = props.find("\"modalities\"");
        jBool(props, "vision", vision, mm);
        jBool(props, "endpoint_metrics", metricsOn);
        jBool(props, "endpoint_slots", slotsOn);
        jBool(props, "is_sleeping", sleeping);
    }
    std::wstring up = L"-";
    if (!g_adopted && g_launchTick) {
        long long sec = (long long)((GetTickCount64() - g_launchTick) / 1000);
        up = fmt(L"%lldm%02llds", sec / 60, sec % 60);
    }
    // /slots is fetched before the perf line because the live rates below are derived from it.
    // ratePp/rateTg are this slot's own instantaneous t/s over the last poll interval (-1 = no
    // fresh same-task delta); pt/dec stay the cumulative per-task counters.
    struct SlotRow { double id, pt, dec, ratePp, rateTg; bool proc, spec; };
    std::vector<SlotRow> rows;
    double livePp = -1, liveTg = -1;
    if (slotsOn && httpGet(s, g_ssl, (INTERNET_PORT)g_port, L"/slots", slots)) {
        auto objs = jsonObjects(slots);
        double sumPrompt = 0, sumDecoded = 0;
        bool anyProcessing = false;
        std::map<int, SlotBase> next;
        // elapsed since the previous poll; computed once so every slot's rate uses the same window.
        double dt = g_slotsTick ? (GetTickCount64() - g_slotsTick) / 1000.0 : 0.0;
        for (size_t i = 0; i < objs.size(); i++) {
            const std::string& e = objs[i]; double id = 0, pt = 0, dec = 0, idTask = -1;
            bool proc = false, spec = false;
            jNum(e, "id", id); jBool(e, "is_processing", proc); jBool(e, "speculative", spec);
            jNum(e, "n_prompt_tokens", pt);
            jNum(e, "id_task", idTask);
            double pp = 0; jNum(e, "n_prompt_tokens_processed", pp);
            size_t nt = e.find("\"next_token\"");
            if (nt != std::string::npos) jNum(e, "n_decoded", dec, nt);
            double rPp = -1, rTg = -1;
            int key = (int)id;
            next[key] = SlotBase{ idTask, pp, dec };
            if (proc) {
                anyProcessing = true;
                auto it = g_slotBase.find(key);
                // only compare against a baseline from the SAME task; a finished or new task restarts
                if (it != g_slotBase.end() && it->second.idTask == idTask && g_slotsTick && dt > 0.05) {
                    double dPrompt = pp - it->second.promptProc;
                    double dDec    = dec - it->second.decoded;
                    // per-slot rate: this slot's own tokens over the interval.
                    if (dPrompt > 0) { sumPrompt += dPrompt; rPp = dPrompt / dt; }
                    if (dDec    > 0) { sumDecoded += dDec;   rTg = dDec / dt; }
                }
            }
            rows.push_back({ id, pt, dec, rPp, rTg, proc, spec });
        }
        if (anyProcessing && dt > 0.05) {
            if (sumPrompt  > 0) livePp = sumPrompt  / dt;
            if (sumDecoded > 0) liveTg = sumDecoded / dt;
        }
        g_slotBase.swap(next);
        g_slotsTick = GetTickCount64();
    } else {
        resetSlotRates();
    }
    std::wstring text;
    text += fmt(L"model : %s  [%s]   ctx=%s  slots=%s\r\n", model.c_str(), ftype.c_str(), nctx.c_str(), nslots.c_str());
    text += fmt(L"state : %s   vision=%s   uptime=%s   build=%s\r\n",
                sleeping ? L"idle/sleeping" : L"ready", vision ? L"on" : L"off", up.c_str(), build.c_str());
    if (metricsOn && httpGet(s, g_ssl, (INTERNET_PORT)g_port, L"/metrics", metrics)) {
        Metrics x = parseMetrics(metrics);
        std::wstring acc = L"-";
        if (x.draft > 0) { wchar_t b[32]; _snwprintf(b, 32, L"%.1f%%", 100.0 * x.accepted / x.draft); acc = b; }
        // The *_seconds gauges are averaged over the window BETWEEN two scrapes and the server resets
        // the bucket on every scrape (server-context.cpp use_cached_metrics), so a burst that misses
        // the window reads 0. The cumulative counters are never reset, so tokens_total/seconds_total
        // is a true busy-time average -- but it is a LIFETIME average, not a live rate.
        // Priority: live rate measured from this request's own slot counters, else the per-scrape
        // gauge if it happened to be non-zero, else the lifetime average tagged " (avg)".
        double cumPp = (x.prompt_secs > 0) ? x.prompt_total / x.prompt_secs : -1;
        double cumTg = (x.predict_secs > 0) ? x.predict_total / x.predict_secs : -1;
        bool ppLive = livePp >= 0 || (x.has_pp && x.pp > 0);
        bool tgLive = liveTg >= 0 || (x.has_tg && x.tg > 0);
        double pp = livePp >= 0 ? livePp : ((x.has_pp && x.pp > 0) ? x.pp : cumPp);
        double tg = liveTg >= 0 ? liveTg : ((x.has_tg && x.tg > 0) ? x.tg : cumTg);
        // NOTE the L"" on both arms: a narrow " (avg)" here would be passed as char* to a wide %s,
        // which reinterprets the bytes as UTF-16 and prints mojibake in place of the marker.
        text += fmt(L"perf  : prefill %s%s t/s   decode %s%s t/s   spec accept %s\r\n",
                    fmtNum(pp, pp >= 0).c_str(), (pp >= 0 && !ppLive) ? L" (avg)" : L"",
                    fmtNum(tg, tg >= 0).c_str(), (tg >= 0 && !tgLive) ? L" (avg)" : L"", acc.c_str());
        text += fmt(L"load  : processing %s   deferred %s   max seq %s\r\n",
                    fmtInt(x.proc, x.has_proc).c_str(), fmtInt(x.def, x.has_def).c_str(),
                    x.has_max ? fmtInt(x.max, true).c_str() : std::wstring(L"-").c_str());
    } else if (metricsOn) {
        text += L"perf  : metrics endpoint not reachable\r\n";
    } else {
        text += L"perf  : metrics endpoint not enabled\r\n";
    }
    if (!rows.empty()) {
        std::wstring line = L"slots : ";
        for (size_t i = 0; i < rows.size(); i++) {
            const SlotRow& r = rows[i];
            wchar_t b[160];
            _snwprintf(b, 160, L"%sslot %.0f: %s %s  prompt=%.0f gen=%.0f",
                       i ? L" | " : L"", r.id, r.proc ? L"busy" : L"idle", r.spec ? L"spec" : L"no-spec", r.pt, r.dec);
            line += b;
            // per-slot live throughput, only when this slot has a fresh same-task delta.
            if (r.ratePp >= 0 || r.rateTg >= 0) {
                wchar_t pb[24], db[24], rb[64];
                if (r.ratePp >= 0) _snwprintf(pb, 24, L"%.0f", r.ratePp); else wcscpy_s(pb, L"-");
                if (r.rateTg >= 0) _snwprintf(db, 24, L"%.0f", r.rateTg); else wcscpy_s(db, L"-");
                _snwprintf(rb, 64, L" [%ls/%ls t/s]", pb, db);
                line += rb;
            }
        }
        text += line;
    }
    st.stat = text;
    st.light = sleeping ? 3 : 2;
    st.state = g_adopted ? fmt(L"Running (pid %lu, external)", g_pid)
                         : (sleeping ? L"Idle (sleeping)" : L"Running");
    return st;
}
static void worker() {
    HINTERNET s = WinHttpOpen(L"StrixLauncher/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    WinHttpSetTimeouts(s, 1500, 1500, 2000, 2000);
    while (g_poll.load()) {
        DWORD code = STILL_ACTIVE;
        if (g_hProc) GetExitCodeProcess(g_hProc, &code);
        if (g_hProc && code != STILL_ACTIVE) {
            Status* st = new Status; st->exited = true; st->exitCode = (int)code; st->log = readLogTail();
            postStatus(st); break;
        }
        Status* st = new Status(pollOnce(s));
        postStatus(st);
        // 1s, matching --watch (the configuration the live-rate deltas were validated under). A
        // slower poll widens the window a short prefill must fit inside to register, and the GUI
        // feels noticeably lumpier; /slots + /metrics on loopback are cheap enough not to matter.
        for (int i = 0; i < 10 && g_poll.load(); i++) Sleep(100);
    }
    if (s) WinHttpCloseHandle(s);
}
static void startWorker() { if (!g_worker.joinable()) { resetSlotRates(); g_poll = true; g_worker = std::thread(worker); } }
static void stopWorker() { g_poll = false; if (g_worker.joinable()) g_worker.join(); }

// ------------------------------------------------------------------ UI state helpers
static COLORREF g_keyHintColor = RGB(90, 90, 90);
static void syncKeyHint();
static void syncLatinEnabled(int modelIdx);   // defined with the window code; setUi re-asserts it
static void setUi(bool running) {
    EnableWindow(g_btnLaunch, !running);
    EnableWindow(g_btnStop, running);
    EnableWindow(g_btnOpen, running);
    EnableWindow(g_hModel, !running);
    EnableWindow(g_ckVision, !running); EnableWindow(g_ckKvQ8, !running);
    EnableWindow(g_ckMtp, !running);    EnableWindow(g_ckLatin, !running);
    EnableWindow(g_ckYarn, !running);   EnableWindow(g_ckSsl, !running);
    EnableWindow(g_hCtx, !running);     EnableWindow(g_hPar, !running);
    EnableWindow(g_hSpin, !running);    EnableWindow(g_hCache, !running);
    EnableWindow(g_hBind, !running);
    if (!running) syncKeyHint(); else EnableWindow(g_hKey, FALSE);
    // EnableWindow(g_ckLatin, TRUE) above blindly re-enables the trim box, which would UNDO the
    // own-head veto made in WM_CREATE / on model change. Re-assert the model policy after it, so a
    // checkpoint whose draft cannot be trimmed stays disabled after every start/stop transition.
    // Invisible while the default checkpoint allowed the trim; the EXL3 default exposed it.
    if (!running) syncLatinEnabled((int)SendMessage(g_hModel, CB_GETCURSEL, 0, 0));
    if (!running) { g_light = LIGHT[0]; InvalidateRect(g_hLight, nullptr, TRUE); SetWindowTextW(g_hState, L"Stopped"); }
}
static void syncKeyHint() {
    // SetWindowText on the key box fires EN_CHANGE, which calls right back in here. Without this
    // guard the loop never terminates and the process dies with a stack overflow (0xC00000FD).
    static bool inGuard = false;
    if (inGuard) return;
    inGuard = true;
    bool net = comboSel(g_hBind) == 1;
    // In loopback mode there is no key to type: hide the box and its label entirely rather than
    // leave a greyed empty edit sitting next to the bind combo (that was the "stale rectangle").
    ShowWindow(g_hKey, net ? SW_SHOW : SW_HIDE);
    ShowWindow(g_hKeyLabel, net ? SW_SHOW : SW_HIDE);
    EnableWindow(g_hKey, net);
    if (!net && editText(g_hKey).size()) SetWindowTextW(g_hKey, L"");
    // hint moves left to fill the space the key box vacated when in loopback
    if (net) SetWindowPos(g_hKeyHint, nullptr, SX(516), SX(174), SX(164), SX(20), SWP_NOZORDER);
    else     SetWindowPos(g_hKeyHint, nullptr, SX(330), SX(174), SX(350), SX(20), SWP_NOZORDER);
    std::wstring key = editText(g_hKey);
    COLORREF c; std::wstring hint;
    if (net && key.empty()) { c = RGB(180, 120, 0); hint = L"no key: open to your LAN"; }
    else if (net)           { c = RGB(0, 100, 0);   hint = L"protected by the key"; }
    else                    { c = RGB(90, 90, 90);  hint = L"loopback only - not reachable from the network"; }
    SetWindowTextW(g_hKeyHint, hint.c_str());
    // recolor handled in WM_CTLCOLORSTATIC via the stored brush color
    g_keyHintColor = c; InvalidateRect(g_hKeyHint, nullptr, TRUE);
    inGuard = false;
}

static bool attachExisting() {
    DWORD pid = findServerPid(g_port);
    if (!pid) return false;
    HANDLE h = OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return false;
    g_hProc = h; g_pid = pid; g_adopted = true; g_launchTick = 0;
    g_base = (g_ssl ? L"https" : L"http") + std::wstring(L"://127.0.0.1:") + std::to_wstring(g_port);
    setUi(true);
    g_light = LIGHT[1]; InvalidateRect(g_hLight, nullptr, TRUE);
    SetWindowTextW(g_hState, fmt(L"Re-attached to pid %lu", pid).c_str());
    SetWindowTextW(g_hStat, fmt(L"Found a server already running on port %d (pid %lu), started outside this window.\r\nMonitoring it; Stop will terminate it.", g_port, pid).c_str());
    startWorker();
    return true;
}

static Opts gatherOpts() {
    Opts o;
    o.model = comboSel(g_hModel);
    int ci = comboSel(g_hCtx); o.ctx = CTX_VALUES[ci < 0 ? 2 : ci];
    o.parallel = (int)SendMessage(g_hSpin, UDM_GETPOS, 0, 0);
    o.loopback = comboSel(g_hBind) == 0;
    o.apiKey = editText(g_hKey);
    o.yarn = isChecked(g_ckYarn); o.kvq8 = isChecked(g_ckKvQ8); o.vision = isChecked(g_ckVision);
    o.mtp = isChecked(g_ckMtp);   o.latin = isChecked(g_ckLatin); o.ssl = isChecked(g_ckSsl);
    // Defense-in-depth: a noLatin checkpoint's own-head draft cannot be trimmed and the server turns
    // that combination into a startup error. If the checkbox is somehow still set (enable-gate
    // regression, or a scripted BM_SETCHECK), drop the flag rather than emit a refused command.
    if (MODELS[o.model].noLatin) o.latin = false;
    o.cache = comboSel(g_hCache); if (o.cache < 0) o.cache = 0;
    o.port = g_port;
    o.cacheDir = dirJoin(g_here, L"prompt-cache");
    // A checkpoint marked embeddedTpl runs with its own GGUF template: leave o.tmpl empty so
    // buildArgs skips --chat-template-file entirely (the empty-CTPL path in strix.bat).
    std::wstring tmpl = dirJoin(g_here, L"chat_template.jinja");
    if (fileExists(tmpl) && !MODELS[o.model].embeddedTpl) o.tmpl = tmpl;
    o.sslKey = dirJoin(g_here, L"key.pem");
    o.sslCert = dirJoin(g_here, L"cert.pem");
    return o;
}

// ------------------------------------------------------------------ window
static HWND mk(const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h,
               HFONT f = nullptr) {
    HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, SX(x), SX(y), SX(w), SX(h),
                             g_hwnd, nullptr, g_hInst, nullptr);
    SendMessage(c, WM_SETFONT, (WPARAM)(f ? f : g_fontUi), TRUE);
    return c;
}
// CreateWindowExW variant taking a control id, so the id survives for WM_COMMAND routing.
static HWND mkid(const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h,
                 int id, HFONT f = nullptr) {
    HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, SX(x), SX(y), SX(w), SX(h),
                             g_hwnd, (HMENU)(INT_PTR)id, g_hInst, nullptr);
    SendMessage(c, WM_SETFONT, (WPARAM)(f ? f : g_fontUi), TRUE);
    return c;
}

// A noLatin checkpoint (own-head draft without a d2t mapping) must not offer Latin trim: the flag
// is dropped in gatherOpts() and the checkbox is unchecked + disabled here so the reason is visible.
// Called from WM_CREATE after CB_SETCURSEL and from WM_COMMAND on model change.
static void syncLatinEnabled(int modelIdx) {
    if (!g_ckLatin || modelIdx < 0 || modelIdx >= N_MODELS) return;
    bool allow = !MODELS[modelIdx].noLatin;
    EnableWindow(g_ckLatin, allow);
    if (!allow) SendMessage(g_ckLatin, BM_SETCHECK, BST_UNCHECKED, 0);
}

static LRESULT CALLBACK WndProc(HWND h, UINT msg, WPARAM w, LPARAM l) {
    switch (msg) {
    case WM_CREATE: {
        g_hwnd = h;   // mk/mkid parent off this; g_hwnd is otherwise set only after CreateWindowExW returns
        // ---- Model group
        mk(L"BUTTON", L"Model", BS_GROUPBOX, 12, 8, 680, 74);
        mk(L"STATIC", L"Checkpoint", 0, 26, 34, 74, 20);
        g_hModel = mkid(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, 104, 31, 250, 200, IDC_MODEL);
        for (int i = 0; i < N_MODELS; i++) SendMessageW(g_hModel, CB_ADDSTRING, 0, (LPARAM)MODELS[i].label);
        SendMessage(g_hModel, CB_SETCURSEL, DEFAULT_MODEL, 0);
        g_hRuntime = mk(L"STATIC", L"", 0, 366, 34, 316, 20);
        // ---- Quick access group
        mk(L"BUTTON", L"Quick access", BS_GROUPBOX, 12, 90, 680, 128);
        g_ckVision = mkid(L"BUTTON", L"Vision (image input)", BS_AUTOCHECKBOX, 26, 114, 168, 22, IDC_VISION);
        g_ckKvQ8   = mkid(L"BUTTON", L"KV cache q8_0",        BS_AUTOCHECKBOX, 26, 140, 168, 22, IDC_KVQ8);
        g_ckMtp    = mkid(L"BUTTON", L"MTP spec decode",      BS_AUTOCHECKBOX, 196, 114, 152, 22, IDC_MTP);
        g_ckLatin  = mkid(L"BUTTON", L"Latin draft trim",     BS_AUTOCHECKBOX, 196, 140, 152, 22, IDC_LATIN);
        g_ckYarn   = mkid(L"BUTTON", L"YaRN rope scaling",    BS_AUTOCHECKBOX, 352, 114, 152, 22, IDC_YARN);
        g_ckSsl    = mkid(L"BUTTON", L"HTTPS (needs certs)",  BS_AUTOCHECKBOX, 352, 140, 152, 22, IDC_SSL);
        // KV q8_0 is deliberately NOT pre-checked: the default ships an unquantized KV cache.
        for (HWND c : { g_ckVision, g_ckMtp, g_ckLatin, g_ckYarn }) SendMessage(c, BM_SETCHECK, BST_CHECKED, 0);
        syncLatinEnabled((int)SendMessage(g_hModel, CB_GETCURSEL, 0, 0));
        mk(L"STATIC", L"Context", 0, 512, 116, 50, 20);
        g_hCtx = mkid(L"COMBOBOX", L"", CBS_DROPDOWNLIST, 566, 113, 100, 200, IDC_CTX);
        for (int v : CTX_VALUES) { std::wstring s = std::to_wstring(v); SendMessageW(g_hCtx, CB_ADDSTRING, 0, (LPARAM)s.c_str()); }
        SendMessage(g_hCtx, CB_SETCURSEL, 2, 0);
        mk(L"STATIC", L"Slots", 0, 512, 142, 50, 20);
        g_hPar = mkid(L"EDIT", L"2", WS_BORDER | ES_NUMBER | ES_RIGHT, 566, 139, 60, 22, IDC_PAR);
        g_hSpin = CreateWindowExW(0, UPDOWN_CLASS, L"", WS_CHILD | WS_VISIBLE | UDS_SETBUDDYINT | UDS_ALIGNRIGHT | UDS_AUTOBUDDY | UDS_ARROWKEYS,
                                  0, 0, 0, 0, h, (HMENU)IDC_SPIN, g_hInst, nullptr);
        SendMessage(g_hSpin, UDM_SETBUDDY, (WPARAM)g_hPar, 0);
        SendMessage(g_hSpin, UDM_SETRANGE, 0, MAKELPARAM(8, 1));
        SendMessage(g_hSpin, UDM_SETPOS, 0, MAKELPARAM(2, 0));
        // ---- serving row: bind choice, then (network mode only) the API key and the hint
        mk(L"STATIC", L"Serving", 0, 26, 174, 56, 20);
        g_hBind = mkid(L"COMBOBOX", L"", CBS_DROPDOWNLIST, 86, 171, 232, 200, IDC_BIND);
        SendMessageW(g_hBind, CB_ADDSTRING, 0, (LPARAM)L"This computer only (127.0.0.1)");
        SendMessageW(g_hBind, CB_ADDSTRING, 0, (LPARAM)L"All interfaces (0.0.0.0)");
        SendMessage(g_hBind, CB_SETCURSEL, 0, 0);
        g_hKeyLabel = mk(L"STATIC", L"API key", 0, 330, 174, 50, 20);
        g_hKey = mkid(L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | ES_PASSWORD, 384, 171, 120, 22, IDC_KEY);
        g_hKeyHint = mkid(L"STATIC", L"", SS_LEFTNOWORDWRAP, 516, 174, 164, 20, IDC_KEYHINT);
        // ---- Cache group
        mk(L"BUTTON", L"Prompt cache on disk", BS_GROUPBOX, 12, 226, 680, 56);
        mk(L"STATIC", L"Mode", 0, 26, 250, 40, 20);
        g_hCache = mkid(L"COMBOBOX", L"", CBS_DROPDOWNLIST, 70, 247, 120, 200, IDC_CACHE);
        for (auto* v : { L"kvchain", L"disk", L"off" }) SendMessageW(g_hCache, CB_ADDSTRING, 0, (LPARAM)v);
        SendMessage(g_hCache, CB_SETCURSEL, 0, 0);
        HWND hCacheDir = mk(L"STATIC", L"", 0, 208, 250, 470, 20);
        SetWindowTextW(hCacheDir, fmt(L"(dir: %s)", dirJoin(g_here, L"prompt-cache").c_str()).c_str());
        // ---- buttons + light + state
        g_btnLaunch = mkid(L"BUTTON", L"Launch", BS_PUSHBUTTON, 12, 292, 110, 34, IDC_LAUNCH);
        g_btnStop   = mkid(L"BUTTON", L"Stop",   BS_PUSHBUTTON, 130, 292, 110, 34, IDC_STOP);
        g_btnOpen   = mkid(L"BUTTON", L"Open Web UI", BS_PUSHBUTTON, 248, 292, 110, 34, IDC_OPEN);
        g_hLight = mkid(L"STATIC", L"", SS_OWNERDRAW, 378, 298, 16, 16, IDC_LIGHT);
        g_hState = mkid(L"STATIC", L"Stopped", 0, 400, 298, 290, 20, IDC_STATE, g_fontMono);
        // ---- status + log
        mk(L"BUTTON", L"Live status", BS_GROUPBOX, 12, 336, 680, 150);
        g_hStat = mkid(L"EDIT", L"", WS_BORDER | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, 26, 358, 652, 116, IDC_STAT, g_fontMono);
        mk(L"BUTTON", L"Server log (stderr)", BS_GROUPBOX, 12, 494, 680, 130);
        g_hLog = mkid(L"EDIT", L"", WS_BORDER | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_AUTOHSCROLL, 26, 516, 652, 96, IDC_LOG, g_fontMono);
        return 0;
    }
    case WM_CTLCOLORSTATIC: {
        HDC dc = (HDC)w; HWND c = (HWND)l;
        // A read-only EDIT (the status + log boxes) also sends WM_CTLCOLORSTATIC, not just STATIC.
        // Returning NULL_BRUSH + TRANSPARENT here was the "text overwrites itself" bug: on the next
        // SetWindowText the control never erased the previous glyphs, so new text printed over old.
        // Hand back an OPAQUE brush matching each control's surface so every repaint clears first.
        if (c == g_hStat || c == g_hLog) {
            SetBkColor(dc, RGB(255, 255, 255));
            SetBkMode(dc, OPAQUE);
            return (LRESULT)GetStockObject(WHITE_BRUSH);
        }
        SetBkColor(dc, GetSysColor(COLOR_BTNFACE));
        SetBkMode(dc, OPAQUE);
        if (c == g_hKeyHint) SetTextColor(dc, g_keyHintColor);
        else if (c == g_hRuntime) SetTextColor(dc, RGB(90, 90, 90));
        return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);
    }
    case WM_DRAWITEM: {
        DRAWITEMSTRUCT* di = (DRAWITEMSTRUCT*)l;
        if (di->CtlID == IDC_LIGHT) {
            HBRUSH b = CreateSolidBrush(g_light);
            FillRect(di->hDC, &di->rcItem, b);
            HPEN p = CreatePen(PS_SOLID, 1, RGB(60, 60, 60));
            HGDIOBJ op = SelectObject(di->hDC, p);
            HBRUSH ob = (HBRUSH)SelectObject(di->hDC, b);
            Ellipse(di->hDC, di->rcItem.left, di->rcItem.top, di->rcItem.right, di->rcItem.bottom);
            SelectObject(di->hDC, op); SelectObject(di->hDC, ob);
            DeleteObject(p); DeleteObject(b);
            return TRUE;
        }
        return DefWindowProc(h, msg, w, l);
    }
    case WM_COMMAND: {
        int id = LOWORD(w), code = HIWORD(w);
        if (id == IDC_BIND && code == CBN_SELCHANGE) syncKeyHint();
        if (id == IDC_KEY && code == EN_CHANGE) syncKeyHint();
        if (id == IDC_MODEL && code == CBN_SELCHANGE) syncLatinEnabled(comboSel(g_hModel));
        if (id == IDC_LAUNCH) {
            Opts o = gatherOpts();
            std::wstring args, err;
            if (!buildArgs(o, args, err)) { MessageBoxW(h, err.c_str(), L"Check the options", MB_OK | MB_ICONWARNING); return 0; }
            DeleteFileW(g_logPath.c_str());
            if (!startServer(args, err)) { MessageBoxW(h, fmt(L"Could not start the server:\n%s", err.c_str()).c_str(), L"Launch failed", MB_OK | MB_ICONERROR); return 0; }
            g_adopted = false; g_launchTick = GetTickCount64();
            g_ssl = o.ssl;
            g_base = (g_ssl ? L"https" : L"http") + std::wstring(L"://127.0.0.1:") + std::to_wstring(g_port);
            setUi(true);
            g_light = LIGHT[1]; InvalidateRect(g_hLight, nullptr, TRUE);
            SetWindowTextW(g_hState, fmt(L"Starting (pid %lu)...", g_pid).c_str());
            SetWindowTextW(g_hStat, fmt(L"pid %lu\r\n%s", g_pid, args.c_str()).c_str());
            startWorker();
            return 0;
        }
        if (id == IDC_STOP) {
            stopWorker();
            if (g_hProc) { TerminateProcess(g_hProc, 0); CloseHandle(g_hProc); g_hProc = nullptr; }
            g_adopted = false; g_launchTick = 0;
            setUi(false);
            SetWindowTextW(g_hStat, readLogTail().c_str());
            return 0;
        }
        if (id == IDC_OPEN) {
            if (!g_base.empty()) ShellExecuteW(h, L"open", g_base.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            return 0;
        }
        return 0;
    }
    case WM_STATUS: {
        Status* s = (Status*)l;
        g_light = LIGHT[s->light]; InvalidateRect(g_hLight, nullptr, TRUE);
        SetWindowTextW(g_hState, s->state.c_str());
        SetWindowTextW(g_hStat, s->stat.c_str());
        SetWindowTextW(g_hLog, s->log.c_str());
        SendMessage(g_hLog, EM_SETSEL, (WPARAM)-1, 0); SendMessage(g_hLog, EM_SCROLLCARET, 0, 0);
        delete s;
        return 0;
    }
    case WM_EXITED: {
        Status* s = (Status*)l;
        g_light = LIGHT[0]; InvalidateRect(g_hLight, nullptr, TRUE);
        SetWindowTextW(g_hState, fmt(L"Exited (code %d)", s->exitCode).c_str());
        SetWindowTextW(g_hStat, fmt(L"The server process exited with code %d.\r\nSee the log below.", s->exitCode).c_str());
        SetWindowTextW(g_hLog, s->log.c_str());
        if (g_hProc) { CloseHandle(g_hProc); g_hProc = nullptr; }
        g_launchTick = 0; setUi(false);
        delete s;
        return 0;
    }
    case WM_DESTROY:
        EnterCriticalSection(&g_cs); g_uiAlive = false; LeaveCriticalSection(&g_cs);
        stopWorker();
        if (g_hProc) { CloseHandle(g_hProc); g_hProc = nullptr; }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(h, msg, w, l);
}

// ------------------------------------------------------------------ runtime auto-detect
static bool resolveRuntime() {
    std::vector<std::wstring> cands = { dirJoin(g_here, L"..\\runtime"), dirJoin(g_here, L"..\\build\\bin"), L"C:\\Strix\\runtime" };
    for (auto& c : cands) {
        std::wstring e = dirJoin(c, L"llama-server.exe");
        if (fileExists(e)) { wchar_t full[MAX_PATH]; GetFullPathNameW(e.c_str(), MAX_PATH, full, nullptr); g_exe = full; return true; }
    }
    return false;
}

// ------------------------------------------------------------------ headless modes (no window)
// --selftest: print the command line each model would launch with (arg parity vs strix.bat).
// --smoke:    windowless launch of a light config, wait for /health, print parsed status, kill.
// Both let the launch + HTTP-parse pipeline be verified without clicking Launch.
static int runSelftest() {
    for (int i = 0; i < N_MODELS; i++) {
        Opts o; o.model = i; o.cacheDir = dirJoin(g_here, L"prompt-cache");
        // mirror gatherOpts(): packaged template unless the checkpoint opts into its embedded one
        std::wstring tp = dirJoin(g_here, L"chat_template.jinja");
        if (!MODELS[i].embeddedTpl) o.tmpl = tp;
        std::wstring args, err;
        printf("=== %s ===\n", wtou8(MODELS[i].label).c_str());
        if (buildArgs(o, args, err)) printf("%s\n\n", wtou8(args).c_str());
        else printf("ERROR: %s\n\n", wtou8(err).c_str());
    }
    printf("runtime: %s\n", wtou8(g_exe).c_str());
    // template policy: a checkpoint flagged embeddedTpl must launch with no --chat-template-file;
    // the others must carry the packaged one. Prove both directions here so a regression in the
    // gatherOpts()/buildArgs() template gate is caught without a live server.
    int tplOk = 1;
    for (int i = 0; i < N_MODELS; i++) {
        Opts o; o.model = i; o.cacheDir = dirJoin(g_here, L"prompt-cache");
        std::wstring tp = dirJoin(g_here, L"chat_template.jinja");
        if (!MODELS[i].embeddedTpl) o.tmpl = tp;
        std::wstring a, e; bool built = buildArgs(o, a, e);
        bool has = built && a.find(L"--chat-template-file") != std::wstring::npos;
        bool want = !MODELS[i].embeddedTpl;
        if (has != want) tplOk = 0;
    }
    printf("template policy (embedded checkpoints omit --chat-template-file): %s\n",
           tplOk ? "yes" : "NO (BUG)");
    // latin policy: a checkpoint flagged noLatin must launch WITHOUT --spec-draft-vocab-latin even
    // when the trim box would be checked (its own-head draft has no d2t, so the server refuses that
    // combination); the others must carry it. Proves the buildArgs gate headlessly.
    int latinOk = 1;
    for (int i = 0; i < N_MODELS; i++) {
        Opts o; o.model = i; o.cacheDir = dirJoin(g_here, L"prompt-cache"); o.latin = true;
        std::wstring a, e; bool built = buildArgs(o, a, e);
        bool has = built && a.find(L"--spec-draft-vocab-latin") != std::wstring::npos;
        bool want = !MODELS[i].noLatin;
        if (has != want) latinOk = 0;
    }
    printf("latin policy (own-head-draft checkpoints omit --spec-draft-vocab-latin): %s\n",
           latinOk ? "yes" : "NO (BUG)");
    // default policy: what a fresh launch gets. Both directions are asserted because the default is
    // now the EXL3 checkpoint with an UNQUANTIZED KV cache - a regression here (array reorder, or a
    // stray -ctk) changes what every user gets without anyone touching a checkbox.
    int defOk = 1;
    {
        Opts o; o.cacheDir = dirJoin(g_here, L"prompt-cache");
        std::wstring a, e; bool built = buildArgs(o, a, e);
        bool okRepo = built && a.find(MODELS[DEFAULT_MODEL].repo) != std::wstring::npos;
        bool okNoQ8 = built && a.find(L"-ctk") == std::wstring::npos;
        printf("default checkpoint: %s\n", wtou8(MODELS[DEFAULT_MODEL].label).c_str());
        printf("default kv unquantized: %s ; default checkpoint in default args: %s\n",
               okNoQ8 ? "yes" : "NO (BUG)", okRepo ? "yes" : "NO (BUG)");
        if (!okNoQ8 || !okRepo) defOk = 0;
    }
    // the API key is optional: 0.0.0.0 with no key must still build (and emit no --api-key), while a
    // provided key must be passed through.
    Opts net0; net0.model = DEFAULT_MODEL; net0.loopback = false; net0.cacheDir = dirJoin(g_here, L"prompt-cache");
    std::wstring a2, e2;
    bool okNoKey = buildArgs(net0, a2, e2) && a2.find(L"--api-key") == std::wstring::npos;
    Opts net1 = net0; net1.apiKey = L"k";
    std::wstring a3, e3;
    bool okKey = buildArgs(net1, a3, e3) && a3.find(L"--api-key") != std::wstring::npos;
    printf("network no-key allowed: %s ; with-key passes --api-key: %s\n",
           okNoKey ? "yes" : "NO (BUG)", okKey ? "yes" : "NO (BUG)");
    return (okNoKey && okKey && tplOk && latinOk && defOk) ? 0 : 2;
}
static int runStatus() {
    HINTERNET s = WinHttpOpen(L"StrixLauncher/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    WinHttpSetTimeouts(s, 1500, 1500, 2000, 2000);
    g_pid = findServerPid(g_port);
    g_adopted = g_pid != 0;
    Status st = pollOnce(s);
    printf("port %d owner pid = %lu\n", g_port, g_pid);
    printf("ok=%d\n%s", st.ok ? 1 : 0, wtou8(st.ok ? st.stat : st.state).c_str());
    if (s) WinHttpCloseHandle(s);
    return st.ok ? 0 : 1;
}
static int runWatch(int seconds) {
    HINTERNET s = WinHttpOpen(L"StrixLauncher/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    WinHttpSetTimeouts(s, 1500, 1500, 2000, 2000);
    g_pid = findServerPid(g_port);
    g_adopted = g_pid != 0;
    printf("watching port %d (pid %lu) for %ds, 1s polls. Ctrl+C to stop.\n", g_port, g_pid, seconds);
    ULONGLONG end = GetTickCount64() + (ULONGLONG)seconds * 1000;
    int rc = 1;
    while (GetTickCount64() < end) {
        Status st = pollOnce(s);
        if (st.ok) {
            rc = 0;
            // print one compact line per poll so a live rate is visible as it changes
            std::string one = wtou8(st.stat);
            std::string perf = "?";
            size_t pp = one.find("perf  : ");
            if (pp != std::string::npos) { size_t e = one.find("\r\n", pp); perf = one.substr(pp + 8, e - pp - 8); }
            // print the per-slot line too: without it a "live" decode reading cannot be told apart
            // from a stale counter on an idle slot -- the processing flags are the proof.
            std::string slotline = "";
            size_t sp = one.find("slots : ");
            if (sp != std::string::npos) { size_t e = one.find("\r\n", sp); slotline = " | " + one.substr(sp, e - sp); }
            printf("[%5.1fs] %s%s\n", (seconds*1000.0 - (end - GetTickCount64())) / 1000.0, perf.c_str(), slotline.c_str());
        } else {
            printf("[%5.1fs] not ready\n", (seconds*1000.0 - (end - GetTickCount64())) / 1000.0);
        }
        fflush(stdout);
        for (int i = 0; i < 10; i++) Sleep(100);
    }
    if (s) WinHttpCloseHandle(s);
    return rc;
}
static int runSmoke() {
    // refuse rather than stack a second server on a live one (and on UMA, OOM trying)
    DWORD busy = findServerPid(g_port);
    if (busy) { printf("port %d already owned by pid %lu - stop it first, or use --status to just read it\n", g_port, busy); return 3; }
    // model 0 (UD) is DELIBERATE, not the shipped default: --smoke must stay light, and the default
    // checkpoint is a ~104 GB six-shard download. Do not replace this with DEFAULT_MODEL.
    Opts o; o.model = 0; o.ctx = 16384; o.parallel = 1; o.yarn = false; o.vision = false;
    o.mtp = false; o.cache = 2; o.loopback = true; o.cacheDir = dirJoin(g_here, L"prompt-cache");
    std::wstring args, err;
    if (!buildArgs(o, args, err)) { printf("buildArgs failed: %s\n", wtou8(err).c_str()); return 1; }
    printf("cmd: \"%s\" %s\n", wtou8(g_exe).c_str(), wtou8(args).c_str());
    DeleteFileW(g_logPath.c_str());
    if (!startServer(args, err)) { printf("start failed: %s\n", wtou8(err).c_str()); return 1; }
    printf("pid=%lu\n", g_pid);
    g_ssl = false;
    HINTERNET s = WinHttpOpen(L"StrixLauncher/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    WinHttpSetTimeouts(s, 1500, 1500, 2000, 2000);
    bool ok = false;
    for (int i = 0; i < 60; i++) {
        Sleep(3000);
        DWORD code = STILL_ACTIVE; GetExitCodeProcess(g_hProc, &code);
        if (code != STILL_ACTIVE) { printf("process exited early, code=%lu\n", code); break; }
        Status st = pollOnce(s);
        if (st.ok) { ok = true; printf("health ok after %ds\n", (i + 1) * 3); printf("%s", wtou8(st.stat).c_str()); break; }
    }
    if (!ok) printf("health never came up\n");
    printf("log tail:\n%s\n", wtou8(readLogTail()).c_str());
    TerminateProcess(g_hProc, 0); CloseHandle(g_hProc); g_hProc = nullptr;
    if (s) WinHttpCloseHandle(s);
    printf("stopped\n");
    return ok ? 0 : 1;
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, LPWSTR cmdLine, int show) {
    g_hInst = inst;
    InitializeCriticalSection(&g_cs);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    g_dpi = GetDpiForSystem();   // fonts and every coordinate scale from this
    INITCOMMONCONTROLSEX icc = { sizeof icc, ICC_STANDARD_CLASSES | ICC_UPDOWN_CLASS };
    InitCommonControlsEx(&icc);

    wchar_t mod[MAX_PATH]; GetModuleFileNameW(inst, mod, MAX_PATH);
    g_here = mod; size_t sl = g_here.find_last_of(L"\\/"); g_here = g_here.substr(0, sl);
    wchar_t tmp[MAX_PATH]; GetTempPathW(MAX_PATH, tmp); g_logPath = std::wstring(tmp) + L"strix_gui_server.log";

    if (!resolveRuntime()) {
        MessageBoxW(nullptr, fmt(L"llama-server.exe not found next to this launcher or in C:\\Strix\\runtime.\nLooked in:\n  %s\\..\\runtime\n  %s\\..\\build\\bin\n  C:\\Strix\\runtime", g_here.c_str(), g_here.c_str()).c_str(),
                    L"Strix launcher", MB_OK | MB_ICONERROR);
        return 1;
    }

    // headless modes: attach a console so stdout is visible when launched from a shell
    if (cmdLine && *cmdLine) {
        bool selftest = wcsstr(cmdLine, L"--selftest") != nullptr;
        bool smoke = wcsstr(cmdLine, L"--smoke") != nullptr;
        bool status = wcsstr(cmdLine, L"--status") != nullptr;
        bool watch = wcsstr(cmdLine, L"--watch") != nullptr;
        // --port N lets the headless modes point at a non-default server (e.g. a quiet test instance)
        // instead of always hitting 9931, which is the only thing that can make an idle control idle.
        {
            const wchar_t* flag = L"--port ";
            const wchar_t* pp = wcsstr(cmdLine, flag);
            if (pp) { int v = _wtoi(pp + 7); if (v > 0 && v < 65536) g_port = v; }
        }
        if (selftest || smoke || status || watch) {
            // A GUI-subsystem exe has no console. When launched from a shell, stdout is already
            // inherited (a pipe/file) and printf just works. Only when stdout is a console or
            // absent do we need to attach the parent console -- and we must NOT freopen over a
            // redirected handle, or the output goes to the screen instead of the caller's pipe.
            HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
            bool redirected = out && out != INVALID_HANDLE_VALUE
                              && GetFileType(out) != FILE_TYPE_CHAR && GetFileType(out) != FILE_TYPE_UNKNOWN;
            if (!redirected) {
                if (AttachConsole(ATTACH_PARENT_PROCESS)) {
                    freopen("CONOUT$", "w", stdout);
                    freopen("CONOUT$", "w", stderr);
                }
            }
            int rc = selftest ? runSelftest() : (status ? runStatus() : (watch ? runWatch(25) : runSmoke()));
            DeleteCriticalSection(&g_cs);
            return rc;
        }
    }

    // fonts
    HDC dc = GetDC(nullptr);
    LOGFONT lf = {}; lf.lfHeight = -MulDiv(9, GetDeviceCaps(dc, LOGPIXELSY), 72); lf.lfCharSet = DEFAULT_CHARSET;
    wcscpy_s(lf.lfFaceName, L"Segoe UI"); g_fontUi = CreateFontIndirect(&lf);
    wcscpy_s(lf.lfFaceName, L"Consolas"); g_fontMono = CreateFontIndirect(&lf);
    ReleaseDC(nullptr, dc);

    WNDCLASSEXW wc = { sizeof wc };
    wc.lpfnWndProc = WndProc; wc.hInstance = inst; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1); wc.lpszClassName = L"StrixLauncher";
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    RegisterClassExW(&wc);

    RECT rc = { 0, 0, SX(704), SX(632) };
    AdjustWindowRect(&rc, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
    g_hwnd = CreateWindowExW(0, L"StrixLauncher", L"Strix launcher", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                             CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
                             nullptr, nullptr, inst, nullptr);
    if (!g_hwnd) return 1;

    SetWindowTextW(g_hRuntime, fmt(L"runtime:  %s", dirJoin(g_here, L"llama-server.exe").c_str()).c_str());
    syncKeyHint();
    setUi(false);
    SetWindowTextW(g_hStat, fmt(L"Pick a checkpoint and press Launch.\r\nRuntime: %s", g_exe.c_str()).c_str());

    ShowWindow(g_hwnd, show ? show : SW_SHOW);
    UpdateWindow(g_hwnd);

    // re-open case: adopt a server left running outside this window
    attachExisting();

    MSG m;
    while (GetMessage(&m, nullptr, 0, 0)) { TranslateMessage(&m); DispatchMessage(&m); }

    DeleteCriticalSection(&g_cs);
    return (int)m.wParam;
}
