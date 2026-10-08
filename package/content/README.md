# Franken-Llama — gfx1151 HIP server package

Self-contained llama.cpp server for the **AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151)**.
The runtime is fully self-contained: every native DLL it imports is in `runtime/`, so the only
external requirement is the installed AMD graphics driver. HTTPS/TLS is built in (OpenSSL).

## Contents
```
launcher\strix.bat            double-click launcher (no gate setup: the binary bakes them)
launcher\chat_template.jinja  Qwen3.8-Flash-Next chat template (optional; model-embedded works too)
runtime\                      llama-server.exe + the DLLs it actually imports (20 modules) +
                              rocblas/hipblaslt kernel data. Nothing else from the build ships:
                              no bench/quantize/completion/perplexity tools, and no hiprtc — the
                              runtime JIT, never loaded here because every shipped kernel is
                              precompiled for gfx1151. Includes libssl-4-x64.dll / libcrypto-4-x64.dll
                              (HTTPS) and mtmd.dll (vision tower; always present, see "Vision")
runtime\.kpack\               the rocBLAS kernel pack (blas_lib_gfx1151.kpack) its DLLs load their
                              own GEMV kernels from
```
Neither the target model nor the MTP draft is bundled — both are resolved and downloaded from
Hugging Face on first launch (into the standard HF cache) and reused from there. That is why the
package is ~330 MB instead of ~2.2 GB. See "Speculative decode (MTP draft)".

## Quick start
The launcher does **not** scan for models. It uses the `MODEL` / `MTP` values in the `EDIT ME` block
(or in the environment), which default to the HF checkpoints resolved by the server itself.
1. Unzip anywhere.
2. Either use the default `MODEL` (an `org/repo:quant` checkpoint the server downloads/resolves), or
   set `MODEL` to a local gguf — for a multi-shard model point it at the **first** split
   (`...-00001-of-00003.gguf`) and keep the other shards beside it.
3. The MTP draft is fetched from Hugging Face automatically (see "Speculative decode (MTP draft)");
   set `MTP=none` to run without speculative decoding, or `MTP` to your own draft file/repo.
4. Double-click `strix.bat`. Server listens on `127.0.0.1:9931` (OpenAI-compatible `/v1/chat/completions`).

## Overriding defaults
Set these in the environment before launching (or edit the `EDIT ME` block in `strix.bat`).
An already-set environment value always wins over the file's default:
- `MODEL`  — model gguf path, or an `org/repo:quant` HF checkpoint (default: the HF `UD-IQ4_XS` checkpoint)
- `MTP`    — MTP draft: an HF `org/repo[:quant]` (default: the shared MTP head in the model's repo,
  downloaded on first launch), a local `.gguf` path, or `none` to disable speculative decoding
- `MTP_FILE` — in-repo path to the draft file when `MTP` is an HF repo whose draft is not at the
  repo root (default: `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`)
- `LATIN`      — `1` trims the draft's borrowed LM head to a 32768-token Latin shortlist **at load**
  (default `1`); `0` keeps the full-vocabulary head. See "Speculative decode (MTP draft)".
- `CTX`    — context size in tokens (default 262144, the size the model was trained on)
- `YARN`     — `1` enables YaRN rope scaling so `CTX` can exceed the trained 262144 (default `0`)
- `YARN_SCALE` — YaRN expansion factor when `YARN=1`; usable context is about 262144 x `YARN_SCALE`
  (default `2`)
- `PORT`   — listen port (default 9931)
- `HOSTIP` — bind address (default 127.0.0.1)
- `PARALLEL`   — concurrent conversations / slots (default `1`)
- `KVCHAIN`    — `1` enables the disk-backed hash-chain KV cache (default `1`; set `0` for RAM/VRAM only)
- `KVCHAIN_GB` — disk quota in GiB for the chain when enabled (default 100)
- `SSL`        — `1` serves HTTPS/TLS instead of plain HTTP (default `0`)
- `SSL_KEY`    — PEM private key path (default `launcher\key.pem`, only used when `SSL=1`)
- `SSL_CERT`   — PEM certificate chain path (default `launcher\cert.pem`, only used when `SSL=1`)
- `VISION`     — `1` enables image input by loading a vision projector (default `0`, text only)
- `MMPROJ`     — explicit local projector gguf; implies `VISION=1` and skips auto-resolution

## Context size and YaRN
The model is trained to **262144** tokens, which is the default `CTX`. Within that range no rope
scaling is wanted or needed.

To go beyond it, enable YaRN, which rescales the rotary positions so longer contexts stay in
distribution:
```
set YARN=1
set YARN_SCALE=2
set CTX=524288
strix.bat
```
`YARN_SCALE` is the expansion factor, so the usable context is roughly `262144 x YARN_SCALE`; the
example above gives 512k. The launcher passes `--rope-scaling yarn --rope-scale %YARN_SCALE%
--yarn-orig-ctx 262144`.

Measured on this box at scale 2 / 512k, single slot, KV in f16, `-ub 8192`: the server loads and runs
using about **97 GB of the ~108 GB available to the GPU** (a 96 GB dedicated carve plus 16 GB of
shared memory), i.e. roughly 10 GB of headroom. Context beyond that
needs KV quantisation (f16 KV costs about 6.5 GB per 262k) or a smaller `-ub`.

The launcher refuses `YARN=1` while `CTX` is still at or below 262144 — scaling there buys nothing and
costs accuracy.

## Vision / image input
Off by default: the shipping command line passes `--no-mmproj`, so the server is text-only and
no projector is downloaded. The vision tower is fully supported and needs no extra files in the
package — `runtime\mtmd.dll` is always present (the server imports it statically and cannot start
without it), and the projector weights come from the same Hugging Face repo as the model.

Turn it on:
```
set VISION=1
strix.bat
```
With a `-hf` checkpoint, `VISION=1` adds `--mmproj-auto`: the server resolves and downloads the
repo's projector (`mmproj-BF16.gguf`, ~865 MB) into the HF cache and image input becomes available
on the OpenAI API (`content` parts of type `image_url` / `image`). For a local `MODEL` gguf, or to
pick a specific precision, point at the projector file directly:
```
set MMPROJ=C:\models\mmproj-BF16.gguf
strix.bat
```
`MMPROJ` implies `VISION=1` and fails loudly if the file is missing. Budget VRAM for it yourself:
the projector is offloaded **in addition to** the 262k context, and `--fit off` means nothing is
shrunk automatically to make room.

## HTTPS / TLS
The server binary is linked against OpenSSL, so it speaks TLS natively — no reverse proxy needed.

**Inbound (serving HTTPS).** Off by default; turn it on with a cert and key:
```
set SSL=1
set SSL_KEY=C:\path\key.pem
set SSL_CERT=C:\path\cert.pem
strix.bat
```
If `SSL_KEY`/`SSL_CERT` are unset the launcher looks for `key.pem` and `cert.pem` in `launcher\`.
It fails loudly if `SSL=1` and a file is missing rather than silently serving plain HTTP. The server
logs `listening on https://...` when TLS is active. A throwaway self-signed pair:
```
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes ^
  -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
```
For anything beyond loopback testing, use a real certificate: clients must be able to verify it.

**Outbound (model downloads).** Independent of `SSL`. Downloads over `https://` — including the
`-hf` / `-hfd` Hugging Face path the launcher uses by default — are fetched by the server's own HTTP
client and are **verified** against the Windows certificate store (root + CA). Untrusted certificates
are rejected, so a MITM cannot silently swap model bytes. No CA bundle to install.

Verified on the build in this package: TLS 1.3 / `TLS_AES_256_GCM_SHA384` on `/health`,
`/v1/models` and `/v1/chat/completions`; plain HTTP against a TLS listener is refused; a cold
`https://` model download from a CA-signed host succeeds; a self-signed HTTPS host is rejected.

## Speculative decode (MTP draft)
The default configuration runs an MTP (multi-token-prediction) draft to speed up decode. The draft
is **not bundled**: the launcher points the server at the shared MTP head published in the same
Hugging Face repo as the model (`-hfd unsloth/Qwen3.8-Flash-Next-GGUF -md MTP/mtp-...-shared-Q4_K_M.gguf`),
which the server downloads on first launch (~1.8 GB, cached afterwards) and reuses. To disable it,
set `MTP=none`; to use your own, set `MTP` to a local `.gguf` or another `org/repo[:quant]`.

### Latin draft-vocab trim (`LATIN=1`, default)
That shared draft carries **no output projection of its own** — it borrows the target's full-vocab
head, a `[2560, 248320]` matrix streamed on **every draft step** (pure memory bandwidth, ~521 MB).
`LATIN=1` makes the server, **at load time**, gather that borrowed head down to a 32768-token
frequency-ranked **Latin** shortlist (32492 frequent tokens + all 276 control tokens) and attach a
`d2t` map back to the full vocabulary. The draft head then reads ~69 MB per draft step instead of
~521 MB. Because the trim is built in-process from the target's own head, **no pre-trimmed draft file
is needed** — the plain HF shared draft is enough.

The target still verifies over the **full** vocabulary, so the output distribution is provably
unchanged — the only possible cost is draft acceptance. The trim costs ~1 s at load.

`--spec-draft-vocab-latin` is a real server argument and the launcher passes it whenever `LATIN=1`.
It makes the trim a hard requirement: if the trim cannot be built, the server refuses to start
instead of silently running the untrimmed head. The log line
`draft LM head trimmed at load from q6_K [2560 x 248320] -> q6_K [2560 x 32768]` confirms engagement.

**This trim is tuned for Latin/English output.** CJK token ids in this vocabulary sit at 95726–145519
and a Latin shortlist covers only ~32% of Chinese text, which would collapse acceptance and make
speculative decoding a net loss. If you serve Chinese or other CJK, set `LATIN=0`:
```
set LATIN=0
strix.bat
```
That keeps the full-vocabulary head (slower per draft step, but handles all scripts). You can also
point `MTP` at your own draft file — with `LATIN=1` a local draft must already carry a `d2t` tensor.

## Requirements / notes
- GPU: gfx1151 (Radeon 8060S) with its installed driver. No other runtime install needed.
- HTTPS needs no extra install: `libssl-4-x64.dll` / `libcrypto-4-x64.dll` ship in `runtime\` and
  outbound verification uses the Windows certificate store.
- Paging file 128 GB recommended for deep-context commit headroom (the server warns but still runs
  if commit headroom is low).
- This is a **chat** model: use `/v1/chat/completions`, not the raw `/completion` endpoint
  (raw completion makes it emit end-of-sequence after one token).
- The **server binary carries its own validated configuration.** It sets every internal tuning
  variable in its own process environment at startup, so there is nothing to configure in a shell,
  in `README.md`, or in `strix.bat` — and a stray `LLAMA_*` / `GGML_*` / `STRIX_*` variable left over
  from another program is ignored rather than silently changing behaviour. The shipped configuration
  is validated as a set; do not try to tune it.
- The one exception, for developers only: `STRIX_ENV_BAKE=0` leaves the inherited environment
  untouched so gates can be steered by hand. It is an escape hatch, not a tuning knob — a wrong gate
  value can corrupt output.

## Disk KV cache (`KVCHAIN=1`, default)
Streams per-chunk KV state to `launcher\kv-chain\` keyed by a content hash-chain, so a shared prompt
prefix is a **cache hit even after a restart** — instead of being evicted from scarce RAM/VRAM. This
suits a unified-memory box where RAM is tight and SSD is cheap.
- On by default. Disable with `set KVCHAIN=0` to keep the cache in RAM/VRAM only.
- `KVCHAIN_GB` caps the folder size (LRU by file mtime); the state survives a reboot.
- Chunks are sized by the batch `-ub` (8192).

## Multiple slots (`PARALLEL>1`)
`PARALLEL>1` serves several conversations at once. It **requires** a unified KV cache, which the launcher
sets for you (`--kv-unified`); with a per-slot cache the dense attention mask returns and reserves ~4 GiB.
So set only `PARALLEL` and let the launcher pair it with the right cache mode.
`PARALLEL=1` (default) serves one conversation at a time.
