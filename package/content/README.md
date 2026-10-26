# Franken-Llama — gfx1151 HIP server package

Self-contained llama.cpp server for the **AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151)**.
The runtime is fully self-contained: every native DLL it imports is in `runtime/`, so the only
external requirement is the installed AMD graphics driver. HTTPS/TLS is built in (OpenSSL).

## Contents
```
launcher\strix.bat            double-click launcher (no gate setup: the binary bakes them)
launcher\strix_gui.exe        graphical launcher: pick a checkpoint, tick the knobs, watch live stats
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
- `CTX`    — context size in tokens (default 524288). The model is trained to 262144, so the default
  context rides on `YARN=1`; set `YARN=0` to stay inside the trained range.
- `YARN`     — `1` enables YaRN rope scaling so `CTX` can exceed the trained 262144 (default `1`)
- `YARN_SCALE` — YaRN expansion factor when `YARN=1`; usable context is about 262144 x `YARN_SCALE`
  (default `2`)
- `PORT`   — listen port (default 9931)
- `HOSTIP` — bind address (default 127.0.0.1)
- `PARALLEL`   — concurrent conversations / slots (default `2`)
- `KVQ8`       — `1` stores the KV cache, and the MTP draft's, as q8_0 instead of f16 (default `1`);
  `0` keeps both unquantized. Roughly halves the cache — about 6.4 GB at 512k context. See
  "Context size and YaRN".
- `CACHE`      — prompt cache on disk: `kvchain` (default) / `disk` / `auto` / `off`. Exactly one
  mechanism runs — the server refuses to start with both. See "Prompt cache on disk".
- `KVCHAIN_GB` — disk quota in GiB for kv-chain mode (default 100)
- `PCCACHE_GB` — disk quota in GiB for disk-prompt-cache mode (default 64)
- `SSL`        — `1` serves HTTPS/TLS instead of plain HTTP (default `0`)
- `SSL_KEY`    — PEM private key path (default `launcher\key.pem`, only used when `SSL=1`)
- `SSL_CERT`   — PEM certificate chain path (default `launcher\cert.pem`, only used when `SSL=1`)
- `VISION`     — `1` enables image input by loading a vision projector (default `1`; `0` = text only)
- `MMPROJ`     — explicit local projector gguf; implies `VISION=1` and skips auto-resolution
- `CTPL`       — chat template file to use (default `launcher\chat_template.jinja`). Set it empty to
  use the template embedded in the model instead.
- `CACHE_DIR`  — directory holding the on-disk prompt cache (default `launcher\prompt-cache`). The
  quota knobs above apply inside it; see "Prompt cache on disk".

## Context size and YaRN
The model is trained to **262144** tokens. The shipping default goes past that: `CTX=524288` together
with `YARN=1`, so two conversations share a ~512k window out of the box. To stay inside the trained
range instead, turn YaRN off and drop the context:
```
set YARN=0
set CTX=262144
strix.bat
```
Within the trained range no rope scaling is wanted or needed, and the launcher refuses `YARN=1` while
`CTX` is still at or below 262144 — scaling there buys nothing and costs accuracy.

YaRN rescales the rotary positions so a longer context stays in distribution. `YARN_SCALE` is the
expansion factor, so the usable context is roughly `262144 x YARN_SCALE`; the default scale of 2
gives the default 524288. The launcher passes `--rope-scaling yarn --rope-scale %YARN_SCALE%
--yarn-orig-ctx 262144`.

Measured on this box at scale 2 / 512k, single slot, KV in **f16** (`KVQ8=0`), `-ub 8192`: the server
loads and runs using about **97 GB of the ~108 GB available to the GPU** (a 96 GB dedicated carve plus
16 GB of shared memory), i.e. roughly 10 GB of headroom.

The default KV type is q8_0 (`KVQ8=1`), which roughly halves that cache: f16 costs about 6.5 GB per
262k of context, q8_0 about 3.2 GB, so the 512k default is about **6.4 GB smaller** than the f16
figure above. Context beyond that needs a smaller `-ub`.

### Batch size (`-b` / `-ub`) - a tuning knob, not a validated default

Both launchers hardcode `-b 8192 -ub 8192`. Unlike the KV/context knobs above, the batch sizes are not
exposed as environment variables: to change them, edit the `ARGS` line in `strix.bat` (or the
`addLine` call in `strix_gui.cpp` and rebuild).

They are the knob to reach for when the memory arithmetic above does not fit: `-ub` sets the size of
the dense activation tensors, so it drives prefill peak memory roughly in proportion, and it is the
first thing to lower if a deep-context load fails to commit. The trade goes both ways - a larger
`-ub` means fewer, wider GEMMs, which can be more efficient on some weight layouts and less on others.

**8192 is the validated launch configuration; nothing else has been validated that way.** Batch size
is not a free knob, because it changes the order in which reductions accumulate: a different `-ub`
can produce different generated text at the same temperature, which is exactly why the perplexity
gate for this package deliberately runs a much smaller fixed batch (`-b 512 -ub 4`) to get a
reproducible number. So treat any other value as an experiment rather than a tuning result: confirm
the model still loads, and do not read a speed difference as a win until output still looks right.
No alternative batch size has been measured as a shipping launch configuration for this package.

## Vision / image input
On by default: the shipping command line passes `--mmproj-auto`, so the server resolves and
downloads the repo's projector and image input is available out of the box. The vision tower needs
no extra files in the package — `runtime\mtmd.dll` is always present (the server imports it
statically and cannot start without it), and the projector weights come from the same Hugging Face
repo as the model.

Turn it off (text only, no projector download):
```
set VISION=0
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

## Graphical launcher (`strix_gui.exe`)
A small native window for the same server, for people who would rather click than edit a file.
**Double-click `strix_gui.exe`.** It is a self-contained program (no PowerShell, no runtime to
install) and starts the server with no console window behind it. From a shell it also has headless
modes: `strix_gui.exe --selftest` prints the exact command line each checkpoint would launch, and
`--status` reads a server already running on the port.
It offers a **checkpoint dropdown**, checkboxes for the quick-access knobs (vision, q8_0 KV cache,
MTP speculative decode, the Latin draft trim, YaRN, HTTPS), a context-size and slot picker, and the
prompt-cache mode. **Launch** starts the server with no console window; **Stop** shuts it down.

The dropdown offers four checkpoints: `UD-IQ4_XS` (the shipping default, and the one every number in
this README was measured on), `ISTA IQ3_S` (per-tensor mixed precision, smaller), `Swift 1.5 IQ4_XS`
(an agent/reasoning fine-tune of the same base by a different quantizer, and the largest at about 98 GB
across its three shards) and `Huihui IQ4_XS` (an abliterated, refusal-disabled re-quant of the same
UD-IQ4_XS base — same quant, same size class). All four are the same architecture and share the same MTP
draft head. The launcher resolves and downloads whichever checkpoint you pick on first launch, so the
first start of a new one is a large download.

**`UD-IQ4_XS`, `ISTA IQ3_S` and `Huihui IQ4_XS` run with the packaged chat template**, not the one
embedded in their own file: the launcher passes `--chat-template-file launcher\chat_template.jinja` for
them. **`Swift 1.5 IQ4_XS` runs with its own embedded template** — it is a fine-tune whose template
differs from the base model's, so the launcher leaves it unmodified (no `--chat-template-file`). For any
checkpoint you can override the packaged template by setting `CTPL` to your own file, or set `CTPL=`
(empty) to force the model-embedded template for the others as well.

**Serving** defaults to *This computer only* (`127.0.0.1`). *All interfaces* (`0.0.0.0`) publishes the
endpoint to the local network. An **API key** is optional: set one and clients must send
`Authorization: Bearer <key>`; leave it blank and the launcher warns that the port is open to your LAN
(anything on it can use the model and read the conversations), then starts anyway.

While it runs, a status panel polls the server over HTTP and shows the loaded model and type, context
size, slot count, whether vision is active, uptime, a live **prefill / decode tokens-per-second**
reading, how many requests are processing or waiting, speculative-draft acceptance, and a per-slot
view. A coloured light distinguishes starting / ready / idle / stopped, and a tail of the server log
is shown at the bottom so a failed load is visible instead of silent.

Closing the window offers to stop the server with it; choosing *No* leaves it running, and reopening the
launcher finds the live server on the port and re-attaches to it (status panel and **Stop** work on it,
**Launch** stays disabled so a second server is not stacked on the same port). A server started from
`strix.bat` is adopted the same way.

`strix.bat` remains the way to launch with anything the window does not expose — it is unaffected by
the GUI, and the two are meant to be used side by side. Nothing is configured in a shell either way:
the GUI's knobs map one-to-one onto the `strix.bat` variables above and produce the same command line.

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

## Prompt cache on disk (`CACHE=kvchain`, default)
Finished or evicted conversations come back from SSD instead of being re-prefilled — on this
unified-memory box RAM is tight and SSD is cheap. Exactly one mechanism runs; the default,
**kv-chain**, covers every config, text and image alike:

- **kv-chain** (`CACHE=kvchain`, the default; `launcher\prompt-cache\`, quota `KVCHAIN_GB`): a content
  hash-chain of KV chunks written as prefill runs. It is the only mechanism that stores conversations
  containing **images** — the text between and after screenshots stays cached (agentic screenshot loops
  keep their prefix), and a changed image correctly invalidates only the text after it. Chunks are sized
  by the batch `-ub` (8192); state survives a reboot.
- **text-only → disk prompt cache** (`launcher\prompt-cache\`, quota `PCCACHE_GB`): the tier under
  the RAM prompt cache — a conversation's attention rows are written once per 4096 positions as they
  are computed, its recurrent state when it leaves its slot; checkpoints page to disk so a warm
  conversation holds no cold RAM. A ~170K-token conversation reads back in a couple of seconds,
  byte-identical to keeping it resident. Conversations containing images are **not** stored.
- `CACHE=off` keeps everything in RAM/VRAM only. `CACHE=auto` picks kv-chain when vision is on and
  the disk tier otherwise. Force one mechanism with `CACHE=kvchain` /
  `CACHE=disk` (e.g. `CACHE=disk` to save the ~865 MB projector's VRAM while text chats still
  persist). The server exits at startup if both are ever requested.

(Legacy: `KVCHAIN=0`/`KVCHAIN=1` still work and map to `CACHE=off`/`CACHE=kvchain`.)

**Changing the cache format invalidates what is already on disk.** The chain is keyed by a hash of
the KV cache type (`KVQ8`), the rope scaling (`YARN`/`CTX`), the slot count (`PARALLEL`) and the
chunk width, so flipping any of them orphans every cached chunk — nothing is corrupted, but the whole
corpus re-prefills once as conversations return. The server logs one line at startup when this
happens (`root changed ... N cached chunk file(s) ... are now unreachable`). Budget for that one-time
re-prefill before changing `KVQ8` on a warm box.

**Long multi-turn continuations no longer crash the server.** When a conversation resumes from a chunk
boundary that doesn't line up with where the previous answer stored its recurrent state, the cache can't
roll that sequence back to the rung — the server now drops the sequence and re-prefills the prompt from
its start (a one-time slowdown on that turn) instead of exiting. The log says `dropping the sequence`
when this happens. To restore the old fail-fast behaviour, set `STRIXLLAMA_SEQ_RM_DROP=0`.

## Multiple slots (`PARALLEL`)
The default `PARALLEL=2` serves two conversations at once. It **requires** a unified KV cache, which
the launcher sets for you (`--kv-unified`); with a per-slot cache the dense attention mask returns and
reserves ~4 GiB. So set only `PARALLEL` and let the launcher pair it with the right cache mode.
`PARALLEL=1` serves one conversation at a time.
