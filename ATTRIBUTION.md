# Attribution

Franken-Llama is assembled from several upstream projects. This file names them and what each
contributes. The project as a whole is MIT-licensed, matching llama.cpp; third-party components
keep their own licenses, listed under `licenses/` and in `vendor/`.

## Base: llama.cpp

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — MIT. The entire inference
  engine, server, tokenizer, quantisation and GGUF format. Everything in `ggml/`, `src/`,
  `common/`, `include/`, `tools/` originates here.
- **[ggml-org/ggml](https://github.com/ggml-org/ggml)** — the tensor library llama.cpp is built on.

## Direct parent: Rulith Inference

- **[rulith-dev/llama.cpp](https://github.com/rulith-dev/llama.cpp)**, branch `rulith` — MIT.
  This fork is built on Rulith's patched llama.cpp (itself a fork of
  [pwilkin/llama.cpp](https://github.com/pwilkin/llama.cpp)), which carries the work that makes
  Qwen3.8-Flash-Next run well on a single AMD Strix Halo machine: the `gfx1151` HIP kernels for
  hyper-connections, MoE routing, gated-delta-net, the QSA sparse-attention indexer, and the
  model definition for the `qwen4exp` architecture.
- **[rulith-dev/rulith-inference](https://github.com/rulith-dev/rulith-inference)** — releases,
  documentation and measurements for that patch set. Thank you for the base work and for keeping
  it public.

## Disk KV cache

- **[riven8192/llama.cpp-kvchain](https://github.com/riven8192/llama.cpp-kvchain)**, branch
  `hash-chain-kv` — the origin of the disk-backed hash-chain KV cache (`--kv-chain-dir`,
  `--kv-chain-limit-gb`), which streams per-chunk KV state to disk keyed by a content hash so a
  shared prompt prefix survives a process restart. We ported that feature across forks into this
  tree (the state serialisation entry points, the chunk store, and the restore path), then added
  our own changes for the async write path, the sparse-attention indexer mirror, and Windows
  overlapped I/O. The design and the hash-chunking idea are theirs.

## Chat template

- **[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)**
  on Hugging Face — the Jinja chat template shipped as `package/content/launcher/chat_template.jinja`
  (template version string `qwen3.8-froggeric-v22.5`). It fixes several bugs in the official
  Qwen chat templates and handles reasoning/tool-call formatting. See also
  [huggingface.co/froggeric](https://huggingface.co/froggeric).

## Model weights

Model weights are **not** part of this repository. The launcher resolves them from Hugging Face
on first use:

- **[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** —
  the quantised target model (`UD-IQ4_XS` by default) and the MTP speculative-decoding draft
  (`MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`).
- The underlying model architecture is Qwen; the quantisations are Unsloth's GGUF conversions.

## Toolchain

- **[ROCm / TheRock](https://rocm.docs.amd.com/projects/TheRock/)** (AMD) — the HIP compiler and
  `gfx1151` device libraries used to build the backend.
- **[OpenSSL](https://www.openssl.org/)** — TLS, both for serving HTTPS and for verifying
  outbound model downloads.

## Vendored third-party code

Bundled under `vendor/` and `licenses/`, each under its own license:

- [cpp-httplib](https://github.com/yhirose/cpp-httplib) — MIT — HTTP server used by `llama-server`
- [nlohmann/json](https://github.com/nlohmann/json) — MIT — JSON handling
- [stb](https://github.com/nothings/stb) — public domain — image decoding for the multimodal path
- [miniaudio](https://github.com/mackron/miniaudio) — public domain — audio decoding
- [subprocess.h](https://github.com/sheredom/subprocess.h) — public domain — process launching

## Our work

What this fork adds on top of the above is the deployment layer and the machine-specific tuning:
the environment defaults baked into the binary at startup, the load-time trimming of the draft
model's output vocabulary, the reproducible release-package builder, and the launcher. Any bug in
those is ours, not the upstreams'.
