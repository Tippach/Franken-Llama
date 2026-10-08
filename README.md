# Franken-Llama

A llama.cpp fork tuned to run **Qwen3.8-Flash-Next** on a single **AMD Ryzen AI Max+ 395**
(Radeon 8060S, `gfx1151`) under Windows, built with HIP against TheRock ROCm.

It is a downstream build of [llama.cpp](https://github.com/ggml-org/llama.cpp) carrying the
[Rulith Inference](https://github.com/rulith-dev/rulith-inference) patch set plus our own
kernels and deployment work for one specific machine. The goal is a server that fits a whole
256k-context conversation in unified memory on a Strix Halo box and serves it over an
OpenAI-compatible API.

See [ATTRIBUTION.md](ATTRIBUTION.md) for the full list of upstream projects this builds on —
this is assembled from several sources and each is credited there.

## What it does

- **Deep context in one box.** 262144-token context by default, and YaRN rope scaling to reach
  512k, on a machine configured with a 96 GB dedicated GPU carve plus 16 GB of shared memory.
- **HIP backend work for `gfx1151`.** Fused kernels for the model's hyper-connections, MoE
  routing, gated-delta-net and sparse-attention (QSA) paths, plus the environment defaults baked
  into the binary so a launch is reproducible.
- **Disk-backed KV cache.** KV state is streamed to disk keyed by a content hash-chain, so a
  shared prompt prefix is a cache hit even after a restart.
- **Speculative decode** with an MTP draft whose output head is trimmed at load to keep draft
  bandwidth down.

## Get the prebuilt package

The fastest path is the release archive — a self-contained Windows runtime, no build tools
needed (only the installed AMD graphics driver):

1. Download `franken-llama-0.1.0.zip` from the [latest release](../../releases).
2. Unzip anywhere.
3. Double-click `launcher\strix.bat`.

The server listens on `127.0.0.1:9931` with an OpenAI-compatible `/v1/chat/completions`
endpoint. Models and the MTP draft are downloaded from Hugging Face on first launch. The
archive's `README.md` documents every launcher knob (context size, YaRN, slots, HTTPS, vision).

## Build from source

Requires Visual Studio C++ workload, TheRock ROCm 10.1, CMake, Ninja and OpenSSL — see
[BUILD.md](BUILD.md). In short:

```
python -m venv toolchain\rocm-venv
toolchain\rocm-venv\Scripts\pip install therock[all]      # see BUILD.md for exact pins
_configure.cmd
_build.cmd
```

The build produces `build\bin\` with the server and its DLLs. To assemble the deployable
package:

```
powershell -ExecutionPolicy Bypass -File package\build_package.ps1
```

That single script builds, stages only the binaries the server actually imports, verifies the
result, and writes the zip. See [package/README.md](package/README.md).

## Layout

| Path | What it is |
| --- | --- |
| `ggml/`, `src/`, `common/`, `include/` | the llama.cpp / ggml source, with our HIP changes |
| `tools/server/` | the HTTP server, including the disk KV-cache store |
| `package/` | the release-package builder and its launcher/template prerequisites |
| `_configure.cmd`, `_build.cmd` | one-shot build against the local TheRock toolchain |
| `docs/` | llama.cpp build and backend documentation |

## License

MIT, as llama.cpp is. Third-party components and their licenses are listed in
[ATTRIBUTION.md](ATTRIBUTION.md) and under `licenses/`.

## Not this project

Franken-Llama is not a general llama.cpp fork and does not track upstream releases. It targets
one GPU (`gfx1151`) on Windows with HIP; other backends are present because they come with
llama.cpp but are not exercised here.
