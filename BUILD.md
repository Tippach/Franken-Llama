# Build

Windows only, `gfx1151` (Radeon 8060S) only. This is not a general build guide for llama.cpp —
it documents how *this* fork is built, because the toolchain is pinned.

## Prerequisites

| Component | What we use | Notes |
| --- | --- | --- |
| Visual Studio | C++ desktop workload, MSVC toolset **14.51** | located with `vswhere`, so the install path is not fixed; override the toolset with `set VS_TOOLS_VER=...` |
| ROCm | **TheRock 10.1** (`10.1.0a20260910`), installed as Python packages | see below |
| CMake | 3.24+ | |
| Ninja | `toolchain\ninja.exe` or on `PATH` | |
| OpenSSL | OpenSSL for Windows (x64) | provides inbound/outbound TLS; override the location with `set OPENSSL_ROOT=...` |

### TheRock ROCm

The build uses the ROCm SDK from TheRock installed into a local virtualenv at
`toolchain\rocm-venv`. The exact set of packages the shipped build was compiled against:

```
rocm                    10.1.0a20260910
rocm-sdk-core           10.1.0a20260910
rocm-sdk-devel          10.1.0a20260910
rocm-sdk-device-gfx1151 10.1.0a20260910
rocm-sdk-libraries      10.1.0a20260910
```

Install them into a venv named `toolchain\rocm-venv` following the official TheRock
instructions (https://rocm.docs.amd.com/projects/TheRock/), which publish the package index to
use. `rocm-sdk-devel` is the one `_configure.cmd` points `HIP_PATH`/`ROCM_PATH` at;
`rocm-sdk-device-gfx1151` supplies the `gfx1151` code objects.

The layout `_configure.cmd` expects:

```
toolchain\
  ninja.exe
  rocm-venv\Lib\site-packages\_rocm_sdk_devel\   <- HIP_PATH / ROCM_PATH
```

## Configure and build

```
_configure.cmd
_build.cmd
```

Both derive every path from their own location, so they work from any clone directory. They set:

- `-DGGML_HIP=ON -DGPU_TARGETS=gfx1151`
- `-DBUILD_SHARED_LIBS=ON`, `CMAKE_BUILD_TYPE=Release`
- `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF`
- `-DLLAMA_OPENSSL=ON`
- Clang from the SDK as C/C++/HIP compiler, and `--rocm-device-lib-path` pointed at the SDK's
  AMDGPU bitcode (this flag matters: without it HIP compilation picks up the wrong device library)

`_build.cmd` builds the `ggml-hip`, `llama` and `llama-server` targets into `build\bin\`.

## Producing the release package

```
powershell -ExecutionPolicy Bypass -File package\build_package.ps1
```

One command: build, stage only the binaries the server actually imports (derived from the PE
import closure, so it cannot drift), verify, zip, re-verify from inside the archive, and smoke
test. See [package/README.md](package/README.md).

## Notes and gotchas

- **The runtime is self-contained by construction.** The HIP runtime pulls `amd_comgr` and
  `rocm_kpack` by name at init, and the `gfx1151` kernel data (`rocblas/`, `hipblaslt/`,
  `.kpack`) is opened by path — none of that is visible in a static import table, so the package
  builder stages them explicitly and a live process module list confirms nothing else is needed.
- **`hiprtc` is not shipped.** Every kernel here is precompiled for `gfx1151`, so the runtime JIT
  is never loaded. Confirmed both by import closure and by a live module dump.
- **`gfx1151` is not in older ROCm releases.** Building against anything other than TheRock 10.1
  line is untested here.
- The server bakes its internal tuning variables into its own process environment at startup, so
  no environment setup is required to build or run it.
