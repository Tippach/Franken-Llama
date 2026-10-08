# Franken-Llama deployment package builder

Single command builds the deployable HIP server package for gfx1151:

```
powershell -NoProfile -ExecutionPolicy Bypass -File package\build_package.ps1
```

Output (inside the clone, git-ignored): `dist\franken-llama\` (staged tree) and
`dist\franken-llama.zip` (~105 MB).

## What it does (6 phases, all-or-nothing)
1. **build** — runs `..\_build.cmd` (skip with `-SkipBuild`).
2. **stage** — copies the **import closure** of `llama-server.exe` (computed by `closure.py` against
   `build\bin`) plus the gfx1151 kernel data dirs (`rocblas/`, `hipblaslt/`, `.kpack/`) and the
   `content\` prerequisites. The binary set is *derived*, never a hardcoded list, so it cannot drift
   when the build gains or drops a dependency.
3. **verify-tree** — closure self-contained (0 unresolved, 0 unreachable modules), every staged DLL
   byte-matches `build\bin`, launcher/README guards, no bundled draft, no key material.
4. **zip** — compresses the staged tree.
5. **verify-zip** — reopens the archive and byte-checks the shipped binaries + capability guards
   (load-time trim, d2t arg, SSL link). Never trusts the file copy.
6. **smoke** — `llama-server.exe --help` loads with no `0xC0000135` (DLL-load only, no model/VRAM).

Switches: `-SkipBuild`, `-NoZip`, `-Out <dir>`.

## What ships and why it is small
Only the server. No tests, tools, benchmarks, and **no hiprtc** (the runtime JIT is never loaded —
every kernel is precompiled for gfx1151). The **MTP draft is not bundled**: it is resolved from
Hugging Face at run time and its borrowed LM head is trimmed at load (`LATIN=1`), which is what keeps
the zip ~105 MB instead of ~1.9 GB. See the package `README.md`.

## Source of truth
`content\` holds the editable prerequisites the builder copies verbatim:
- `content\launcher\strix.bat` — the launcher (tracked despite the repo-wide `*.bat` ignore via a
  negation in the top-level `.gitignore`; same precedent as `!/examples/*.bat`).
- `content\launcher\chat_template.jinja`
- `content\README.md`

Edit these, then re-run the builder. Do **not** edit the copy under `dist\` — it is
overwritten on every run.

## `closure.py`
Standalone PE import-closure tool. `--dir <runtime> --entry llama-server.exe [--list|--json]`.
Reports reachable/unresolved/unreachable. Caveat it documents: static closure cannot see
`LoadLibrary`-by-name (how `amdhip64` pulls in `amd_comgr`/`rocm_kpack`) or path-opened data dirs,
so those are handled explicitly by the builder and confirmed by a live process module list.
