@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ============================================================================
REM  FRANKEN-LLAMA SERVER - launcher for AMD Ryzen AI Max+ 395 (gfx1151)
REM ----------------------------------------------------------------------------
REM  MODEL and MTP each take either a local file or a Hugging Face checkpoint;
REM  the launcher only picks the flag llama-server already understands:
REM      value ends in .gguf  ->  -m FILE       /  -md FILE
REM      anything else        ->  -hf REPO:QTY  /  -hfd REPO:QTY
REM  So MODEL can be "user/repo:quant" or "C:\path\model.gguf".
REM
REM  The runtime (llama-server.exe + all DLLs) lives in the sibling ..\runtime
REM  folder and is fully self-contained.
REM  Requires: AMD Radeon 8060S (gfx1151) with its installed graphics driver.
REM  Recommended: paging file 128000 / 128000 MB (deep-context commit headroom).
REM ============================================================================

set "HERE=%~dp0"
set "RUNTIME=%HERE%..\runtime"

if not exist "%RUNTIME%\llama-server.exe" (
  echo [ERROR] runtime\llama-server.exe not found at: "%RUNTIME%"
  echo         Keep this .bat inside the launcher\ folder of the package.
  pause
  exit /b 1
)

REM --- all native DLLs resolve from the runtime folder first ---
set "PATH=%RUNTIME%;%PATH%"

REM ============================================================================
REM  QUICK ACCESS - the four knobs most people actually change. Edit these here.
REM  Each keeps an already-set environment value (a shell can still override),
REM  and the details live in the matching sections further down.
REM    VISION=1     image input: loads the ~865 MB vision projector. Also selects
REM                 kv-chain as the prompt cache on disk - the only one that stores
REM                 conversations containing images (agentic screenshot loops).
REM    PARALLEL=2   two conversations at once; auto-switches the cache to --kv-unified. With two or
REM                 more generating at the same time the server also adapts the draft length to what
REM                 is actually being accepted (STRIX_SPEC_ADAPT, on by default) - nothing to set.
REM    YARN=1       rope scaling so CTX can exceed the trained 262144. Pair with
REM                 CTX=524288 for ~512k context. Leave YARN=0 at or below 262144.
REM    CTX=524288   context size in tokens; go above 262144 only together with YARN=1.
REM    KVQ8=1       quantize the KV cache - and the MTP draft's - to q8_0: adds
REM                 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0. Default on; KVQ8=0 keeps
REM                 the cache unquantized.
REM  The defaults below are the validated shipping config: two 512k conversations with
REM  YaRN and the KV cache at q8_0. A shell can still override any of them.
REM ============================================================================
if not defined VISION set "VISION=1"
if not defined PARALLEL set "PARALLEL=2"
if not defined YARN set "YARN=1"
if not defined CTX set "CTX=524288"
if not defined KVQ8 set "KVQ8=1"

REM ============================================================================
REM  EDIT ME - the end-user knobs. An already-set environment value wins over
REM  each 'if not defined' line, so a shell can override without editing this file.
REM ============================================================================

REM Model: an HF checkpoint (resolved by the server) or a local .gguf.
if not defined MODEL set "MODEL=unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS"

REM MTP draft. Default: pulled from Hugging Face on first launch, cached afterwards.
REM   MTP=<local.gguf>   a local draft file
REM   MTP=<repo:quant>   a different HF draft repo (set MTP_FILE too if it is not at repo root)
REM   MTP=none           no speculative decoding
if not defined MTP (
  set "MTP=unsloth/Qwen3.8-Flash-Next-GGUF"
  set "MTP_FILE=MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"
)
if /I "!MTP!"=="none" set "MTP="

REM Chat template shipped with the package. Empty = the model's embedded template.
if not defined CTPL set "CTPL=%HERE%chat_template.jinja"

REM --- context size ---------------------------------------------------------------
REM The model is trained to 262144 tokens. CTX above that needs YaRN (see below) or
REM quality degrades past the trained range. Default set in QUICK ACCESS above.

REM --- YaRN rope scaling (opt-in; default OFF) ------------------------------------
REM YaRN lets the model address a context longer than it was trained on. YARN=1 adds
REM --rope-scaling yarn --rope-scale <YARN_SCALE> and is meant to be paired with a CTX
REM above the trained 262144. YARN_SCALE is the expansion factor: the usable context is
REM roughly 262144 x YARN_SCALE.
REM   YARN=1 YARN_SCALE=2  with CTX=524288  -> 512k context. On a Ryzen AI Max+ 395 configured
REM     with a 96 GB dedicated carve plus 16 GB shared, this loads and runs using about 97 GB.
REM Leave YARN=0 for CTX up to 262144 (no rope scaling needed in the trained range).
REM Default set in QUICK ACCESS above (shipping config is YARN=1).
if not defined YARN_SCALE set "YARN_SCALE=2"

if not defined PORT set "PORT=9931"
if not defined HOSTIP set "HOSTIP=127.0.0.1"

REM --- Latin draft-vocab trim (default ON) ----------------------------------------
REM The MTP draft borrows the target's full-vocabulary output head, which is streamed on
REM every draft step. LATIN=1 trims it at load to a 32768-token Latin shortlist: the same
REM output (the target still verifies over the full vocabulary) with far less bandwidth.
REM Tuned for Latin/English. For Chinese or other CJK set LATIN=0.
if not defined LATIN set "LATIN=1"

REM --- prompt cache on disk ---------------------------------------------------------
REM Exactly ONE mechanism runs (they are mutually exclusive - the server refuses both).
REM Default CACHE=kvchain: the hash-chain KV cache handles text AND image conversations, so
REM it is the safe choice for every config - it is the only one that stores a conversation
REM containing images, and it caches the text between and after screenshots (an agentic
REM screenshot loop keeps its prefix). Other values:
REM   CACHE=disk  the rulith disk prompt cache under --cache-ram (text-only: image chats are
REM               not stored); rows written once per 4096 positions, checkpoints paged to disk.
REM   CACHE=auto  pick kvchain when VISION=1, disk otherwise (the pre-split behavior).
REM   CACHE=off   no prompt cache on disk, RAM/VRAM only.
REM Quotas: KVCHAIN_GB (kvchain) / PCCACHE_GB (disk) are GiB on disk.
if not defined KVCHAIN_GB set "KVCHAIN_GB=100"
if not defined PCCACHE_GB set "PCCACHE_GB=64"
REM back-compat: the old KVCHAIN=0/1 switch still works (this bat no longer defaults it,
REM so "defined" here means the user set it): 0 -> nothing on disk, 1 -> force kv-chain
if defined KVCHAIN if not defined CACHE (
  if "!KVCHAIN!"=="0" set "CACHE=off"
  if "!KVCHAIN!"=="1" set "CACHE=kvchain"
)
if not defined CACHE set "CACHE=kvchain"

REM --- concurrent conversations (slots) -------------------------------------------
REM PARALLEL=1 runs one conversation at a time. PARALLEL>1 runs several at once and
REM switches the cache to a single unified stream automatically. Default set in QUICK ACCESS.
if "%PARALLEL%"=="1" (
  set "KV_MODE=--no-kv-unified"
) else (
  set "KV_MODE=--kv-unified"
)

REM --- vision / image input (default ON; set VISION=0 for text only) -------------
REM   VISION=1            auto-resolve the model repo's vision projector (adds ~865 MB
REM                       download and offloads it in addition to the context)
REM   MMPROJ=<file.gguf>  explicit local projector; implies VISION=1
if not defined VISION set "VISION=1"
if defined MMPROJ set "VISION=1"

REM --- HTTPS / TLS (opt-in; default OFF = plain HTTP on loopback) ------------------
REM SSL_KEY / SSL_CERT are PEM paths; if unset the launcher looks for key.pem / cert.pem
REM in the launcher folder. Generate a throwaway self-signed pair with:
REM   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes ^
REM     -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
if not defined SSL set "SSL=0"
if not defined SSL_KEY set "SSL_KEY=%HERE%key.pem"
if not defined SSL_CERT set "SSL_CERT=%HERE%cert.pem"

REM ============================================================================
REM  Derive the flags from the knobs above
REM ============================================================================

REM local file -> -m ; anything else -> -hf. Substring expansion, not 'echo | findstr':
REM cmd feeds the space before the pipe into the echoed text, so an end-anchored match
REM would never see a trailing ".gguf".
set "MODEL_FLAG=-hf"
if /I "!MODEL:~-5!"==".gguf" set "MODEL_FLAG=-m"

REM draft: local .gguf -> -md ; HF repo -> -hfd, plus -md <in-repo file> when MTP_FILE is set
set "MTP_ARGS="
if defined MTP if /I "!MTP:~-5!"==".gguf" set "MTP_ARGS=-md "!MTP!""
if defined MTP if /I not "!MTP:~-5!"==".gguf" if defined MTP_FILE set "MTP_ARGS=-hfd "!MTP!" -md "!MTP_FILE!""
if defined MTP if /I not "!MTP:~-5!"==".gguf" if not defined MTP_FILE set "MTP_ARGS=-hfd "!MTP!""

REM projector
if "!VISION!"=="1" (set "MM_MODE=--mmproj-auto") else (set "MM_MODE=--no-mmproj")
if defined MMPROJ set "MM_MODE=-mm "!MMPROJ!""

REM prompt cache on disk: resolve auto now that VISION is final (MMPROJ= implies vision),
REM then derive the flag set. Exactly ONE mechanism - the server refuses to run with both.
REM The kv-chain path runs with the server's default RAM prompt cache (no --cache-ram 0):
REM the hash-chain store owns prompt state on disk and does not need the RAM cache disabled.
REM The disk tier needs a RAM prompt cache to be a tier under - 1 GiB keeps short
REM conversations resident and hands the rest to disk (rulith's own manager value); idle
REM slots stay in their slots (--no-cache-idle-slots).
if /I "!CACHE!"=="auto" (
  if "!VISION!"=="1" (set "CACHE=kvchain") else (set "CACHE=disk")
)
REM one shared directory for whichever mechanism runs: the two formats are unrelated, so a
REM flip of VISION/CACHE must not leave the other mechanism's bytes behind - the stamp file
REM below records who owns the dir and the whole dir is wiped when ownership changes.
if not defined CACHE_DIR set "CACHE_DIR=%HERE%prompt-cache"
if /I "!CACHE!"=="kvchain" (
  set "PC_ARGS=--kv-chain-dir "!CACHE_DIR!" --kv-chain-limit-gb %KVCHAIN_GB%"
)
if /I "!CACHE!"=="disk" (
  set /A "PC_MIB = PCCACHE_GB * 1024"
  set "PC_ARGS=--cache-ram 1024 --no-cache-idle-slots"
  set "PC_ARGS=!PC_ARGS! --cache-disk-dir "!CACHE_DIR!" --cache-disk-limit-mib !PC_MIB!"
)
if /I "!CACHE!"=="off" set "PC_ARGS=--cache-ram 0"
if /I not "!CACHE!"=="kvchain" if /I not "!CACHE!"=="disk" if /I not "!CACHE!"=="off" (
  echo [ERROR] CACHE must be auto, kvchain, disk, or off ^(got "!CACHE!"^).
  pause
  exit /b 1
)

REM ownership stamp: owner.txt inside the shared dir names the mechanism that last wrote it.
REM A flip (VISION switched, or CACHE forced) wipes the dir's CONTENTS first - the two on-disk
REM formats are unrelated and neither store evicts the other's files, so without this the dir
REM silently accumulates dead chunks/runs from the mechanism no longer running.
if /I not "!CACHE!"=="off" (
  REM migration: the old bat gave each mechanism its own folder (launcher\kv-chain); once the
  dir is shared, a leftover one is exactly the ancient clutter this section exists to prevent
  if /I not "!CACHE_DIR!"=="%HERE%kv-chain" if exist "%HERE%kv-chain" (
    echo  CACHE : removing old per-mechanism cache folder %HERE%kv-chain
    rd /s /q "%HERE%kv-chain"
  )
  if not exist "!CACHE_DIR!" mkdir "!CACHE_DIR!"
  set "PC_OWNER="
  if exist "!CACHE_DIR!\owner.txt" set /p PC_OWNER=<"!CACHE_DIR!\owner.txt"
  if /I not "!PC_OWNER!"=="!CACHE!" (
    REM a real flip: the dir held the other mechanism's bytes - clear them before restamping.
    REM on a first-ever run PC_OWNER is empty and there is nothing of the other arm to remove,
    REM so only stamp; del/rd are silenced so an empty dir never prints "File Not Found".
    if not "!PC_OWNER!"=="" (
      echo  CACHE : wiping !CACHE_DIR! ^(held !PC_OWNER! data, now !CACHE!^)
      del /q /f "!CACHE_DIR!\*.*" >nul 2>nul
      for /d %%D in ("!CACHE_DIR!\*") do rd /s /q "%%D" >nul 2>nul
    )
    >"!CACHE_DIR!\owner.txt" (echo !CACHE!)
  )
)

REM YaRN: scale the rope so CTX beyond the trained 262144 stays in distribution.
REM Appended to ARGS further down (not inline) so an inactive YARN=0 leaves no stray spaces.

REM ============================================================================
REM  Preflight - fail loudly on a config error rather than start something wrong
REM ============================================================================

if "!VISION!"=="1" if defined MMPROJ if not exist "!MMPROJ!" (
  echo [ERROR] MMPROJ was set but the file was not found: "!MMPROJ!"
  pause
  exit /b 1
)

if defined MTP if /I "!MTP:~-5!"==".gguf" if not exist "!MTP!" (
  echo [ERROR] MTP points at a local draft that was not found: "!MTP!"
  echo         Set MTP to an existing file, an HF repo, or MTP=none.
  pause
  exit /b 1
)

if "!SSL!"=="1" (
  if not exist "!SSL_KEY!" (
    echo [ERROR] SSL=1 but the key file was not found: "!SSL_KEY!"
    echo         Set SSL_KEY / SSL_CERT, or put key.pem and cert.pem in: "%HERE%"
    pause
    exit /b 1
  )
  if not exist "!SSL_CERT!" (
    echo [ERROR] SSL=1 but the certificate file was not found: "!SSL_CERT!"
    echo         Set SSL_KEY / SSL_CERT, or put key.pem and cert.pem in: "%HERE%"
    pause
    exit /b 1
  )
)

REM a stale server on this port would already be holding the VRAM
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo [ERROR] Port %PORT% is already in use. A llama-server may still be running.
  echo         Close it first, or set PORT to a free port and re-run.
  pause
  exit /b 1
)

REM YaRN is pointless at or below the trained context and only costs accuracy there
if "!YARN!"=="1" if %CTX% LEQ 262144 (
  echo [ERROR] YARN=1 but CTX=%CTX% is within the trained 262144 - rope scaling is not
  echo         needed. Set YARN=0, or raise CTX above 262144 ^(e.g. 524288 for scale 2^).
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  MODEL : !MODEL!  ^(flag !MODEL_FLAG!^)
if defined MTP ( echo  DRAFT : !MTP_ARGS! ) else ( echo  DRAFT : ^(none - MTP disabled^) )
if defined CTPL ( echo  TPL   : !CTPL! ) else ( echo  TPL   : ^(model-embedded^) )
if "!SSL!"=="1" (echo  SCHEME: https  cert=!SSL_CERT!) else (echo  SCHEME: http)
if "!VISION!"=="1" (echo  VISION: ON   !MM_MODE!) else (echo  VISION: off  ^(text only; set VISION=1 for image input^))
if /I "!CACHE!"=="kvchain" echo  CACHE : kv-chain  dir=!CACHE_DIR!  quota=%KVCHAIN_GB% GiB - the only tier that stores image conversations
if /I "!CACHE!"=="disk"    echo  CACHE : disk prompt cache  dir=!CACHE_DIR!  quota=%PCCACHE_GB% GiB - text-only, image chats are not cached
if /I "!CACHE!"=="off"     echo  CACHE : off - RAM/VRAM only
if "!YARN!"=="1" (echo  ROPE  : YaRN scale !YARN_SCALE! ^(orig ctx 262144^)) else (echo  ROPE  : native)
echo  CTX   : %CTX%   PORT: %HOSTIP%:%PORT%
echo  RUNTIME: %RUNTIME%
echo ============================================================
echo.

if "!KVQ8!"=="1" echo  KVQ8  : K/V + draft K/V quantized to q8_0 ^(-ctk -ctv -ctkd -ctvd^)
if "!LATIN!"=="1" if defined MTP echo  TRIM  : Latin draft vocab, trimmed at load ^(32768-token d2t^)
if not "!LATIN!"=="1" if defined MTP echo  TRIM  : disabled, full draft vocab ^(LATIN=!LATIN!^)
if defined MTP echo.

REM ============================================================================
REM  Build the command line
REM ============================================================================
set "ARGS=!MODEL_FLAG! "!MODEL!" -ngl 99 --ctx-size %CTX% -b 8192 -ub 8192 --flash-attn on -t 4 --parallel %PARALLEL% --port %PORT% --host %HOSTIP% --temp 1.0 --presence-penalty 0.0 --repeat-penalty 1.0 %PC_ARGS% %KV_MODE% %MM_MODE% --fit off --load-mode none --lazy-mode on-direct"

REM KV cache quantization: q8_0 for the target K/V and the MTP draft's K/V. Halves the KV
REM footprint at depth for negligible quality loss; KVQ8=0 leaves the cache unquantized.
if "!KVQ8!"=="1" set "ARGS=!ARGS! -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0"

REM rope scaling only when asked for; appended so the default line is untouched
if "!YARN!"=="1" set "ARGS=%ARGS% --rope-scaling yarn --rope-scale !YARN_SCALE! --yarn-orig-ctx 262144"

if defined CTPL set "ARGS=%ARGS% --jinja --chat-template-file "!CTPL!" --reasoning on --reasoning-format deepseek --reasoning-effort low --reasoning-preserve"

REM Speculative decode: the draft proposes up to 3 tokens, every one of them still
REM verified by the target, so the output distribution is unchanged - this is a pure
REM speed knob. p_min 0.30 truncates a draft chain once a token's confidence falls
REM below it; it changes draft LENGTH only, never correctness.
if defined MTP set "ARGS=%ARGS% !MTP_ARGS! --spec-type draft-mtp --spec-draft-ngl 99 --spec-draft-n-max 3 --spec-draft-p-min 0.30"
if defined MTP if "!LATIN!"=="1" set "ARGS=%ARGS% --spec-draft-vocab-latin"

if "!SSL!"=="1" set "ARGS=%ARGS% --ssl-key-file "!SSL_KEY!" --ssl-cert-file "!SSL_CERT!""



REM ============================================================================
REM  Launch
REM ============================================================================
"%RUNTIME%\llama-server.exe" %ARGS%

echo.
echo Server exited. Press any key...
pause >nul
