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
REM quality degrades past the trained range.
if not defined CTX set "CTX=262144"

REM --- YaRN rope scaling (opt-in; default OFF) ------------------------------------
REM YaRN lets the model address a context longer than it was trained on. YARN=1 adds
REM --rope-scaling yarn --rope-scale <YARN_SCALE> and is meant to be paired with a CTX
REM above the trained 262144. YARN_SCALE is the expansion factor: the usable context is
REM roughly 262144 x YARN_SCALE.
REM   YARN=1 YARN_SCALE=2  with CTX=524288  -> 512k context. On a Ryzen AI Max+ 395 configured
REM     with a 96 GB dedicated carve plus 16 GB shared, this loads and runs using about 97 GB.
REM Leave YARN=0 for CTX up to 262144 (no rope scaling needed in the trained range).
if not defined YARN set "YARN=0"
if not defined YARN_SCALE set "YARN_SCALE=2"

if not defined PORT set "PORT=9931"
if not defined HOSTIP set "HOSTIP=127.0.0.1"

REM --- Latin draft-vocab trim (default ON) ----------------------------------------
REM The MTP draft borrows the target's full-vocabulary output head, which is streamed on
REM every draft step. LATIN=1 trims it at load to a 32768-token Latin shortlist: the same
REM output (the target still verifies over the full vocabulary) with far less bandwidth.
REM Tuned for Latin/English. For Chinese or other CJK set LATIN=0.
if not defined LATIN set "LATIN=1"

REM --- disk-backed KV cache (default ON) ------------------------------------------
REM Streams KV state to disk so prefix cache hits survive a restart and are not lost to
REM RAM/VRAM pressure. KVCHAIN_GB is the disk quota in GiB.
if not defined KVCHAIN set "KVCHAIN=1"
if not defined KVCHAIN_GB set "KVCHAIN_GB=100"

REM --- concurrent conversations (slots) -------------------------------------------
REM PARALLEL=1 runs one conversation at a time. PARALLEL>1 runs several at once and
REM switches the cache to a single unified stream automatically.
if not defined PARALLEL set "PARALLEL=1"
if "%PARALLEL%"=="1" (
  set "KV_MODE=--no-kv-unified"
) else (
  set "KV_MODE=--kv-unified"
)

REM --- vision / image input (opt-in; default OFF = text only) ---------------------
REM   VISION=1            auto-resolve the model repo's vision projector (adds ~865 MB
REM                       download and offloads it in addition to the context)
REM   MMPROJ=<file.gguf>  explicit local projector; implies VISION=1
if not defined VISION set "VISION=0"
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
if "!YARN!"=="1" (echo  ROPE  : YaRN scale !YARN_SCALE! ^(orig ctx 262144^)) else (echo  ROPE  : native)
echo  CTX   : %CTX%   PORT: %HOSTIP%:%PORT%
echo  RUNTIME: %RUNTIME%
echo ============================================================
echo.

if "!LATIN!"=="1" if defined MTP echo  TRIM  : Latin draft vocab, trimmed at load ^(32768-token d2t^)
if not "!LATIN!"=="1" if defined MTP echo  TRIM  : disabled, full draft vocab ^(LATIN=!LATIN!^)
if defined MTP echo.

REM ============================================================================
REM  Build the command line
REM ============================================================================
set "ARGS=!MODEL_FLAG! "!MODEL!" -ngl 99 --ctx-size %CTX% -b 8192 -ub 8192 --flash-attn on -t 4 --parallel %PARALLEL% --port %PORT% --host %HOSTIP% --temp 1.0 --presence-penalty 0.0 --repeat-penalty 1.0 --cache-ram 0 %KV_MODE% %MM_MODE% --fit off --load-mode none --lazy-mode on-direct"

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

if "!KVCHAIN!"=="1" (
  if not exist "%HERE%kv-chain" mkdir "%HERE%kv-chain"
  set "ARGS=%ARGS% --kv-chain-dir "!HERE!kv-chain" --kv-chain-limit-gb %KVCHAIN_GB%"
  echo  KVCHAIN: ON  dir=%HERE%kv-chain  quota=%KVCHAIN_GB% GiB
) else (
  echo  KVCHAIN: OFF ^(RAM/VRAM cache only^)
)

REM ============================================================================
REM  Launch
REM ============================================================================
"%RUNTIME%\llama-server.exe" %ARGS%

echo.
echo Server exited. Press any key...
pause >nul
