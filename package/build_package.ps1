<#
.SYNOPSIS
  Build the Franken-Llama HIP server package end to end: build -> stage -> verify -> zip -> re-verify.
.DESCRIPTION
  Single entry point for the deployable package. Nothing here is hand-maintained except the content
  prerequisites under .\content (launcher bat, chat template, README) - the binary set is DERIVED
  from the live import closure of build\bin, so it cannot silently drift when the build gains or
  drops a dependency.

  Stages ONLY what the server needs: the transitive static-import closure of llama-server.exe (plus
  the HIP runtime modules loaded by name and the gfx1151 kernel data dirs). No tests, tools,
  benchmarks, or the hiprtc JIT. The MTP draft is NOT bundled (resolved from Hugging Face at run
  time and trimmed at load), which is what keeps the zip ~105 MB instead of ~1.9 GB.

  Phases (each fails loudly, nothing partial ships):
    1. build        - run ..\_build.cmd (skip with -SkipBuild)
    2. stage        - closure-derived runtime + content into an out dir
    3. verify-tree  - closure self-contained, runtime-sync vs build\bin, launcher/README guards,
                      no bundled draft, no key material
    4. zip          - compress the staged tree
    5. verify-zip   - reopen the archive and byte-check the shipped binaries + guards
    6. smoke        - llama-server.exe --help loads with no 0xC0000135 (DLL-load only, no model)
.LINK
  closure.py  - the import-closure tool used by phases 2/3.
.PARAMETER Out
  Where the staged tree + zip land. Default: <repo>\dist\franken-llama (inside the clone, git-ignored).
.PARAMETER SkipBuild
  Reuse the existing build\bin instead of rebuilding.
.PARAMETER NoZip
  Stage + verify only; do not produce the zip (fast iteration).
#>
[CmdletBinding()]
param(
    [string]$Out,
    [switch]$SkipBuild,
    [switch]$NoZip
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$here     = Split-Path -Parent $MyInvocation.MyCommand.Path   # ...\<repo>\package
$repoRoot = Split-Path -Parent $here                          # ...\<repo>
$buildBin = Join-Path $repoRoot 'build\bin'
$buildKpack = Join-Path $repoRoot 'build\.kpack'
$content  = Join-Path $here 'content'
$closurePy = Join-Path $here 'closure.py'
# Default output stays INSIDE the clone so a fresh checkout never writes outside itself.
if (-not $Out) { $Out = Join-Path $repoRoot 'dist\franken-llama' }

$python = (Get-Command python -EA SilentlyContinue).Source
if (-not $python) { throw 'python not on PATH - needed to run closure.py' }

function Write-Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Sha256Bytes([byte[]]$b) {
    (([Security.Cryptography.SHA256]::Create().ComputeHash($b) | ForEach-Object { $_.ToString('x2') }) -join '')
}
function Sha256([string]$p) { Sha256Bytes ([IO.File]::ReadAllBytes($p)) }
function Fail($m) { throw "PACKAGE BUILD FAILED: $m" }

# ---- data directories shipped alongside the closure-derived DLLs ----------------------------
# rocblas/ + hipblaslt/ come from build\bin (the build copies the Tensile data there); .kpack is
# emitted to build\.kpack. All are already filtered to gfx1151, so they are copied whole.
$dataDirs = @(
    @{ rel = 'rocblas';  src = (Join-Path $buildBin 'rocblas') },
    @{ rel = 'hipblaslt'; src = (Join-Path $buildBin 'hipblaslt') },
    @{ rel = '.kpack';   src = $buildKpack }
)

# ============================================================================
# 1. BUILD
# ============================================================================
Write-Step 'phase 1/6 build'
if ($SkipBuild) {
    Write-Host 'SKIP (-SkipBuild); using existing build\bin'
} else {
    $buildCmd = Join-Path $repoRoot '_build.cmd'
    if (-not (Test-Path $buildCmd)) { Fail "build script not found: $buildCmd" }
    Write-Host "running $buildCmd"
    & cmd.exe /c "`"$buildCmd`""
    if ($LASTEXITCODE -ne 0) { Fail "_build.cmd exit $LASTEXITCODE" }
}
foreach ($need in 'llama-server.exe', 'ggml-hip.dll', 'llama.dll') {
    if (-not (Test-Path (Join-Path $buildBin $need))) { Fail "build\bin lacks $need - build did not produce it" }
}

# ============================================================================
# 2. STAGE  (closure-derived; no hardcoded DLL list)
# ============================================================================
Write-Step 'phase 2/6 stage'
$runtime = Join-Path $Out 'runtime'
$launch  = Join-Path $Out 'launcher'

# Ask the closure what llama-server.exe actually imports, resolved against build\bin.
$stageDlls = & $python $closurePy --dir $buildBin --entry llama-server.exe --list
if ($LASTEXITCODE -ne 0 -or -not $stageDlls) { Fail 'closure.py produced no reachable set from build\bin' }
$stageDlls = @($stageDlls | Where-Object { $_ })
Write-Host ("closure-derived root modules: {0}" -f $stageDlls.Count)

# Re-run closure against the STAGED set later; here just copy the reachable roots + data dirs.
if (Test-Path $runtime) { Remove-Item $runtime -Recurse -Force }
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
foreach ($f in $stageDlls) {
    Copy-Item -LiteralPath (Join-Path $buildBin $f) -Destination (Join-Path $runtime $f) -Force
}
foreach ($d in $dataDirs) {
    if (-not (Test-Path $d.src)) { Fail "data dir missing in build output: $($d.src)" }
    Copy-Item -LiteralPath $d.src -Destination (Join-Path $runtime $d.rel) -Recurse -Force
}
# content prerequisites (source-controlled)
if (Test-Path $launch) { Remove-Item $launch -Recurse -Force }
New-Item -ItemType Directory -Force -Path $launch | Out-Null
foreach ($cf in 'launcher\strix.bat', 'launcher\chat_template.jinja') {
    $src = Join-Path $content $cf
    if (-not (Test-Path $src)) { Fail "content prerequisite missing: $src" }
    Copy-Item -LiteralPath $src -Destination (Join-Path $launch (Split-Path -Leaf $cf)) -Force
}
$rdSrc = Join-Path $content 'README.md'
if (-not (Test-Path $rdSrc)) { Fail "content prerequisite missing: $rdSrc" }
Copy-Item -LiteralPath $rdSrc -Destination (Join-Path $Out 'README.md') -Force
# a package must never ship a bundled draft; if one lingers in Out from an older layout, drop it.
if (Test-Path (Join-Path $Out 'draft')) { Remove-Item (Join-Path $Out 'draft') -Recurse -Force; Write-Host 'removed stale draft\ from Out' }
Write-Host ("staged runtime files: {0}" -f (Get-ChildItem $runtime -Recurse -File).Count)

# ============================================================================
# 3. VERIFY TREE
# ============================================================================
Write-Step 'phase 3/6 verify-tree'
# 3a. closure of the staged runtime must be self-contained (no unresolved, no unreachable DLLs)
$closureJson = & $python $closurePy --dir $runtime --entry llama-server.exe --json | Out-String
$cl = $closureJson | ConvertFrom-Json
if ($cl.unresolved.Count -gt 0) { Fail ("staged runtime not self-contained, unresolved: " + ($cl.unresolved -join ', ')) }
$unreachDll = @($cl.unreachable | Where-Object { $_ -match '\.dll$|\.exe$' })
if ($unreachDll.Count -gt 0) { Fail ("staged runtime has unreachable modules (bloat): " + ($unreachDll -join ', ')) }
Write-Host ("closure: {0} reachable, 0 unresolved, 0 unreachable modules - self-contained" -f $cl.reachable_root.Count)

# 3b. runtime-sync: every root DLL must byte-match build\bin (catches a stale partial copy)
$stale = @()
foreach ($n in (Get-ChildItem $runtime -File | ForEach-Object { $_.Name })) {
    $src = Join-Path $buildBin $n
    if (-not (Test-Path $src)) { continue }
    if ((Sha256 (Join-Path $runtime $n)) -ne (Sha256 $src)) { $stale += $n }
}
if ($stale.Count) { Fail ("staged runtime STALE vs build\bin: " + ($stale -join ', ')) }
Write-Host 'runtime-sync: every staged root DLL byte-matches build\bin'

# 3c. launcher guards: the shipping intent must be intact
$bat = Get-Content (Join-Path $launch 'strix.bat') -Raw
if ($bat -notmatch 'spec-draft-p-min 0\.30') { Fail 'bat p_min not 0.30' }
if ($bat -notmatch 'spec-draft-n-max 3')     { Fail 'bat dmax not 3' }
if ($bat -notmatch '(?m)^\s*if not defined LATIN set "LATIN=1"') { Fail 'bat does not default LATIN=1' }
if ($bat -notmatch 'set "MTP=unsloth/Qwen3\.8-Flash-Next-GGUF"') { Fail 'bat does not default the draft to HF' }
if ($bat -notmatch 'MTP_FILE=MTP/mtp-Qwen3\.8-Flash-Next-shared-Q4_K_M\.gguf') { Fail 'bat does not name the shared draft file' }
if ($bat -notmatch '--spec-draft-vocab-latin') { Fail 'bat does not enforce the trim' }
if ($bat -match '\\draft\\') { Fail 'bat references a bundled draft\ path' }
# KV chain is ON by default now (prefix hits survive a restart on this UMA box)
if ($bat -notmatch '(?m)^\s*if not defined KVCHAIN set "KVCHAIN=1"') { Fail 'bat does not default KVCHAIN=1' }
# YaRN must be a real knob (default off, scale 2) and must actually emit the rope flags
if ($bat -notmatch '(?m)^\s*if not defined YARN set "YARN=0"') { Fail 'bat has no YARN knob defaulting to 0' }
if ($bat -notmatch '(?m)^\s*if not defined YARN_SCALE set "YARN_SCALE=2"') { Fail 'bat has no YARN_SCALE defaulting to 2' }
if ($bat -notmatch '--rope-scaling yarn --rope-scale !YARN_SCALE! --yarn-orig-ctx 262144') { Fail 'bat never emits the YaRN rope flags' }
# CTX must still be driven by the CTX var (a hardcoded -c would make the knob dead)
if ($bat -notmatch '--ctx-size %CTX%') { Fail 'bat does not pass --ctx-size %CTX%' }
$activeIS = @(($bat -split "`n") | Where-Object { $_ -match '^\s*set\s+LLAMA_MTP_INDEX_SHARE' })
if ($activeIS.Count) { Fail 'bat actively sets LLAMA_MTP_INDEX_SHARE' }
Write-Host 'launcher guards: PASS'

# 3d. no bundled draft dir; no key material; README present + documents HTTPS + HF draft
if (Test-Path (Join-Path $Out 'draft')) { Fail 'Out still contains a draft\ directory' }
# KVCHAIN defaults ON, so the server writes launcher\kv-chain\ the moment it is run - including by
# a validation load. Staging rebuilds launcher\ from content\, so that state cannot reach the zip;
# the only way it ships is if it was committed into content\, which this catches.
if (Test-Path (Join-Path $content 'launcher\kv-chain')) { Fail 'content\ ships kv-chain state - remove it from source control' }
$pemLeak = Get-ChildItem $Out -Recurse -File -Include '*.pem', '*.key' -EA SilentlyContinue
if ($pemLeak) { Fail ('key material in tree: ' + (($pemLeak | ForEach-Object Name) -join ', ')) }
$rd = Get-Content (Join-Path $Out 'README.md') -Raw
if ($rd -notmatch 'HTTPS / TLS') { Fail 'README has no HTTPS section' }
if ($rd -notmatch 'Speculative decode') { Fail 'README has no Speculative decode section' }
if ($rd -notmatch 'Context size and YaRN') { Fail 'README does not document the YaRN / deep-context knob' }
if ($rd -match '1\.86 GB|bundled \(1\.86') { Fail 'README still claims a bundled draft' }
Write-Host 'draft/key/README guards: PASS'

if ($NoZip) { Write-Host "`n(-NoZip) staged + verified at $Out"; exit 0 }

# ============================================================================
# 4. ZIP
# ============================================================================
Write-Step 'phase 4/6 zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path (Split-Path -Parent $Out) ((Split-Path -Leaf $Out) + '.zip')
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
[IO.Compression.ZipFile]::CreateFromDirectory($Out, $zipPath, [IO.Compression.CompressionLevel]::Optimal, $false)
Write-Host ("zip: {0}  ({1:N1} MB)" -f $zipPath, ((Get-Item $zipPath).Length / 1MB))

# ============================================================================
# 5. VERIFY ZIP (reopen; never trust the file copy)
# ============================================================================
Write-Step 'phase 5/6 verify-zip'
$z = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    Write-Host ("entries: {0}" -f $z.Entries.Count)
    $leak = @($z.Entries | Where-Object { $_.FullName -match 'rollback|\.bak$|scratch|promoted|\.(pem|key)$|^draft[\\/]|kv-chain' })
    if ($leak.Count) { Fail ('leak entries in zip: ' + (($leak | ForEach-Object FullName) -join ', ')) }
    Write-Host 'zip leak guard: PASS (no rollback/key/bundled-draft/kv-chain entries)'

    function Read-Entry($name) {
        $e = $z.Entries | Where-Object { $_.FullName -replace '\\','/' -eq $name }
        if (-not $e) { return $null }
        $ms = New-Object IO.MemoryStream; $st = $e.Open(); $st.CopyTo($ms); $st.Dispose()
        $b = $ms.ToArray(); $ms.Dispose(); return $b
    }
    # embedded binaries must byte-match the staged tree (proves the copy shipped, not a stale one)
    foreach ($chk in 'runtime/ggml-hip.dll', 'runtime/llama.dll', 'runtime/llama-common.dll', 'runtime/llama-server-impl.dll') {
        $zb = Read-Entry $chk
        if (-not $zb) { Fail "zip missing $chk" }
        $treePath = Join-Path $Out ($chk -replace '/', '\')
        if ((Sha256Bytes $zb) -ne (Sha256 $treePath)) { Fail "zipped $chk != tree" }
    }
    Write-Host 'zip binary match: ggml-hip, llama, llama-common, server-impl all == tree'

    # the load-time trim + d2t must actually be in the shipped binaries
    $llA = [Text.Encoding]::ASCII.GetString((Read-Entry 'runtime/llama.dll'))
    if ($llA -notmatch 'draft LM head trimmed at load') { Fail 'zipped llama.dll has no load-time trim' }
    $comA = [Text.Encoding]::ASCII.GetString((Read-Entry 'runtime/llama-common.dll'))
    if ($comA -notmatch '--spec-draft-vocab-latin') { Fail 'zipped llama-common lacks the trim arg' }
    $implA = [Text.Encoding]::ASCII.GetString((Read-Entry 'runtime/llama-server-impl.dll'))
    if ($implA -notmatch 'libssl-4-x64\.dll') { Fail 'zipped impl is not the SSL build' }
    Write-Host 'zip capability guards: trim + d2t arg + SSL link all present'
} finally { $z.Dispose() }

# ============================================================================
# 6. SMOKE (DLL-load only; no model, no VRAM)
# ============================================================================
Write-Step 'phase 6/6 smoke'
$exe = Join-Path $runtime 'llama-server.exe'
$prev = $env:PATH
try {
    $env:PATH = "$runtime;$env:PATH"
    $help = & $exe --help 2>&1 | Out-String
    # --help prints usage and exits non-zero by design; the ONLY thing that matters here is that
    # the binary LOADED (a missing DLL is the loader's 0xC0000135, before main runs). Do not gate
    # on the exit code -- it is not a load-failure signal.
    if ($LASTEXITCODE -eq -1073741515) { Fail 'llama-server.exe --help -> 0xC0000135 (missing DLL)' }
    $lines = ($help -split "`n" | Where-Object { $_.Trim() }).Count
    if ($lines -lt 100) { Fail "llama-server.exe --help produced only $lines lines (expected ~740)" }
    Write-Host ("smoke: --help loaded, {0} lines, no 0xC0000135" -f $lines)
} finally { $env:PATH = $prev }

Write-Host "`nPACKAGE BUILD OK" -ForegroundColor Green
Write-Host ("  tree: {0}" -f $Out)
Write-Host ("  zip : {0}  ({1:N1} MB)" -f $zipPath, ((Get-Item $zipPath).Length / 1MB))
# Exit with OUR status, not whatever the smoke binary left in $LASTEXITCODE.
exit 0
