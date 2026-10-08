@echo off
REM Build the Franken-Llama HIP targets. Paths derive from this file's location.
REM Run _configure.cmd first. See BUILD.md for the toolchain prerequisites.
setlocal EnableExtensions
set "ROOT=%~dp0"
set "TC=%ROOT%toolchain\rocm-venv\Lib\site-packages\_rocm_sdk_devel"

if not defined VS_TOOLS_VER set "VS_TOOLS_VER=14.51"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
  echo [ERROR] vswhere found no Visual Studio with the VC++ tools component.
  exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VS_TOOLS_VER% >nul 2>&1 || exit /b 1

set "HIP_PATH=%TC%"
set "ROCM_PATH=%TC%"
set "HIP_DEVICE_LIB_PATH=%TC%\lib\llvm\amdgcn\bitcode"
set "PATH=%TC%\bin;%TC%\lib\llvm\bin;%ROOT%toolchain;%PATH%"

if exist "%ROOT%toolchain\ninja.exe" (
  "%ROOT%toolchain\ninja.exe" -C "%ROOT%build" ggml-hip llama llama-server
) else (
  ninja -C "%ROOT%build" ggml-hip llama llama-server
)
set RC=%ERRORLEVEL%
echo BUILD_EXIT=%RC%
exit /b %RC%
