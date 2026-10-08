@echo off
REM Configure Franken-Llama (HIP / gfx1151) against a local TheRock ROCm toolchain.
REM
REM Paths are derived from this file's own location, so nothing machine-specific is hardcoded.
REM Prerequisites (see BUILD.md):
REM   - Visual Studio with the C++ workload (located via vswhere; override toolset with VS_TOOLS_VER)
REM   - TheRock ROCm 10.1 python package installed into .\toolchain\rocm-venv
REM   - ninja.exe at .\toolchain\ninja.exe (or on PATH)
REM   - OpenSSL for Windows; override the location with OPENSSL_ROOT
setlocal EnableExtensions
set "ROOT=%~dp0"
set "TC=%ROOT%toolchain\rocm-venv\Lib\site-packages\_rocm_sdk_devel"

REM Pinned MSVC toolset: the shipped package was built with this exact version.
if not defined VS_TOOLS_VER set "VS_TOOLS_VER=14.51"
if not defined OPENSSL_ROOT set "OPENSSL_ROOT=C:/Program Files/OpenSSL-Win64"

if not exist "%TC%" (
  echo [ERROR] ROCm toolchain not found at: "%TC%"
  echo         See BUILD.md for installing TheRock ROCm into .\toolchain\rocm-venv
  exit /b 1
)

REM Locate the VS C++ environment with vswhere instead of a fixed install path.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [ERROR] vswhere.exe not found at "%VSWHERE%" - install the Visual Studio C++ workload.
  exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (
  echo [ERROR] vswhere found no Visual Studio with the VC++ tools component.
  exit /b 1
)
call "%VSROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%VS_TOOLS_VER% || exit /b 1

set "HIP_PATH=%TC%"
set "ROCM_PATH=%TC%"
set "HIP_DEVICE_LIB_PATH=%TC%\lib\llvm\amdgcn\bitcode"
set "PATH=%TC%\bin;%TC%\lib\llvm\bin;%ROOT%toolchain;%PATH%"

pushd "%ROOT%" || exit /b 1
cmake -S . -B build -G Ninja ^
  -DGGML_HIP=ON ^
  -DGPU_TARGETS=gfx1151 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_NATIVE=OFF ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_CURL=OFF ^
  -DCMAKE_C_COMPILER=%TC:/=\%\lib\llvm\bin\clang.exe ^
  -DCMAKE_CXX_COMPILER=%TC:/=\%\lib\llvm\bin\clang++.exe ^
  -DCMAKE_HIP_COMPILER=%TC:/=\%\lib\llvm\bin\clang++.exe ^
  -DCMAKE_MAKE_PROGRAM=%ROOT:/=\%\toolchain\ninja.exe ^
  -DCMAKE_HIP_FLAGS="--rocm-device-lib-path=%TC:/=\%\lib\llvm\amdgcn\bitcode" ^
  -DLLAMA_OPENSSL=ON ^
  -DOPENSSL_ROOT_DIR="%OPENSSL_ROOT%" ^
  -DSSL_EAY="%OPENSSL_ROOT%/lib/VC/x64/MD/libssl.lib" ^
  -DLIB_EAY="%OPENSSL_ROOT%/lib/VC/x64/MD/libcrypto.lib"
set RC=%ERRORLEVEL%
popd
echo CONFIGURE_EXIT=%RC%
exit /b %RC%
