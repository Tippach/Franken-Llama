@echo off
REM build_gui.cmd -- compile strix_gui.cpp to strix_gui.exe with MSVC.
REM Run from anywhere; it locates vcvars64 via vswhere.
setlocal
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" ( echo vswhere not found & exit /b 1 )
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT ( echo no VS install with VC tools & exit /b 1 )
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

cd /d "%~dp0"
echo compiling strix_gui.cpp ...
cl /nologo /EHsc /W3 /O2 /MT strix_gui.cpp /Fe:strix_gui.exe /link /MANIFEST:EMBED
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
if exist strix_gui.obj del strix_gui.obj
if exist strix_gui.exe.manifest del strix_gui.exe.manifest
echo BUILD OK: %~dp0strix_gui.exe
endlocal
