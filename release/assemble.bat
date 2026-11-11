@echo off
rem ============================================================================
rem  release\assemble.bat - 把预编译包装配出来（二进制不进 git，走 GitHub Releases）
rem
rem  在一台已装 oneAPI + 已编译过 SYCL 版的机器上运行本脚本，它会：
rem    1) 从 build\sycl\ 拷 nova_sycl.exe
rem    2) 从 oneAPI 运行时目录拷 sycl9 / libmmd / svml_dispmd / ur_*
rem    3) 从 System32 拷 MSVC 运行时
rem  然后整个 release\ 目录就是可分发的绿色包（约 30 MB）。
rem
rem  用法:  release\assemble.bat [exe路径]
rem ============================================================================
setlocal
cd /d "%~dp0"

set "EXE=%~1"
if "%EXE%"=="" set "EXE=..\build\sycl\sycl_e2e.exe"
if not exist "%EXE%" goto noexe

echo [1/3] exe
copy /y "%EXE%" "nova_sycl.exe" >nul
echo       <- %EXE%

echo [2/3] Intel oneAPI 运行时
set "CB=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin"
for %%d in (sycl9.dll libmmd.dll svml_dispmd.dll) do (
    if exist "%CB%\%%d" (copy /y "%CB%\%%d" . >nul && echo       %%d) else (echo       MISSING %%d)
)
for %%d in (ur_win_proxy_loader.dll ur_loader.dll) do (
    for /f "delims=" %%p in ('where /r "C:\Program Files (x86)\Intel\oneAPI" %%d 2^>nul') do (
        copy /y "%%p" . >nul && echo       %%d
        goto :next_%%d
    )
    echo       MISSING %%d
    :next_%%d
)

echo [3/3] MSVC 运行时
for %%d in (msvcp140.dll msvcp140_1.dll vcruntime140.dll) do (
    if exist "%SystemRoot%\System32\%%d" (copy /y "%SystemRoot%\System32\%%d" . >nul && echo       %%d)
)

echo.
echo 完成。release\ 现在可以直接打包发出去：
dir /b nova_sycl.exe *.dll 2>nul
echo.
echo 提示：目标机器还需要 **Intel Arc 显卡驱动**；装了 oneAPI 的机器会自动用它。
exit /b 0

:noexe
echo 找不到 exe: %EXE%
echo 先编译:  build_sycl_e2e.bat 1.5b
exit /b 1
