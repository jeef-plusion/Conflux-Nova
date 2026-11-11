@echo off
rem ============================================================================
rem  build.bat - 统一构建入口（内部只调 CMake）
rem
rem  用法:
rem    build.bat                  CUDA + 7B（默认）
rem    build.bat sycl 1.5b        SYCL + 1.5B
rem    build.bat host             只编 CPU 垫片（无显卡也能过）
rem    build.bat cuda 7b clean    先删构建目录
rem
rem  SYCL 的两个硬前提（缺一个编不过，本脚本已代劳）：
rem    1) 先 call vcvars64.bat 再 call setvars.bat（反了 icpx 找不到 MSVC 的标准库头）
rem    2) 必须用 Ninja 生成器（MSBuild 剥环境 -> LNK1104: libmmd.lib）
rem
rem  注意：本文件必须是 CRLF 行尾（LF 会让 cmd.exe 把注释当命令执行）。见 .editorconfig。
rem ============================================================================
setlocal
cd /d "%~dp0"

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cuda"
set "MODELARG=%~2"
if "%MODELARG%"=="" set "MODELARG=7b"
set "DOCLEAN=%~3"

if /i "%BACKEND%"=="cuda" goto be_cuda
if /i "%BACKEND%"=="sycl" goto be_sycl
if /i "%BACKEND%"=="host" goto be_host
echo 未知后端: %BACKEND%  [cuda ^| sycl ^| host]
exit /b 2
:be_cuda
set "NOVA_BACKEND=CUDA"
goto model
:be_sycl
set "NOVA_BACKEND=SYCL"
goto model
:be_host
set "NOVA_BACKEND=HOST"

:model
if /i "%MODELARG%"=="7b"   set "NOVA_MODEL=QWEN25_7B" & goto cm
if /i "%MODELARG%"=="1.5b" set "NOVA_MODEL=QWEN25_1P5B" & goto cm
if /i "%MODELARG%"=="dsv2" set "NOVA_MODEL=DEEPSEEK_V2_LITE" & goto cm
echo 未知模型: %MODELARG%  [7b ^| 1.5b ^| dsv2]
exit /b 2

:cm
set "CMAKE_EXE="
where cmake >nul 2>&1
if not errorlevel 1 set "CMAKE_EXE=cmake"
if defined CMAKE_EXE goto have_cmake
set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if exist "%CMAKE_EXE%" goto have_cmake
echo [build] 找不到 cmake：装 CMake 3.20+ 或 VS 的 "CMake 工具" 组件
exit /b 2

:have_cmake
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" goto no_vcvars
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
:no_vcvars
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"

if /i not "%NOVA_BACKEND%"=="SYCL" goto no_oneapi
if not exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" goto no_oneapi
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
:no_oneapi

set "NINJA="
if /i not "%NOVA_BACKEND%"=="SYCL" goto runbuild
where ninja >nul 2>&1
if not errorlevel 1 set "NINJA=ninja"
if defined NINJA goto have_ninja
set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if exist "%NINJA%" goto have_ninja
echo [build] SYCL 需要 ninja：装 ninja 或用 VS 安装器勾上 "CMake 工具"
exit /b 2
:have_ninja

:runbuild
echo [build] backend=%NOVA_BACKEND% model=%NOVA_MODEL%
set "BDIR=build-cmake\%NOVA_BACKEND%"
if /i "%DOCLEAN%"=="clean" if exist "%BDIR%" rmdir /s /q "%BDIR%"

if /i "%NOVA_BACKEND%"=="SYCL" goto build_sycl
"%CMAKE_EXE%" -B "%BDIR%" -DNOVA_BACKEND=%NOVA_BACKEND% -DNOVA_MODEL=%NOVA_MODEL% || exit /b 1
"%CMAKE_EXE%" --build "%BDIR%" --config Release || exit /b 1
goto done

:build_sycl
"%CMAKE_EXE%" -B "%BDIR%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DNOVA_BACKEND=SYCL -DNOVA_MODEL=%NOVA_MODEL% || exit /b 1
"%CMAKE_EXE%" --build "%BDIR%" || exit /b 1

:done
echo.
echo [build] 产物:
dir /b "%BDIR%\*.exe" 2>nul
dir /b "%BDIR%\Release\*.exe" 2>nul
echo.
echo [build] 数值闸门（不需要显卡）:
echo     ctest --test-dir "%BDIR%" -C Release --output-on-failure
exit /b 0
