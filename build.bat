@echo off
rem ============================================================================
rem  build.bat - 统一构建入口（内部只调 CMake）
rem
rem  用法:
rem    build.bat                       :: CUDA + 7B（默认）
rem    build.bat sycl 1.5b             :: SYCL + 1.5B
rem    build.bat host                  :: 只编 CPU 垫片（任何机器都能过）
rem    build.bat cuda 7b clean         :: 先删 build-cmake 再配
rem
rem  后端: cuda | sycl | host      模型: 7b | 1.5b | dsv2
rem  依赖: CMake 3.20+ 与对应工具链（cuda -> nvcc；sycl -> icpx/oneAPI）
rem
rem  旧路径 build_core.bat 仍然可用（绕过 CMake 直接调 nvcc），但新代码请优先走这里。
rem ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cuda"
set "MODELARG=%~2"
if "%MODELARG%"=="" set "MODELARG=7b"

if /i "%BACKEND%"=="cuda" (set "NOVA_BACKEND=CUDA") else (
if /i "%BACKEND%"=="sycl" (set "NOVA_BACKEND=SYCL") else (
if /i "%BACKEND%"=="host" (set "NOVA_BACKEND=HOST") else (echo 未知后端: %BACKEND% & exit /b 2)))

if /i "%MODELARG%"=="7b"   (set "NOVA_MODEL=QWEN25_7B") else (
if /i "%MODELARG%"=="1.5b" (set "NOVA_MODEL=QWEN25_1P5B") else (
if /i "%MODELARG%"=="dsv2" (set "NOVA_MODEL=DEEPSEEK_V2_LITE") else (echo 未知模型: %MODELARG% & exit /b 2)))

rem ---- CMake：优先 PATH，其次用 VS BuildTools 自带的那份 ----
where cmake >nul 2>&1
if errorlevel 1 (
    set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if not exist "!CMAKE_EXE!" (
        echo [build] 找不到 cmake：装 CMake 3.20+ 或 Visual Studio 的 "CMake 工具" 组件
        exit /b 2
    )
) else (
    set "CMAKE_EXE=cmake"
)

rem ---- MSVC 环境 ----
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if defined CUDA_PATH (set "PATH=%CUDA_PATH%\bin;%PATH%")

rem ---- SYCL 需要 oneAPI 环境 ----
if /i "%NOVA_BACKEND%"=="SYCL" (
    if exist "%ONEAPI_ROOT%\setvars.bat" (call "%ONEAPI_ROOT%\setvars.bat" >nul 2>&1)
    if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1)
)



echo [build] backend=%NOVA_BACKEND% model=%NOVA_MODEL%

rem 每个后端一个独立构建目录：同一个目录里换后端会串味（CMake 缓存 + 旧产物）
set "BDIR=build-cmake\%NOVA_BACKEND%"
if /i "%~3"=="clean" (if exist "%BDIR%" rmdir /s /q "%BDIR%")


"%CMAKE_EXE%" --build "%BDIR%" --config Release || exit /b 1

echo.
echo [build] 产物:
dir /b "%BDIR%"\Release\*.exe 2>nul
echo.
echo [build] 数值闸门（不需要显卡）:
echo     ctest --test-dir "%BDIR%" -C Release --output-on-failure
exit /b 0
