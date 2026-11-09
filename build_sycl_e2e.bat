@echo off
rem ============================================================================
rem nova_v2/build_sycl_e2e.bat - build the end-to-end decode runner (real model)
rem
rem Builds code/model/sycl/runner.cpp + the whole SYCL operator layer + the GGUF
rem loader + tests/sycl_e2e.cpp into build\sycl\sycl_e2e.exe.
rem
rem Requires: vcvars64.bat then setvars.bat (in that order), icpx 2026.1.x.
rem
rem Usage:
rem   build_sycl_e2e.bat            1.5B (default)
rem   build_sycl_e2e.bat 7b         7B
rem
rem Run:
rem   build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 32
rem
rem NOTE: ASCII-only, single-line commands, no (...) blocks - the .bat files in
rem this repo are LF, and cmd.exe splits non-ASCII lines at LF (see HANDOFF_ARC).
rem ============================================================================
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

set "MODEL_TAG=1p5b"
set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_1P5B"
if /I "%~1"=="7b" goto use_7b
goto model_ready
:use_7b
set "MODEL_TAG=7b"
set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_7B"
:model_ready

rem NOTE: DP4A=0 and SG=16 = exactly the configuration every PERF_BASELINE number
rem was taken with. Both probes are skipped on purpose so this build is reproducible.
set "FLAGS=-fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel %MODEL_DEF% -DNOVA_SYCL_DP4A=0 -DNOVA_SYCL_SG=16 -DNOVA_SYCL_GEMV_ROWS=8"

where icpx >nul 2>nul
if not errorlevel 1 goto have_icpx
if not exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" goto no_icpx
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
where icpx >nul 2>nul
if errorlevel 1 goto no_icpx
:have_icpx

if not exist build mkdir build
if not exist build\sycl mkdir build\sycl

echo [e2e] building model=%MODEL_TAG%
icpx %FLAGS% -Iinclude -Icode -Icode\device\sycl -Icode\model\sycl -Isrc\kernel\sycl code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv4.cpp src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp code\model\sycl\runner.cpp tests\sycl_e2e.cpp -o build\sycl\sycl_e2e.exe
if errorlevel 1 goto fail

echo.
echo [e2e] OK - build\sycl\sycl_e2e.exe
if /I "%MODEL_TAG%"=="7b" goto hint_7b
echo [e2e] run: build\sycl\sycl_e2e.exe --gguf build\q4k_15b.gguf --tokens 32
goto done
:hint_7b
echo [e2e] run: build\sycl\sycl_e2e.exe --gguf build\q4k.gguf --tokens 32
goto done

:no_icpx
echo.
echo [e2e] FATAL: icpx not found. Install oneAPI Base Toolkit or call setvars.bat first.
popd
exit /b 2

:fail
echo.
echo [e2e] FATAL: build failed (see errors above).
popd
exit /b 1

:done
popd
exit /b 0
