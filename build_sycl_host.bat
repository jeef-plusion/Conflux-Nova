@echo off
rem ============================================================================
rem nova_v2/build_sycl_host.bat - GEMV numeric check on CPU (no Intel GPU, no oneAPI)
rem
rem Runs the REAL kernel body from src/kernel/sycl/q4k_gemv.cpp on CPU threads
rem through the tiny SYCL shim in tests/host_sycl (thread = work-item,
rem condition variable = barrier, local_accessor = one heap block per WG), with
rem the activation produced by the production quant.cpp, and compares against
rem "dequantize + double accumulate". Needs cl.exe only, not icpx.
rem
rem It proves lane mapping / scheme C segment split / cross-subgroup reduction /
rem local+barrier usage. It does NOT prove bandwidth - that needs the target
rem machine and sycl_bench (docs/SYCL_PORT.md section 11.5).
rem
rem Usage:
rem   build_sycl_host.bat             1.5B + SG=16 (the PERF_BASELINE GEMV setup)
rem   build_sycl_host.bat 1.5b 32     1.5B + SG=32
rem   build_sycl_host.bat 7b 16       7B + SG=16
rem
rem Output: build\sycl_gemv_host_check_<model>_sg<sg>.exe
rem Run:    build\sycl_gemv_host_check_1p5b_sg16.exe build\q4k_15b.gguf
rem
rem NOTE: deliberately ASCII-only, no (...) blocks and no ^ continuation lines.
rem       The .bat files in this repo use LF line endings, and cmd.exe under a
rem       non-UTF-8 console code page mis-parses non-ASCII comments in such files
rem       (it swallows the following newline). single-line + goto is the safe form.
rem ============================================================================
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

rem ---- 0. model + subgroup -----------------------------------------------------
set "MODEL_TAG=1p5b"
set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_1P5B"
if /I "%~1"=="7b" goto use_7b
if /I "%~1"=="8b" goto use_8b
goto model_ready
:use_7b
set "MODEL_TAG=7b"
set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_7B"
goto model_ready
:use_8b
set "MODEL_TAG=8b"
set "MODEL_DEF=-DNOVA_SYCL_MODEL_R1_LLAMA8B"
:model_ready

set "SG=16"
if not "%~2"=="" set "SG=%~2"

rem ---- 1. find MSVC (call vcvars64 if cl is not on PATH) ------------------------
where cl >nul 2>nul
if not errorlevel 1 goto have_cl
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" goto no_cl
echo [host] cl not on PATH, calling BuildTools vcvars64.bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
where cl >nul 2>nul
if errorlevel 1 goto no_cl
:have_cl

rem ---- 2. build ----------------------------------------------------------------
if not exist build mkdir build
if not exist "build\hc_%MODEL_TAG%_%SG%" mkdir "build\hc_%MODEL_TAG%_%SG%"

echo [host] building model=%MODEL_TAG% SG=%SG%
cl /nologo /std:c++17 /O2 /EHsc /utf-8 /wd4068 /wd5030 /Iinclude /Icode\device\sycl /Isrc\kernel\sycl /Itests\host_sycl %MODEL_DEF% -DNOVA_SYCL_DP4A=0 -DNOVA_SYCL_SG=%SG% -DNOVA_SYCL_GEMV_ROWS=8 tests\sycl_gemv_host_check.cpp src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\elem.cpp src\loader\gguf.cpp /Fe:build\sycl_gemv_host_check_%MODEL_TAG%_sg%SG%.exe /Fo:build\hc_%MODEL_TAG%_%SG%\
if errorlevel 1 goto fail

echo.
echo [host] OK - build\sycl_gemv_host_check_%MODEL_TAG%_sg%SG%.exe
if /I "%MODEL_TAG%"=="7b" goto hint_7b
echo [host] run: build\sycl_gemv_host_check_%MODEL_TAG%_sg%SG%.exe build\q4k_15b.gguf
goto done
:hint_7b
echo [host] run: build\sycl_gemv_host_check_7b_sg%SG%.exe build\q4k.gguf
goto done

:no_cl
echo.
echo [host] FATAL: cl.exe not found. Install VS 2022 BuildTools (C++ workload) and retry.
popd
exit /b 2

:fail
echo.
echo [host] FATAL: build failed.
popd
exit /b 1

:done
popd
exit /b 0
