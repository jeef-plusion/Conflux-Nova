@echo off
rem ============================================================================
rem nova_v2/build_dsv2_host.bat - DeepSeek-V2-Lite (MLA + MoE) host check
rem
rem Runs the real mla.cpp / moe.cpp kernels on CPU threads through tests/host_sycl
rem and compares them against double-precision references. Needs MSVC only.
rem
rem NO MODEL FILE IS REQUIRED - that is the point: the V2-Lite weight file is not
rem available yet (the target machine has no network), so the operators are
rem written and verified against synthetic inputs first.
rem
rem Usage:
rem   build_dsv2_host.bat
rem   build\sycl_dsv2_host_check.exe
rem
rem NOTE: ASCII-only, single-line, no (...) blocks (see HANDOFF_ARC on LF bats).
rem ============================================================================
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

where cl >nul 2>nul
if not errorlevel 1 goto have_cl
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" goto no_cl
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>nul
if errorlevel 1 goto no_cl
:have_cl

if not exist build mkdir build
if not exist build\dsv2 mkdir build\dsv2

echo [dsv2] building host check (MSVC only, no GPU, no model file)
cl /nologo /std:c++17 /O2 /EHsc /utf-8 /wd4068 /wd5030 /Iinclude /Icode\device\sycl /Isrc\kernel\sycl /Itests\host_sycl /DNOVA_SYCL_DP4A=0 /DNOVA_SYCL_SG=32 /DNOVA_SYCL_MODEL_QWEN25_1P5B tests\sycl_dsv2_host_check.cpp /Fe:build\sycl_dsv2_host_check.exe /Fo:build\dsv2\
if errorlevel 1 goto fail

echo.
echo [dsv2] OK - build\sycl_dsv2_host_check.exe
echo [dsv2] run: build\sycl_dsv2_host_check.exe
goto done

:no_cl
echo.
echo [dsv2] FATAL: cl.exe not found. Install VS 2022 BuildTools (C++ workload) and retry.
popd
exit /b 2

:fail
echo.
echo [dsv2] FATAL: build failed.
popd
exit /b 1

:done
popd
exit /b 0
