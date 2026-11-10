@echo off
rem ============================================================================
rem nova_v2/build_scope_host.bat - NovaScope core self-check (no GPU, no oneAPI)
rem
rem Compiles tests/scope_host_check.cpp (synthetic event stream) with MSVC and
rem runs it. Verifies the statistics, the Chrome trace JSON and the traffic-light
rem verdicts before the SYCL probe is attached on the target machine.
rem
rem Usage:
rem   build_scope_host.bat
rem   build\scope_host_check.exe
rem
rem ASCII only, no (...) blocks, CRLF (see HANDOFF_ARC on LF bats).
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
if not exist build\scope mkdir build\scope

echo [scope] building core self-check (MSVC only)
cl /nologo /std:c++17 /O2 /EHsc /utf-8 /wd4068 /Iinclude /Icode /Icode\model\sycl tests\scope_host_check.cpp /Fe:build\scope_host_check.exe /Fo:build\scope\
if errorlevel 1 goto fail

echo.
echo [scope] OK - build\scope_host_check.exe
echo [scope] run: build\scope_host_check.exe
goto done

:no_cl
echo.
echo [scope] FATAL: cl.exe not found. Install VS 2022 BuildTools (C++ workload).
popd
exit /b 2

:fail
echo.
echo [scope] FATAL: build failed.
popd
exit /b 1

:done
popd
exit /b 0
