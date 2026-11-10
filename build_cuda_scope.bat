@echo off
rem ============================================================================
rem nova_v2/build_cuda_scope.bat - NovaScope CUDA driver (GTX 1080 Ti, sm_61)
rem
rem Builds tests/cuda_scope.cpp together with the real CUDA kernels and loader,
rem exactly like build_core.bat does for nova_core.exe.
rem
rem Usage:
rem   build_cuda_scope.bat                 (7B, NOVA_MODEL_QWEN25_7B)
rem   build_cuda_scope.bat NOVA_MODEL_QWEN25_1P5B
rem
rem ASCII only, no (...) blocks, CRLF.
rem ============================================================================
setlocal
if "%1"=="" (set MODEL=NOVA_MODEL_QWEN25_7B) else (set MODEL=%1)
set "ROOT=%~dp0"
pushd "%ROOT%"

if not exist build mkdir build
if not exist build\cudascope mkdir build\cudascope

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not exist "E:\12\bin\nvcc.exe" goto no_cuda
set PATH=E:\12\bin;%PATH%

echo [cudascope] nvcc sm_61 -D%MODEL%
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -D%MODEL% -DNOVA_MAX_SESSIONS=1 -DNOVA_MAX_CTX=256 -DNOVA_SESS_CTX=32 -Iinclude -Icode ^
     code\kernel\1080ti\gemv_q4k.cu code\kernel\1080ti\attn.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp ^
     tests\cuda_scope.cpp ^
     -o build\cuda_scope.exe
if errorlevel 1 goto fail

echo.
echo [cudascope] OK - build\cuda_scope.exe
echo [cudascope] run: set NOVA_SCOPE=1 ^&^& build\cuda_scope.exe build\q4k.gguf 16
goto done

:no_cuda
echo [cudascope] FATAL: E:\12\bin\nvcc.exe not found
popd
exit /b 2

:fail
echo [cudascope] FATAL: build failed
popd
exit /b 1

:done
popd
exit /b 0
