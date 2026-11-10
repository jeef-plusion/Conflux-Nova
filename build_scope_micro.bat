@echo off
rem build_scope_micro.bat - NovaScope CUDA micro driver (real Q4_K GEMV kernel)
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set PATH=E:\12\bin;%PATH%
if not exist build mkdir build
echo [mic] nvcc sm_61
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -DNOVA_MODEL_QWEN25_1P5B -Iinclude -Icode ^
     code\kernel\1080ti\gemv_q4k.cu code\kernel\1080ti\attn.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp ^
     tests\cuda_scope_micro.cpp ^
     -o build\cuda_scope_micro.exe
if errorlevel 1 goto fail
echo [mic] OK - build\cuda_scope_micro.exe
goto done
:fail
echo [mic] FATAL
popd
exit /b 1
:done
popd
exit /b 0
