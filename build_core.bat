@echo off
rem V2.0.0-Core 唯一构建入口：一个模型一个二进制
setlocal
if "%1"=="" (set MODEL=NOVA_MODEL_QWEN25_7B) else (set MODEL=%1)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=E:\12\bin;%PATH%

if not exist build mkdir build
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -D%MODEL% -Iinclude -Icode ^
     code\kernel\1080ti\gemv_q4k.cu code\kernel\1080ti\attn.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp code\main.cpp ^
     -o build\nova_core.exe
echo BUILD=%ERRORLEVEL%