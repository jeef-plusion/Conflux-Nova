@echo off
rem 一个模型一个二进制：build.bat NOVA_MODEL_QWEN25_1P5B
setlocal
if "%1"=="" (set MODEL=NOVA_MODEL_QWEN25_7B) else (set MODEL=%1)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=E:\12\bin;%PATH%

if not exist build mkdir build
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -D%MODEL% -Iinclude ^
     src\kernel\q4k_gemv.cu src\kernel\norm.cu src\kernel\rope.cu src\kernel\attn.cu src\kernel\mmq.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\main.cpp ^
     -o build\nova.exe
echo BUILD=%ERRORLEVEL%