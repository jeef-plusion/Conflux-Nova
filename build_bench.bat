@echo off
rem 构建 benchmark（含权重上传 + 逐层 forward）
setlocal
if "%1"=="" (set MODEL=NOVA_MODEL_QWEN25_7B) else (set MODEL=%1)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=E:\12\bin;%PATH%

if not exist build mkdir build
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -D%MODEL% -Iinclude ^
     src\kernel\q4k_gemv.cu src\kernel\norm.cu src\kernel\rope.cu src\kernel\attn.cu src\kernel\elem.cu src\kernel\mmq.cu ^
     src\loader\gguf.cpp src\loader\validate.cpp src\model.cpp bench\benchmark.cpp ^
     -o build\bench.exe
echo BUILD=%ERRORLEVEL%