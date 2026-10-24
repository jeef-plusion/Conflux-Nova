@echo off
rem 全部验证：CPU 对拍 + 真卡 GEMM/RoPE/Attention + loader 正负例
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=E:\12\bin;%PATH%
if not exist build mkdir build

echo === CPU: Q4_K dequant / warp 分工 ===
nvcc -std=c++17 -O2 -Xcompiler /utf-8 -Iinclude tests\selftest_q4k.cpp -o build\selftest_q4k.exe || exit /b 1
build\selftest_q4k.exe || exit /b 1

echo === GPU: q_proj<1>/<4> ===
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -DNOVA_MODEL_QWEN25_7B -Iinclude tests\gpu_gemv.cu src\kernel\q4k_gemv.cu -o build\gpu_gemv.exe || exit /b 1
build\gpu_gemv.exe || exit /b 1

echo === GPU: RoPE / Attention ===
nvcc -std=c++17 -O3 -arch=sm_61 -Wno-deprecated-gpu-targets -Xcompiler /utf-8 ^
     -DNOVA_MODEL_QWEN25_7B -Iinclude tests\gpu_misc.cu src\kernel\rope.cu src\kernel\attn.cu -o build\gpu_misc.exe || exit /b 1
build\gpu_misc.exe || exit /b 1

echo ALL TESTS PASS
