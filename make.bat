@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
if not exist build\sycl mkdir build\sycl
icpx --version > build\sycl\ver.txt 2>&1
icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=0 -Iinclude -Icode\device\sycl -Isrc\kernel\sycl code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv4.cpp src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp tests\sycl_bench.cpp -o build\sycl\sycl_bench.exe > build\sycl\build2.log 2>&1
echo EXITCODE=%ERRORLEVEL% >> build\sycl\build2.log
