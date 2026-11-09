@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
del build\sycl\sg_result.txt 2>nul
for %%S in (16 32) do (
  icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=0 -DNOVA_SYCL_SG=%%S -Iinclude -Icode\device\sycl -Isrc\kernel\sycl code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv4.cpp src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp tests\sycl_bench.cpp -o build\sycl\bench_sg%%S.exe > build\sycl\sg%%S_build.log 2>&1
  if errorlevel 1 ( echo FAIL%%S >> build\sycl\sg_result.txt ) else (
    build\sycl\bench_sg%%S.exe --gguf C:\nova\build\q4k_15b.gguf > build\sycl\sg%%S_bench.txt 2>&1
    echo OK%%S >> build\sycl\sg_result.txt )
)
echo ALLDONE >> build\sycl\sg_result.txt
