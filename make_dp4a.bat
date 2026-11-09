@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
set DP4A=0
icpx -fsycl -std=c++17 -c -DNOVA_PROBE=1 code\device\sycl\dp4a_probe.cpp -o build\sycl\p1.obj >nul 2>&1
if not errorlevel 1 set DP4A=1
if "%DP4A%"=="0" ( icpx -fsycl -std=c++17 -c -DNOVA_PROBE=2 code\device\sycl\dp4a_probe.cpp -o build\sycl\p2.obj >nul 2>&1
  if not errorlevel 1 set DP4A=2 )
echo DP4A=%DP4A%
icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=%DP4A% -Iinclude -Icode\device\sycl -Isrc\kernel\sycl code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv4.cpp src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp tests\sycl_bench.cpp -o build\sycl\sycl_bench.exe > build\sycl\build_dp4a.log 2>&1
echo BUILD_EXIT=%ERRORLEVEL% >> build\sycl\build_dp4a.log
if %ERRORLEVEL%==0 ( build\sycl\sycl_bench.exe --gguf C:\nova\build\q4k_15b.gguf > bench_dp4a.txt 2>&1 )
echo DONE >> build\sycl\build_dp4a.log
