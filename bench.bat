@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
build\sycl\sycl_bench.exe --gguf build\q4k_15b.gguf > bench_out.txt 2>&1
echo EXIT=%ERRORLEVEL% >> bench_out.txt
