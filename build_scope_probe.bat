@echo off
rem build_scope_probe.bat - minimal SYCL profiling/barrier probe (Arc, no model needed)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
if not exist build\sycl mkdir build\sycl
icpx -fsycl -O2 -std=c++17 -DNOVA_SYCL_MODEL_QWEN25_1P5B -Iinclude -Icode -Icode\model\sycl -Isrc\kernel\sycl tests\sycl_scope_probe.cpp -o build\sycl\sycl_scope_probe.exe
echo BUILD=%ERRORLEVEL%
