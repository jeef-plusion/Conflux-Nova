@echo off
rem build_soa_bench.bat - AoS vs SoA read-bandwidth micro benchmark (Arc)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%~dp0"
if not exist build\sycl mkdir build\sycl
icpx -fsycl -O2 -std=c++17 -Iinclude -Icode tests\sycl_soa_bench.cpp -o build\sycl\sycl_soa_bench.exe
echo BUILD=%ERRORLEVEL%
