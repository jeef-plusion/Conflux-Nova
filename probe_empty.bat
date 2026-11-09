@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova
icpx -fsycl -O3 -std=c++17 tools\dispatch_probe.cpp -o build\sycl\dispatch_probe.exe > build\sycl\probe_build.log 2>&1
echo BUILD=%ERRORLEVEL% >> build\sycl\probe_build.log
build\sycl\dispatch_probe.exe > dispatch_out.txt 2>&1
