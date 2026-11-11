@echo off
rem pd_sweep.bat - build sycl_e2e with a given prefetch depth and measure per-op bandwidth
rem usage: pd_sweep.bat <PD>     (writes build\sycl\pd_<PD>.txt)
setlocal
set PD=%1
rem 妯″瀷璺緞鍙互鐢ㄧ浜屼釜鍙傛暟瑕嗙洊锛堟ā鍨嬫尓杩囩洰褰曟椂涓嶇敤鏀硅剼鏈級
if "%2"=="" (set MODEL=C:\nova\build\q4k_15b.gguf) else (set MODEL=%2)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d C:\nova

echo === build PD=%PD% ===
icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel ^
     -DNOVA_SYCL_MODEL_QWEN25_1P5B -DNOVA_SYCL_DP4A=0 -DNOVA_SYCL_SG=16 -DNOVA_SYCL_GEMV_ROWS=8 ^
     -DNOVA_SYCL_GEMV_PD=%PD% ^
     -Iinclude -Icode -Icode\device\sycl -Icode\model\sycl -Isrc\kernel\sycl ^
     code\device\sycl\device.cpp src\kernel\sycl\q4k_gemv.cpp src\kernel\sycl\q4k_gemv_soa.cpp src\kernel\sycl\q4k_gemv4.cpp ^
     src\kernel\sycl\quant.cpp src\kernel\sycl\norm.cpp src\kernel\sycl\rope.cpp ^
     src\kernel\sycl\elem.cpp src\kernel\sycl\attn.cpp src\loader\gguf.cpp ^
     code\model\sycl\runner.cpp tests\sycl_e2e.cpp ^
     -o build\sycl\sycl_e2e_pd%PD%.exe
if errorlevel 1 goto fail

if exist e2e_pd_%PD%.txt del e2e_pd_%PD%.txt
set NOVA_SCOPE=1
build\sycl\sycl_e2e_pd%PD%.exe --gguf %MODEL% --tokens 10 --warmup 2 --scope > e2e_pd_%PD%.txt 2>&1
echo RUN_EXIT=%ERRORLEVEL%
findstr /C:"ms/token" e2e_pd_%PD%.txt
findstr /C:"gemv.down" /C:"gemv.up" /C:"gemv.gate" /C:"GPU" e2e_pd_%PD%.txt
goto done
:fail
echo BUILD_FAILED
:done
exit /b 0
