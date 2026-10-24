@echo off
rem ============================================================================
rem nova_v2/build_sycl.bat — SYCL 后端的唯一构建入口
rem 目标机：Intel Arc 140T / Core Ultra 7 255H / Windows 11 / oneAPI DPC++
rem
rem 与 build_core.bat（CUDA sm_61 那条线）互不影响：不同的源文件、不同的输出目录。
rem
rem 用法:
rem   build_sycl.bat            编译 7B 版（默认）
rem   build_sycl.bat 1.5b       编译 1.5B 版
rem   build_sycl.bat 8b         编译 Llama-3.1-8B 版
rem
rem 做的事:
rem   1) 找 icpx（PATH 里没有就 call oneAPI 的 setvars.bat）
rem   2) 用 code\device\sycl\dp4a_probe.cpp 探 sycl::dp4a 的两种候选写法
rem      -> NOVA_SYCL_DP4A=1/2，都不过则 0（纯标量）并大字告警
rem   3) icpx -fsycl 一把编完 -> build\sycl\sycl_bench.exe
rem ============================================================================
setlocal enabledelayedexpansion
set ROOT=%~dp0
pushd "%ROOT%"

if not exist build        mkdir build
if not exist build\sycl   mkdir build\sycl

rem ---- 0. 模型 ---------------------------------------------------------------
rem 注意: set 的值里带括号时必须整体加引号，否则块内的 ")" 会提前结束 if 块。
set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_7B"
set "MODEL_TAG=7B (Qwen2.5-7B-Instruct)"
if /I "%~1"=="1.5b" (
    set "MODEL_DEF=-DNOVA_SYCL_MODEL_QWEN25_1P5B"
    set "MODEL_TAG=1.5B (Qwen2.5-1.5B-Instruct)"
)
if /I "%~1"=="8b" (
    set "MODEL_DEF=-DNOVA_SYCL_MODEL_R1_LLAMA8B"
    set "MODEL_TAG=8B (Llama-3.1-8B)"
)

rem ---- 1. 找编译器 -----------------------------------------------------------
where icpx >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" (
        echo [build_sycl] PATH 里没有 icpx，载入 "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
        call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul
    )
)
where icpx >nul 2>nul
if errorlevel 1 (
    echo.
    echo [build_sycl] FATAL: 找不到 icpx。
    echo   装 Intel oneAPI Base Toolkit ^(>= 2024.0^)，然后在 "Intel oneAPI command prompt"
    echo   里跑本脚本；或在别的 shell 里先 call setvars.bat。
    echo.
    popd
    exit /b 2
)
echo [build_sycl] 编译器:
icpx --version

rem ---- 2. dp4a 探针 ----------------------------------------------------------
set PROBE_SRC=code\device\sycl\dp4a_probe.cpp
set DP4A=0
for %%N in (1 2) do (
    if "!DP4A!"=="0" (
        icpx -fsycl -std=c++17 -c -DNOVA_PROBE=%%N "%PROBE_SRC%" -o "build\sycl\probe_%%N.obj" >nul 2>nul
        if not errorlevel 1 (
            set DP4A=%%N
            echo [build_sycl] dp4a 探针 %%N: 通过
            del /q "build\sycl\probe_%%N.obj" >nul 2>nul
        ) else (
            echo [build_sycl] dp4a 探针 %%N: 编译失败（该写法不可用）
        )
    )
)
if "%DP4A%"=="0" (
    echo.
    echo [build_sycl] ****************************************************************
    echo [build_sycl]  警告: 没探到硬件 dp4a intrinsic，本次构建落回纯标量。
    echo [build_sycl]        正确性不受影响；第 4 项性能门槛（dp4a >= 1.8x）必然不过。
    echo [build_sycl]        处理办法见 docs/SYCL_PORT.md 7.2。
    echo [build_sycl] ****************************************************************
    echo.
)
echo [build_sycl] NOVA_SYCL_DP4A=%DP4A%   model=%MODEL_TAG%

rem ---- 3. 正式构建 -----------------------------------------------------------
set SYCL_SRC=code\device\sycl\device.cpp ^
 src\kernel\sycl\q4k_gemv.cpp ^
 src\kernel\sycl\q4k_gemv4.cpp ^
 src\kernel\sycl\quant.cpp ^
 src\kernel\sycl\norm.cpp ^
 src\kernel\sycl\rope.cpp ^
 src\kernel\sycl\elem.cpp ^
 src\kernel\sycl\attn.cpp ^
 src\loader\gguf.cpp ^
 tests\sycl_bench.cpp

echo [build_sycl] 开始编译（首次约 2~5 分钟，SYCL 要把每个 kernel 编成 SPIR-V）...
icpx -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel ^
     %MODEL_DEF% -DNOVA_SYCL_DP4A=%DP4A% ^
     -Iinclude -Icode\device\sycl -Isrc\kernel\sycl ^
     %SYCL_SRC% ^
     -o build\sycl\sycl_bench.exe
if errorlevel 1 (
    echo.
    echo [build_sycl] FATAL: 编译失败（见上面的 icpx 报错）。
    popd
    exit /b 1
)

echo.
echo [build_sycl] OK -> build\sycl\sycl_bench.exe
echo [build_sycl] 下一步（两条命令出数据）:
echo     build\sycl\sycl_bench.exe --gguf build\q4k.gguf
echo.
echo [build_sycl] 备注: 1.5B 模型请看 build\q4k_15b.gguf，用 "build_sycl.bat 1.5b" 编译对应二进制。
popd
exit /b 0
