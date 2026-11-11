@echo off
rem ============================================================================
rem  nova_sycl.exe 一键跑分 / one-click benchmark
rem
rem  用法:
rem    1) 把 q4k_15b.gguf 放到本目录（或改名成 model.gguf）
rem    2) 双击本文件
rem
rem  模型也可以传路径:  run_bench.bat D:\models\q4k_15b.gguf
rem
rem  运行时:
rem    - 如果本机装了 Intel oneAPI，本脚本会自动 call setvars.bat（最稳）；
rem    - 没装时退回本目录自带的 dll（sycl9 / libmmd / svml_dispmd / ur_*）；
rem    - 两种都不行，说明缺 Intel Arc 显卡驱动 —— 装官方驱动即可。
rem ============================================================================
setlocal
cd /d "%~dp0"

set "EXE=nova_sycl.exe"
set "MODEL=%~1"
if "%MODEL%"=="" if exist "model.gguf" set "MODEL=model.gguf"
if "%MODEL%"=="" if exist "q4k_15b.gguf" set "MODEL=q4k_15b.gguf"

if "%MODEL%"=="" goto nomodel
if not exist "%MODEL%" goto nomodel

rem ---- 运行时环境：优先用已安装的 oneAPI，其次用自带 dll ----
set "SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if exist "%SETVARS%" (
    echo [runtime] 使用本机 oneAPI 运行时
    call "%SETVARS%" >nul 2>&1
) else (
    echo [runtime] 未检测到 oneAPI，使用本目录自带的 dll
    set "PATH=%~dp0;%PATH%"
)

echo.
echo ============================================================
echo  Conflux Nova - SYCL backend (Intel Arc) end-to-end decode
echo  model : %MODEL%
echo ============================================================
echo.

"%EXE%" --gguf "%MODEL%" --tokens 32 --warmup 2
set RC=%ERRORLEVEL%

echo.
echo ------------------------------------------------------------
echo  exit code = %RC%
if not "%RC%"=="0" (
    echo  非 0 退出：
    echo    -1073741515 / 0xC0000135  ^= 缺 dll，见 QUICKSTART.md 第三节
    echo    -1073740791 / 0xC0000409  ^= SYCL 运行时初始化失败，装 Intel Arc 驱动
    echo    其它                        ^= 把窗口截图发出来
)
echo  "tok/s" 那行就是端到端速度；"ms/token" 越小越好。
echo ------------------------------------------------------------
pause
exit /b %RC%

:nomodel
echo.
echo ============================================================
echo  没有找到模型文件。
echo.
echo  请把 q4k_15b.gguf 放到:
echo      %~dp0
echo  或者改名为 model.gguf，或者传路径:
echo      run_bench.bat D:\models\q4k_15b.gguf
echo ============================================================
pause
exit /b 2
