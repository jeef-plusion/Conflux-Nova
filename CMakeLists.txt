cmake_minimum_required(VERSION 3.20)

# ============================================================================
#  Conflux Nova — 统一构建入口（取代根目录的 23 个 .bat）
#
#  用法:
#     cmake -B build-cmake -DNOVA_BACKEND=CUDA  -DNOVA_MODEL=QWEN25_7B
#     cmake -B build-cmake -DNOVA_BACKEND=SYCL  -DNOVA_MODEL=QWEN25_1P5B
#     cmake -B build-cmake -DNOVA_BACKEND=HOST            # 只编 CPU 垫片（CI 用）
#     cmake --build build-cmake -j
#     ctest --test-dir build-cmake --output-on-failure
#
#  选项:
#     NOVA_BACKEND   CUDA (默认) | SYCL | HOST
#     NOVA_MODEL     QWEN25_7B (默认) | QWEN25_1P5B | DEEPSEEK_V2_LITE
#     NOVA_BUILD_APPS / NOVA_BUILD_BENCH / NOVA_ENABLE_SCOPE   ON|OFF
#     NOVA_CUDA_ARCH sm_61 (默认，1080 Ti) | sm_75 | sm_86 | native
#
#  工程分层（逻辑上四块，物理路径见每节注释；物理迁移清单在 docs/LAYOUT.md）：
#     core   引擎：kernel + device/HAL + graph + model（不含任何 main）
#     apps   端到端入口：CLI（CUDA 辩论演示 / SYCL decode）
#     bench  微基准 + CPU 垫片对拍（不需要 GPU 就能跑的那部分）
#     novascope  探针：**只有头文件**，NOVA_ENABLE_SCOPE=OFF 时零代码
# ============================================================================

project(conflux_nova LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
#  选项
# ---------------------------------------------------------------------------
set(NOVA_BACKEND "CUDA" CACHE STRING "后端: CUDA | SYCL | HOST")
set_property(CACHE NOVA_BACKEND PROPERTY STRINGS CUDA SYCL HOST)

set(NOVA_MODEL "QWEN25_7B" CACHE STRING "模型: QWEN25_7B | QWEN25_1P5B | DEEPSEEK_V2_LITE")
set_property(CACHE NOVA_MODEL PROPERTY STRINGS QWEN25_7B QWEN25_1P5B DEEPSEEK_V2_LITE)

option(NOVA_BUILD_APPS   "构建端到端启动器 (apps/)" ON)
option(NOVA_BUILD_BENCH  "构建基准与垫片对拍 (bench/)" ON)
option(NOVA_ENABLE_SCOPE "把 NovaScope 探针编进引擎（纯头文件，关掉则零代码）" ON)

set(NOVA_CUDA_ARCH "sm_61" CACHE STRING "CUDA 架构 (sm_61 / sm_75 / sm_86 / native)")

string(TOUPPER "${NOVA_BACKEND}" NOVA_BACKEND_U)

# ---------------------------------------------------------------------------
#  模型宏
#  ⚠️ 两套宏名并存：CUDA 线用 NOVA_MODEL_*，SYCL 头用 NOVA_SYCL_MODEL_*（历史原因），
#     两个都要定义 —— 只定义一套会让另一条线静默回落到默认模型（踩过：kD 变成 3584）。
# ---------------------------------------------------------------------------
if(NOVA_MODEL STREQUAL "QWEN25_7B")
    set(NOVA_MODEL_MACRO      NOVA_MODEL_QWEN25_7B)
    set(NOVA_SYCL_MODEL_MACRO NOVA_SYCL_MODEL_QWEN25_7B)
elseif(NOVA_MODEL STREQUAL "QWEN25_1P5B")
    set(NOVA_MODEL_MACRO      NOVA_MODEL_QWEN25_1P5B)
    set(NOVA_SYCL_MODEL_MACRO NOVA_SYCL_MODEL_QWEN25_1P5B)
elseif(NOVA_MODEL STREQUAL "DEEPSEEK_V2_LITE")
    set(NOVA_MODEL_MACRO      NOVA_MODEL_DEEPSEEK_V2_LITE)
    set(NOVA_SYCL_MODEL_MACRO NOVA_SYCL_MODEL_DEEPSEEK_V2_LITE)
else()
    message(FATAL_ERROR "未知 NOVA_MODEL=${NOVA_MODEL}（可选 QWEN25_7B / QWEN25_1P5B / DEEPSEEK_V2_LITE）")
endif()

if(NOVA_ENABLE_SCOPE)
    add_compile_definitions(NOVA_SCOPE=1)
endif()

message(STATUS "Conflux Nova: backend=${NOVA_BACKEND_U} model=${NOVA_MODEL} scope=${NOVA_ENABLE_SCOPE}")

# ---------------------------------------------------------------------------
#  编译期卫生：源码里有大量中文注释与格式串，MSVC 必须用 /utf-8 才不会按 936 代码页
#  错解（实测：不传时中文格式串会让 MSVC 误报 "snprintf 参数不足"，运行期输出乱码）
# ---------------------------------------------------------------------------
if(MSVC)
    # 只给 C/C++ 用 —— nvcc 不认识 /utf-8，会直接报错（CUDA 侧走 -Xcompiler /utf-8）
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3>)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
endif()

# ---------------------------------------------------------------------------
#  core：引擎库，不含 main
#    [core] src/loader/*.cpp, src/model.cpp
# ---------------------------------------------------------------------------
# nova_engine 是 **CUDA 线**的引擎库（src/model.cpp 会拉进 cuda_runtime.h），
# 只有 CUDA 后端建它；HOST 与 SYCL 都不需要（SYCL 的引擎在 nova_sycl 那条命令里）。
if(NOVA_BACKEND_U STREQUAL "CUDA")
    add_library(nova_engine STATIC
        src/loader/gguf.cpp
        src/loader/validate.cpp
        src/model.cpp
    )
    target_include_directories(nova_engine PUBLIC include code)
    target_compile_definitions(nova_engine PUBLIC ${NOVA_MODEL_MACRO} ${NOVA_SYCL_MODEL_MACRO})
endif()

# ---------------------------------------------------------------------------
#  后端
# ---------------------------------------------------------------------------
if(NOVA_BACKEND_U STREQUAL "CUDA")
    # ⚠️ 架构必须在 enable_language(CUDA) **之前**设定：CMake 会先把
    # CMAKE_CUDA_ARCHITECTURES 初始化成 52，而 __dp4a 只在 sm_61+ 存在（compute_52 直接
    # 报 "identifier __dp4a is undefined"）。所以这里无条件覆盖。
    if(NOVA_CUDA_ARCH STREQUAL "native")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "" FORCE)
    else()
        string(REGEX REPLACE "^sm_" "" _arch "${NOVA_CUDA_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${_arch} CACHE STRING "" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # [core] src/kernel/1080ti/*.cu
    add_library(nova_kernels_cuda STATIC
        code/kernel/1080ti/gemv_q4k.cu
        code/kernel/1080ti/attn.cu
    )
    target_include_directories(nova_kernels_cuda PUBLIC include code)
    target_compile_definitions(nova_kernels_cuda PUBLIC ${NOVA_MODEL_MACRO} ${NOVA_SYCL_MODEL_MACRO})
    if(MSVC)
        target_compile_options(nova_kernels_cuda PRIVATE -Xcompiler /utf-8)
    endif()
    target_link_libraries(nova_engine PUBLIC nova_kernels_cuda)

    # [apps] code/main.cpp（CUDA 辩论演示）
    if(NOVA_BUILD_APPS)
        add_executable(nova_cuda code/main.cpp)
        target_link_libraries(nova_cuda PRIVATE nova_engine)
        set_target_properties(nova_cuda PROPERTIES OUTPUT_NAME "nova_core")
    endif()

elseif(NOVA_BACKEND_U STREQUAL "SYCL")
    # ---------------------------------------------------------------------
    #  SYCL 用**一条 icpx 命令**构建 —— 与已验证的 build_sycl_e2e.bat 完全相同。
    #
    #  为什么不走 MSBuild 的多目标（add_library + add_executable）：
    #   1) Windows 上 icpx 必须同时拿到 vcvars64 与 oneAPI 两套环境，缺一个就报
    #      "'type_traits' file not found"（实测）；按文件驱动时这个错误出现在每个 TU，
    #      被 MSBuild 吞掉后表现为"编到第 1 个 object 就停住"；
    #   2) 一次调用让 device code split 与链接在同一个进程里完成，比 11 个 TU 分别做
    #      device 编译再链接快得多。
    #
    #  所以这里把那条命令原样包成 custom target：CMake 负责"一键"，icpx 负责"编译"。
    #  前提：先 call vcvars64.bat，再 call setvars.bat（顺序不能反）。
    # ---------------------------------------------------------------------
    find_program(ICX NAMES icpx icpx.exe)
    if(NOT ICX)
        message(FATAL_ERROR
            "找不到 icpx。SYCL 后端需要先执行:\n"
            "  call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat\"\n"
            "  call \"C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat\"")
    endif()

    set(NOVA_SYCL_SOURCES
        code/device/sycl/device.cpp
        src/kernel/sycl/q4k_gemv.cpp
        src/kernel/sycl/q4k_gemv4.cpp
        src/kernel/sycl/quant.cpp
        src/kernel/sycl/norm.cpp
        src/kernel/sycl/rope.cpp
        src/kernel/sycl/elem.cpp
        src/kernel/sycl/attn.cpp
        code/model/sycl/runner.cpp
        src/loader/gguf.cpp
        tests/sycl_e2e.cpp
    )
    # 只编**仓库里真实存在**的源：mla/moe 是 DeepSeek-V2-Lite 的算子，可能还没同步到
    # 某些工作副本（踩过：目标机上报 "no such file or directory: mla.cpp"）。缺了就明说。
    set(_sycl_present "")
    foreach(_src IN LISTS NOVA_SYCL_SOURCES)
        if(EXISTS "${CMAKE_SOURCE_DIR}/${_src}")
            list(APPEND _sycl_present "${CMAKE_SOURCE_DIR}/${_src}")
        else()
            message(STATUS "SYCL: 跳过不存在的源 ${_src}")
        endif()
    endforeach()
    set(NOVA_SYCL_SOURCES "${_sycl_present}")

    # MSBuild 执行自定义命令时会剥掉 Intel 的 LIB 路径，链接器就找不到 libmmd.lib
    # （踩过：LNK1104）。所以这里显式把 oneAPI 的 lib 目录加上。
    get_filename_component(_icx_bin "${ICX}" DIRECTORY)
    get_filename_component(_oneapi_lib "${_icx_bin}/../lib" ABSOLUTE)
    message(STATUS "SYCL: oneAPI lib = ${_oneapi_lib}")

    set(NOVA_SYCL_OUT "${CMAKE_BINARY_DIR}/nova_sycl.exe")
    add_custom_command(
        OUTPUT "${NOVA_SYCL_OUT}"
        COMMAND "${ICX}"
                -fsycl -O3 -std=c++17 -fsycl-device-code-split=per_kernel
                -DNOVA_SYCL_MODEL_${NOVA_MODEL}
                -DNOVA_SYCL_SG=16 -DNOVA_SYCL_GEMV_ROWS=8 -DNOVA_SYCL_DP4A=0
                -I${CMAKE_SOURCE_DIR}/include
                -I${CMAKE_SOURCE_DIR}/code
                -I${CMAKE_SOURCE_DIR}/code/device/sycl
                -I${CMAKE_SOURCE_DIR}/code/model/sycl
                -I${CMAKE_SOURCE_DIR}/src/kernel/sycl
                ${NOVA_SYCL_SOURCES}
                -o "${NOVA_SYCL_OUT}"
        DEPENDS ${NOVA_SYCL_SOURCES} ${CMAKE_SOURCE_DIR}/CMakeLists.txt
        COMMENT "icpx: 一条命令编 SYCL 端到端（等价于 build_sycl_e2e.bat）"
        VERBATIM)
    add_custom_target(nova_sycl ALL DEPENDS "${NOVA_SYCL_OUT}")

elseif(NOVA_BACKEND_U STREQUAL "HOST")
    message(STATUS "HOST 模式：只构建 CPU 垫片对拍（不需要 GPU / oneAPI）")
else()
    message(FATAL_ERROR "未知 NOVA_BACKEND=${NOVA_BACKEND}（可选 CUDA / SYCL / HOST）")
endif()

# ---------------------------------------------------------------------------
#  bench：微基准 + CPU 垫片对拍
#    tests/host_sycl 是一套假 SYCL 头：真内核在 CPU 上跑，用来对 double 参考。
#    它不需要显卡、不需要 oneAPI —— 这就是 CI 每次 push 的数值闸门。
# ---------------------------------------------------------------------------
if(NOVA_BUILD_BENCH AND NOT NOVA_BACKEND_U STREQUAL "CUDA" OR NOVA_BACKEND_U STREQUAL "HOST")
    add_executable(nova_host_check
        tests/sycl_gemv_host_check.cpp
        # 注意：q4k_gemv.cpp / q4k_gemv_soa.cpp 由 tests 里的 #include 拉进来（同一 TU 才能
        # 调到匿名命名空间的 launcher），**不能**再单独列在这里，否则 LNK2005 重复定义。
        src/kernel/sycl/quant.cpp
        src/kernel/sycl/norm.cpp
        src/kernel/sycl/elem.cpp
        src/loader/gguf.cpp
    )
    target_include_directories(nova_host_check PRIVATE
        tests/host_sycl include code code/device/sycl code/model/sycl src/kernel/sycl)
    target_compile_definitions(nova_host_check PRIVATE
        ${NOVA_MODEL_MACRO} ${NOVA_SYCL_MODEL_MACRO} NOVA_SYCL_SG=16 NOVA_SYCL_GEMV_ROWS=8 NOVA_SYCL_DP4A=0)
    if(MSVC)
        target_compile_options(nova_host_check PRIVATE /utf-8 /O2 /wd4068 /wd5030)
    else()
        target_compile_options(nova_host_check PRIVATE -O2 -Wno-unknown-pragmas)
        target_compile_options(nova_host_check PRIVATE -include cstring)
    endif()
endif()

# ---------------------------------------------------------------------------
#  测试注册：ctest 直接跑数值闸门
# ---------------------------------------------------------------------------
enable_testing()
if(TARGET nova_host_check)
    add_test(NAME gemv_numeric COMMAND nova_host_check ${CMAKE_SOURCE_DIR}/build/q4k_15b.gguf)
    set_tests_properties(gemv_numeric PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()
