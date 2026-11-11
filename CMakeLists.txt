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
# HOST 模式不建 nova_core：src/model.cpp 会拉进 cuda_runtime.h，HOST 机器没有 CUDA。
if(NOT NOVA_BACKEND_U STREQUAL "HOST")
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
    # icpx 必须在 project() 之前就成为编译器 —— 所以这里**不能** set(CMAKE_CXX_COMPILER)，
    # 那已经太晚（project() 早就选好了）。正确做法：先 call setvars.bat（icpx 进 PATH），
    # 再跑 cmake；这里只做校验，选错编译器时给出可执行的提示。
    if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
        message(FATAL_ERROR
            "SYCL 后端要求用 icpx 作为 C++ 编译器，当前是: ${CMAKE_CXX_COMPILER}\n"
            "先执行:  call \"C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat\"\n"
            "再执行:  cmake -B build-cmake\\SYCL -DNOVA_BACKEND=SYCL ...")
    endif()

    # [core] src/kernel/sycl/*.cpp + code/device/sycl/device.cpp
    add_library(nova_kernels_sycl STATIC
        code/device/sycl/device.cpp
        src/kernel/sycl/q4k_gemv.cpp
        src/kernel/sycl/q4k_gemv4.cpp
        src/kernel/sycl/quant.cpp
        src/kernel/sycl/norm.cpp
        src/kernel/sycl/rope.cpp
        src/kernel/sycl/elem.cpp
        src/kernel/sycl/attn.cpp
        src/kernel/sycl/mla.cpp
        src/kernel/sycl/moe.cpp
    )
    target_include_directories(nova_kernels_sycl PUBLIC
        include code code/device/sycl code/model/sycl src/kernel/sycl)
    target_compile_options(nova_kernels_sycl PUBLIC -fsycl -O3)
    target_compile_definitions(nova_kernels_sycl PUBLIC ${NOVA_MODEL_MACRO} ${NOVA_SYCL_MODEL_MACRO})
    target_link_libraries(nova_engine PUBLIC nova_kernels_sycl)

    # [apps] tests/sycl_e2e.cpp + code/model/sycl/runner.cpp
    if(NOVA_BUILD_APPS)
        add_executable(nova_sycl tests/sycl_e2e.cpp code/model/sycl/runner.cpp)
        target_include_directories(nova_sycl PRIVATE
            include code code/device/sycl code/model/sycl src/kernel/sycl)
        target_compile_options(nova_sycl PUBLIC -fsycl -O3)
        target_link_libraries(nova_sycl PRIVATE nova_engine)
    endif()

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
