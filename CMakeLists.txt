cmake_minimum_required(VERSION 3.20)
project(nova_v2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

# 一次只编一个模型：-DNOVA_MODEL=NOVA_MODEL_QWEN25_7B / NOVA_MODEL_R1_QWEN7B / ...
set(NOVA_MODEL "NOVA_MODEL_QWEN25_7B" CACHE STRING "which model this binary is welded to")

set(NOVA_SRC
    src/kernel/q4k_gemv.cu
    src/kernel/norm.cu
    src/kernel/rope.cu
    src/kernel/attn.cu
    src/loader/gguf.cpp
    src/loader/validate.cpp
    src/main.cpp)

add_executable(nova ${NOVA_SRC})
target_include_directories(nova PRIVATE include)
target_compile_definitions(nova PRIVATE ${NOVA_MODEL})
target_compile_options(nova PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

# 测试
add_executable(selftest_q4k tests/selftest_q4k.cpp)
target_include_directories(selftest_q4k PRIVATE include)

add_executable(gpu_gemv tests/gpu_gemv.cu src/kernel/q4k_gemv.cu)
target_include_directories(gpu_gemv PRIVATE include)
target_compile_definitions(gpu_gemv PRIVATE ${NOVA_MODEL})

add_executable(gpu_misc tests/gpu_misc.cu src/kernel/rope.cu src/kernel/attn.cu)
target_include_directories(gpu_misc PRIVATE include)
target_compile_definitions(gpu_misc PRIVATE ${NOVA_MODEL})
