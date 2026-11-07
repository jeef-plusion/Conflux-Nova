cmake_minimum_required(VERSION 3.24)
project(ConfluxNovaServer CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(NOVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(nova_engine STATIC
    ${NOVA_ROOT}/code/kernel/1080ti/gemv_q4k.cu
    ${NOVA_ROOT}/code/kernel/1080ti/attn.cu
    ${NOVA_ROOT}/src/loader/gguf.cpp
    ${NOVA_ROOT}/src/loader/validate.cpp
    ${NOVA_ROOT}/src/model.cpp)
target_include_directories(nova_engine PUBLIC ${NOVA_ROOT}/include ${NOVA_ROOT}/code)
target_compile_definitions(nova_engine PUBLIC NOVA_MODEL_QWEN25_7B)
target_compile_options(nova_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

add_executable(ConfluxNova src/server.cpp src/engine_host.cpp)
target_include_directories(ConfluxNova PRIVATE src)
target_compile_options(ConfluxNova PRIVATE /utf-8)
target_link_libraries(ConfluxNova PRIVATE nova_engine ws2_32 bcrypt)
set_property(TARGET ConfluxNova PROPERTY CUDA_RUNTIME_LIBRARY Static)

# 把 web/ 拷到 exe 旁边，方便直接运行
add_custom_command(TARGET ConfluxNova POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/web $<TARGET_FILE_DIR:ConfluxNova>/web)