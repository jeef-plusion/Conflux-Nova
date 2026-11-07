cmake_minimum_required(VERSION 3.24)
project(ConfluxNovaNative CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(NOVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(WV2 ${CMAKE_CURRENT_SOURCE_DIR}/packages/wv2/build/native)

# ---- 引擎 ----
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

# ---- WebView2 外壳 ----
add_executable(ConfluxNova WIN32
    src/main.cpp
    src/ws_server.cpp
    src/engine_host.cpp)
target_include_directories(ConfluxNova PRIVATE src ${WV2}/include)
target_compile_options(ConfluxNova PRIVATE /utf-8)
target_link_libraries(ConfluxNova PRIVATE
    nova_engine ${WV2}/x64/WebView2LoaderStatic.lib
    dwmapi ole32 shell32 ws2_32 bcrypt)
target_link_options(ConfluxNova PRIVATE /SUBSYSTEM:WINDOWS)
set_property(TARGET ConfluxNova PROPERTY CUDA_RUNTIME_LIBRARY Static)

# web/ 随 exe 输出
add_custom_command(TARGET ConfluxNova POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/web $<TARGET_FILE_DIR:ConfluxNova>/web)

# ---- 无 GUI 验证 ----
add_executable(ConfluxHeadless src/headless_main.cpp src/engine_host.cpp)
target_include_directories(ConfluxHeadless PRIVATE src)
target_compile_options(ConfluxHeadless PRIVATE /utf-8)
target_link_libraries(ConfluxHeadless PRIVATE nova_engine d3d11 dxgi dwmapi)
set_property(TARGET ConfluxHeadless PROPERTY CUDA_RUNTIME_LIBRARY Static)