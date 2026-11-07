cmake_minimum_required(VERSION 3.24)
project(ConfluxNovaNative CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
# 全局静态 CRT：保证所有目标一致（否则 LNK2038）
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CUDA_ARCHITECTURES 61)

set(NOVA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)   # 引擎仓库根目录

include(FetchContent)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.9)
FetchContent_MakeAvailable(imgui)

# 引擎源码（与 build_core.bat 一致）
add_library(nova_engine STATIC
    ${NOVA_ROOT}/code/kernel/1080ti/gemv_q4k.cu
    ${NOVA_ROOT}/code/kernel/1080ti/attn.cu
    ${NOVA_ROOT}/src/loader/gguf.cpp
    ${NOVA_ROOT}/src/loader/validate.cpp
    ${NOVA_ROOT}/src/model.cpp)
target_include_directories(nova_engine PUBLIC ${NOVA_ROOT}/include ${NOVA_ROOT}/code)
target_compile_definitions(nova_engine PUBLIC NOVA_MODEL_QWEN25_7B)
set_property(TARGET nova_engine PROPERTY CUDA_ARCHITECTURES 61)
target_compile_options(nova_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

add_executable(ConfluxNova WIN32
    src/main.cpp
    src/engine_host.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp)

target_compile_options(ConfluxNova PRIVATE /utf-8)
target_include_directories(ConfluxNova PRIVATE src ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(ConfluxNova PRIVATE nova_engine d3d11 dxgi dwmapi)
target_link_options(ConfluxNova PRIVATE /SUBSYSTEM:WINDOWS)
# 单文件产物：静态 CRT + 静态 CUDA 运行时
set_property(TARGET ConfluxNova PROPERTY CUDA_RUNTIME_LIBRARY Static)
# 无 GUI 验证程序（同样链接引擎）
add_executable(ConfluxHeadless src/headless_main.cpp src/engine_host.cpp)
target_include_directories(ConfluxHeadless PRIVATE src)
target_compile_options(ConfluxHeadless PRIVATE /utf-8)
target_link_libraries(ConfluxHeadless PRIVATE nova_engine d3d11 dxgi dwmapi)
set_property(TARGET ConfluxHeadless PROPERTY CUDA_RUNTIME_LIBRARY Static)
