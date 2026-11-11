# CMake generated Testfile for 
# Source directory: C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2
# Build directory: C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build-cmake/HOST
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[gemv_numeric]=] "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build-cmake/HOST/Debug/nova_host_check.exe" "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build/q4k_15b.gguf")
  set_tests_properties([=[gemv_numeric]=] PROPERTIES  SKIP_RETURN_CODE "77" TIMEOUT "600" _BACKTRACE_TRIPLES "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;213;add_test;C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[gemv_numeric]=] "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build-cmake/HOST/Release/nova_host_check.exe" "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build/q4k_15b.gguf")
  set_tests_properties([=[gemv_numeric]=] PROPERTIES  SKIP_RETURN_CODE "77" TIMEOUT "600" _BACKTRACE_TRIPLES "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;213;add_test;C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[gemv_numeric]=] "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build-cmake/HOST/MinSizeRel/nova_host_check.exe" "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build/q4k_15b.gguf")
  set_tests_properties([=[gemv_numeric]=] PROPERTIES  SKIP_RETURN_CODE "77" TIMEOUT "600" _BACKTRACE_TRIPLES "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;213;add_test;C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[gemv_numeric]=] "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build-cmake/HOST/RelWithDebInfo/nova_host_check.exe" "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/build/q4k_15b.gguf")
  set_tests_properties([=[gemv_numeric]=] PROPERTIES  SKIP_RETURN_CODE "77" TIMEOUT "600" _BACKTRACE_TRIPLES "C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;213;add_test;C:/Users/Administrator/Documents/deepseek-harness/default-workspace/nova_v2/CMakeLists.txt;0;")
else()
  add_test([=[gemv_numeric]=] NOT_AVAILABLE)
endif()
