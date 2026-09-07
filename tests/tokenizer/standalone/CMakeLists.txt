# Standalone driver to build + test the tokenizer module without touching the project's own CMake files.
#   cmake -S tests/tokenizer/standalone -B build-tok -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-tok -j 8 && ctest --test-dir build-tok --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(iian_tokenizer_standalone LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(IIAN_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "")
endif()
add_subdirectory(${IIAN_SOURCE_ROOT}/third_party/ggml     ${CMAKE_BINARY_DIR}/third_party/ggml)
add_subdirectory(${IIAN_SOURCE_ROOT}/src/iian/tokenizer   ${CMAKE_BINARY_DIR}/src/iian/tokenizer)

enable_testing()
add_subdirectory(${IIAN_SOURCE_ROOT}/tests/tokenizer      ${CMAKE_BINARY_DIR}/tests/tokenizer)
