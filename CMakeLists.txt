cmake_minimum_required(VERSION 3.16)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(iian VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(IIAN_BUILD_TESTS  "Build unit tests"            ON)
option(IIAN_BUILD_TOOLS  "Build CLI / server binaries" ON)
option(IIAN_CURL         "Enable libcurl (model pull)" ON)
option(IIAN_WARNINGS     "Enable strict warnings"      ON)

# ---- ggml (vendored) -------------------------------------------------------
# All ggml backend options (GGML_CUDA, GGML_METAL, GGML_VULKAN, GGML_HIP, ...)
# pass straight through: cmake -DGGML_CUDA=ON ...
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "")
endif()
# llama.cpp enables tinyBLAS (llamafile sgemm) by default; keep the same CPU matmul kernels so that
# greedy outputs match llama.cpp bit-for-bit and prompt processing gets the same speed-up.
if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ON CACHE BOOL "")
endif()
add_subdirectory(third_party/ggml)

# ---- third party header-only -----------------------------------------------
add_library(iian_third_party INTERFACE)
target_include_directories(iian_third_party INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_subdirectory(src)
if(IIAN_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if(IIAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
