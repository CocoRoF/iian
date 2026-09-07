# Standalone driver: builds the engine + server + CLI without touching the project's own CMake files.
#   export PATH="$HOME/.local/bin:$PATH"
#   cmake -S tests/server/standalone -B build-srv -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-srv -j 8 && python3 tests/server/test-server.py build-srv/tools/iian/iian models/SmolLM2-135M-Instruct-F16.gguf
cmake_minimum_required(VERSION 3.16)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(iian_srv LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(IIAN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
# the tokenizer/chat sub-projects locate the repo root through these when CMAKE_SOURCE_DIR is not the root
set(IIAN_ROOT_DIR    ${IIAN_ROOT})
set(IIAN_SOURCE_ROOT ${IIAN_ROOT})
option(IIAN_WARNINGS "Enable strict warnings" ON)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "")
endif()
add_subdirectory(${IIAN_ROOT}/third_party/ggml ${CMAKE_BINARY_DIR}/third_party/ggml)

add_library(iian_third_party INTERFACE)
target_include_directories(iian_third_party INTERFACE ${IIAN_ROOT}/third_party)

add_subdirectory(${IIAN_ROOT}/src             ${CMAKE_BINARY_DIR}/src)
add_subdirectory(${IIAN_ROOT}/src/iian/server ${CMAKE_BINARY_DIR}/src/iian/server)
add_subdirectory(${IIAN_ROOT}/src/iian/cli    ${CMAKE_BINARY_DIR}/src/iian/cli)
add_subdirectory(${IIAN_ROOT}/tools/iian      ${CMAKE_BINARY_DIR}/tools/iian)

enable_testing()
add_subdirectory(${IIAN_ROOT}/tests/server    ${CMAKE_BINARY_DIR}/tests/server)
