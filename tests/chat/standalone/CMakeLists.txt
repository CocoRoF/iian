# Standalone build of the chat-template module + its tests, without touching the top-level CMake files:
#   cmake -S tests/chat/standalone -B build-chat -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-chat && ctest --test-dir build-chat --output-on-failure
cmake_minimum_required(VERSION 3.16)
project(iian_chat_standalone LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

get_filename_component(IIAN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# ggml (only needed for the gguf reader used by the tests)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "")
endif()
add_subdirectory(${IIAN_ROOT_DIR}/third_party/ggml ${CMAKE_BINARY_DIR}/third_party/ggml)

add_subdirectory(${IIAN_ROOT_DIR}/src/iian/chat ${CMAKE_BINARY_DIR}/src/iian/chat)

enable_testing()
add_subdirectory(${IIAN_ROOT_DIR}/tests/chat ${CMAKE_BINARY_DIR}/tests/chat)
