file(GLOB IIAN_ARCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/iian/arch/*.cpp)

set(IIAN_CORE_SOURCES
    iian/version.cpp
    iian/log.cpp
    iian/gguf.cpp
    iian/model_loader.cpp
    iian/graph.cpp
    iian/kv_cache.cpp
    iian/sampler.cpp
    iian/scheduler.cpp
    iian/engine.cpp
    iian/paged_attn.cpp
    iian/draft.cpp
    iian/grammar/grammar.cpp
    iian/grammar/json_schema.cpp
)

# Shared library so that architecture self-registration (static initializers in arch/*.cpp)
# is never dead-stripped by the linker.
add_library(iian SHARED ${IIAN_CORE_SOURCES} ${IIAN_ARCH_SOURCES})
target_include_directories(iian PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(iian PUBLIC ggml iian_third_party)
if(GGML_NATIVE AND NOT MSVC)
    target_compile_options(iian PRIVATE -march=native)   # SIMD paths in the paged-attention kernel
endif()
if(IIAN_WARNINGS)
    target_compile_options(iian PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/iian/tokenizer/CMakeLists.txt)
    add_subdirectory(iian/tokenizer)
    target_link_libraries(iian PUBLIC iian_tokenizer)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/iian/chat/CMakeLists.txt)
    add_subdirectory(iian/chat)
    target_link_libraries(iian PUBLIC iian_chat)
endif()

if(IIAN_BUILD_TOOLS)
    add_subdirectory(iian/server)
    add_subdirectory(iian/cli)
endif()
