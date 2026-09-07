# iian_server: OpenAI-compatible HTTP server on top of the engine (cpp-httplib + nlohmann/json).
# Built as an OBJECT library; the `iian` CLI executable links it together with the engine.
if(NOT DEFINED IIAN_ROOT_DIR)
    set(IIAN_ROOT_DIR ${CMAKE_SOURCE_DIR})
endif()

add_library(iian_server OBJECT
    ${IIAN_ROOT_DIR}/third_party/cpp-httplib/httplib.cpp
    server_context.cpp
    openai.cpp
    generation.cpp
    metrics.cpp
    http_server.cpp
    tool_calls.cpp
    model_registry.cpp
)

target_include_directories(iian_server
    PUBLIC
        ${IIAN_ROOT_DIR}/include
        ${IIAN_ROOT_DIR}/third_party
        ${IIAN_ROOT_DIR}/src
)
target_link_libraries(iian_server PUBLIC iian)
target_compile_features(iian_server PUBLIC cxx_std_20)
set_target_properties(iian_server PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TLS (needed for `iian pull` from Hugging Face): enabled when OpenSSL is available.
find_package(OpenSSL QUIET)
if(OPENSSL_FOUND)
    target_compile_definitions(iian_server PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(iian_server PUBLIC OpenSSL::SSL OpenSSL::Crypto)
    message(STATUS "iian_server: OpenSSL ${OPENSSL_VERSION} found, TLS enabled")
else()
    message(STATUS "iian_server: OpenSSL not found, TLS disabled (iian pull falls back to curl)")
endif()

find_package(Threads REQUIRED)
target_link_libraries(iian_server PUBLIC Threads::Threads)

if(NOT MSVC)
    target_compile_options(iian_server PRIVATE -Wall -Wextra)
    # third-party code is not held to our warning level
    set_source_files_properties(${IIAN_ROOT_DIR}/third_party/cpp-httplib/httplib.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()
