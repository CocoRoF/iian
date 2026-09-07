# iian_chat: Jinja chat-template engine (port of llama.cpp common/jinja, MIT) + ChatTemplate facade
add_library(iian_chat OBJECT
    chat_template.cpp
    jinja/caps.cpp
    jinja/lexer.cpp
    jinja/parser.cpp
    jinja/runtime.cpp
    jinja/string.cpp
    jinja/unicode.cpp
    jinja/value.cpp
)

# IIAN_ROOT_DIR defaults to the top-level source dir; the standalone test build (tests/chat/standalone)
# sets it explicitly because there CMAKE_SOURCE_DIR is not the project root.
if(NOT DEFINED IIAN_ROOT_DIR)
    set(IIAN_ROOT_DIR ${CMAKE_SOURCE_DIR})
endif()

target_include_directories(iian_chat
    PUBLIC
        ${IIAN_ROOT_DIR}/include
        ${IIAN_ROOT_DIR}/third_party
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(iian_chat PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(iian_chat PRIVATE -Wall -Wextra)
endif()
