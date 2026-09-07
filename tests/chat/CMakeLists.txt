# Chat template tests (expects targets `iian_chat` and `ggml` to exist)
if(NOT DEFINED IIAN_ROOT_DIR)
    set(IIAN_ROOT_DIR ${CMAKE_SOURCE_DIR})
endif()

add_executable(test-chat-template test-chat-template.cpp)
target_link_libraries(test-chat-template PRIVATE iian_chat ggml)
target_include_directories(test-chat-template PRIVATE
    ${IIAN_ROOT_DIR}/include
    ${IIAN_ROOT_DIR}/third_party
    ${IIAN_ROOT_DIR}/src/iian/chat          # jinja engine internals (engine-level tests)
)
target_compile_features(test-chat-template PRIVATE cxx_std_20)
target_compile_definitions(test-chat-template PRIVATE
    IIAN_TEST_CHAT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    IIAN_MODELS_DIR="${IIAN_ROOT_DIR}/models"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test-chat-template PRIVATE -Wall -Wextra)
endif()

add_test(NAME test-chat-template COMMAND test-chat-template)
