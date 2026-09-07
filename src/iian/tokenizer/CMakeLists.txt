# iian tokenizer module (llama.cpp vocab port, standalone)
# Built as an OBJECT library; the umbrella library links it (target_link_libraries(iian PUBLIC iian_tokenizer)).

# Allow a standalone driver (tests/tokenizer/standalone) to point at the project root explicitly.
if(NOT DEFINED IIAN_SOURCE_ROOT)
    set(IIAN_SOURCE_ROOT ${CMAKE_SOURCE_DIR})
endif()

add_library(iian_tokenizer OBJECT
    vocab.cpp
    unicode.cpp
    unicode-data.cpp
    detokenizer.cpp
)

target_include_directories(iian_tokenizer
    PUBLIC
        ${IIAN_SOURCE_ROOT}/include
        ${IIAN_SOURCE_ROOT}/third_party
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(iian_tokenizer PUBLIC ggml)

target_compile_features(iian_tokenizer PUBLIC cxx_std_20)
set_target_properties(iian_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_compile_options(iian_tokenizer PRIVATE -Wall -Wextra)
endif()
