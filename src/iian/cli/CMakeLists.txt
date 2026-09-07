# iian_cli: subcommand implementations for the `iian` binary (serve/ps/stop/logs/status/run/complete/
# pull/ls/rm/info/bench/version). OBJECT library linked by tools/iian.
if(NOT DEFINED IIAN_ROOT_DIR)
    set(IIAN_ROOT_DIR ${CMAKE_SOURCE_DIR})
endif()

file(GLOB IIAN_CLI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(iian_cli OBJECT ${IIAN_CLI_SOURCES})

target_include_directories(iian_cli
    PUBLIC
        ${IIAN_ROOT_DIR}/include
        ${IIAN_ROOT_DIR}/third_party
        ${IIAN_ROOT_DIR}/src
)
target_link_libraries(iian_cli PUBLIC iian iian_server)
target_compile_features(iian_cli PUBLIC cxx_std_20)
set_target_properties(iian_cli PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(iian_cli PRIVATE -Wall -Wextra)
endif()
