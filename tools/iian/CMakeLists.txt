# The `iian` command-line binary. The CMake target is `iian-cli` because `iian` is the engine library
# target; the produced executable is still named `iian`.
add_executable(iian-cli main.cpp)
set_target_properties(iian-cli PROPERTIES OUTPUT_NAME iian)
target_link_libraries(iian-cli PRIVATE iian iian_server iian_cli)
if(NOT MSVC)
    target_compile_options(iian-cli PRIVATE -Wall -Wextra)
endif()
