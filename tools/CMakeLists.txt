add_executable(iian-smoke smoke.cpp)
target_link_libraries(iian-smoke PRIVATE iian)
add_subdirectory(iian)
