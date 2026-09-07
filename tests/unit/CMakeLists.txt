add_executable(test-block-pool test-block-pool.cpp)
target_link_libraries(test-block-pool PRIVATE iian)
add_test(NAME block-pool COMMAND test-block-pool)
