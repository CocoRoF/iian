add_test(NAME server-e2e COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test-server.py $<TARGET_FILE:iian-cli> ${CMAKE_SOURCE_DIR}/models/SmolLM2-135M-Instruct-F16.gguf)
set_tests_properties(server-e2e PROPERTIES TIMEOUT 900)
