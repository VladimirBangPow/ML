cmake_minimum_required(VERSION 3.10)
project(ML C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Build our executable
add_executable(ml_tests
    src/main.c
    src/tensor.c
    src/lr.c
    # any other .c files
)

# Make sure the compiler and linker use ASan
target_compile_options(ml_tests PRIVATE
    -fsanitize=address
    -fno-omit-frame-pointer
    -g
)

# On CMake 3.13 or later, you can set link options as well:
target_link_options(ml_tests PRIVATE
    -fsanitize=address
)

target_include_directories(ml_tests
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# DataFrame subproject
add_subdirectory(
    "../DataFrame"
    "${CMAKE_CURRENT_BINARY_DIR}/DataFrame_build"
)
target_link_libraries(ml_tests PRIVATE DataFrame)

# Register test
add_test(NAME ml_test_suite COMMAND ml_tests)
