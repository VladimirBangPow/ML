# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# If you want, you could set a separate project name just for tests:
project(MyProjectTests C)

# We'll build a separate test executable for our linear regression test.
add_executable(test_lr
    test_lr.c
    # If you have other .c test files, list them here.
)

# We assume 'my_lib' is the library from top-level CMake. 
# So we link it:
target_link_libraries(test_lr
    PRIVATE
    my_lib         # from the top-level
    # DataFrame    # if needed
)

# If the test code needs to include headers:
target_include_directories(test_lr
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Register test with CTest, so `ctest` can run it
add_test(NAME test_lr COMMAND test_lr)
