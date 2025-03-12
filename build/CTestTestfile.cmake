# CMake generated Testfile for 
# Source directory: /Users/hassanamad/Projects/ML
# Build directory: /Users/hassanamad/Projects/ML/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ml_test_suite "/Users/hassanamad/Projects/ML/build/ml_tests")
set_tests_properties(ml_test_suite PROPERTIES  _BACKTRACE_TRIPLES "/Users/hassanamad/Projects/ML/CMakeLists.txt;45;add_test;/Users/hassanamad/Projects/ML/CMakeLists.txt;0;")
subdirs("DataFrame_build")
