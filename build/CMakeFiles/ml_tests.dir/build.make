# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hassanamad/Projects/ML

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hassanamad/Projects/ML/build

# Include any dependencies generated for this target.
include CMakeFiles/ml_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ml_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ml_tests.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ml_tests.dir/flags.make

CMakeFiles/ml_tests.dir/codegen:
.PHONY : CMakeFiles/ml_tests.dir/codegen

CMakeFiles/ml_tests.dir/src/main.c.o: CMakeFiles/ml_tests.dir/flags.make
CMakeFiles/ml_tests.dir/src/main.c.o: /Users/hassanamad/Projects/ML/src/main.c
CMakeFiles/ml_tests.dir/src/main.c.o: CMakeFiles/ml_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/ml_tests.dir/src/main.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ml_tests.dir/src/main.c.o -MF CMakeFiles/ml_tests.dir/src/main.c.o.d -o CMakeFiles/ml_tests.dir/src/main.c.o -c /Users/hassanamad/Projects/ML/src/main.c

CMakeFiles/ml_tests.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/ml_tests.dir/src/main.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/ML/src/main.c > CMakeFiles/ml_tests.dir/src/main.c.i

CMakeFiles/ml_tests.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/ml_tests.dir/src/main.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/ML/src/main.c -o CMakeFiles/ml_tests.dir/src/main.c.s

CMakeFiles/ml_tests.dir/src/tensor.c.o: CMakeFiles/ml_tests.dir/flags.make
CMakeFiles/ml_tests.dir/src/tensor.c.o: /Users/hassanamad/Projects/ML/src/tensor.c
CMakeFiles/ml_tests.dir/src/tensor.c.o: CMakeFiles/ml_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/ml_tests.dir/src/tensor.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ml_tests.dir/src/tensor.c.o -MF CMakeFiles/ml_tests.dir/src/tensor.c.o.d -o CMakeFiles/ml_tests.dir/src/tensor.c.o -c /Users/hassanamad/Projects/ML/src/tensor.c

CMakeFiles/ml_tests.dir/src/tensor.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/ml_tests.dir/src/tensor.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/ML/src/tensor.c > CMakeFiles/ml_tests.dir/src/tensor.c.i

CMakeFiles/ml_tests.dir/src/tensor.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/ml_tests.dir/src/tensor.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/ML/src/tensor.c -o CMakeFiles/ml_tests.dir/src/tensor.c.s

CMakeFiles/ml_tests.dir/src/lr.c.o: CMakeFiles/ml_tests.dir/flags.make
CMakeFiles/ml_tests.dir/src/lr.c.o: /Users/hassanamad/Projects/ML/src/lr.c
CMakeFiles/ml_tests.dir/src/lr.c.o: CMakeFiles/ml_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/ml_tests.dir/src/lr.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ml_tests.dir/src/lr.c.o -MF CMakeFiles/ml_tests.dir/src/lr.c.o.d -o CMakeFiles/ml_tests.dir/src/lr.c.o -c /Users/hassanamad/Projects/ML/src/lr.c

CMakeFiles/ml_tests.dir/src/lr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/ml_tests.dir/src/lr.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/ML/src/lr.c > CMakeFiles/ml_tests.dir/src/lr.c.i

CMakeFiles/ml_tests.dir/src/lr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/ml_tests.dir/src/lr.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/ML/src/lr.c -o CMakeFiles/ml_tests.dir/src/lr.c.s

CMakeFiles/ml_tests.dir/tests/test_lr.c.o: CMakeFiles/ml_tests.dir/flags.make
CMakeFiles/ml_tests.dir/tests/test_lr.c.o: /Users/hassanamad/Projects/ML/tests/test_lr.c
CMakeFiles/ml_tests.dir/tests/test_lr.c.o: CMakeFiles/ml_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/ml_tests.dir/tests/test_lr.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ml_tests.dir/tests/test_lr.c.o -MF CMakeFiles/ml_tests.dir/tests/test_lr.c.o.d -o CMakeFiles/ml_tests.dir/tests/test_lr.c.o -c /Users/hassanamad/Projects/ML/tests/test_lr.c

CMakeFiles/ml_tests.dir/tests/test_lr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/ml_tests.dir/tests/test_lr.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/ML/tests/test_lr.c > CMakeFiles/ml_tests.dir/tests/test_lr.c.i

CMakeFiles/ml_tests.dir/tests/test_lr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/ml_tests.dir/tests/test_lr.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/ML/tests/test_lr.c -o CMakeFiles/ml_tests.dir/tests/test_lr.c.s

# Object files for target ml_tests
ml_tests_OBJECTS = \
"CMakeFiles/ml_tests.dir/src/main.c.o" \
"CMakeFiles/ml_tests.dir/src/tensor.c.o" \
"CMakeFiles/ml_tests.dir/src/lr.c.o" \
"CMakeFiles/ml_tests.dir/tests/test_lr.c.o"

# External object files for target ml_tests
ml_tests_EXTERNAL_OBJECTS =

ml_tests: CMakeFiles/ml_tests.dir/src/main.c.o
ml_tests: CMakeFiles/ml_tests.dir/src/tensor.c.o
ml_tests: CMakeFiles/ml_tests.dir/src/lr.c.o
ml_tests: CMakeFiles/ml_tests.dir/tests/test_lr.c.o
ml_tests: CMakeFiles/ml_tests.dir/build.make
ml_tests: DataFrame_build/libDataFrame.a
ml_tests: DataFrame_build/DataStructures_build/libMyDataStructures.a
ml_tests: CMakeFiles/ml_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C executable ml_tests"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ml_tests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ml_tests.dir/build: ml_tests
.PHONY : CMakeFiles/ml_tests.dir/build

CMakeFiles/ml_tests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ml_tests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ml_tests.dir/clean

CMakeFiles/ml_tests.dir/depend:
	cd /Users/hassanamad/Projects/ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hassanamad/Projects/ML /Users/hassanamad/Projects/ML /Users/hassanamad/Projects/ML/build /Users/hassanamad/Projects/ML/build /Users/hassanamad/Projects/ML/build/CMakeFiles/ml_tests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ml_tests.dir/depend

