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

# Utility rule file for NightlyCoverage.

# Include any custom commands dependencies for this target.
include DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/compiler_depend.make

# Include the progress variables for this target.
include DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/progress.make

DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage:
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build/DataStructures_build/tests && /opt/homebrew/bin/ctest -D NightlyCoverage

DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/codegen:
.PHONY : DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/codegen

NightlyCoverage: DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage
NightlyCoverage: DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/build.make
.PHONY : NightlyCoverage

# Rule to build all files generated by this target.
DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/build: NightlyCoverage
.PHONY : DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/build

DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/clean:
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build/DataStructures_build/tests && $(CMAKE_COMMAND) -P CMakeFiles/NightlyCoverage.dir/cmake_clean.cmake
.PHONY : DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/clean

DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/depend:
	cd /Users/hassanamad/Projects/ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hassanamad/Projects/ML /Users/hassanamad/Projects/DataStructures/tests /Users/hassanamad/Projects/ML/build /Users/hassanamad/Projects/ML/build/DataFrame_build/DataStructures_build/tests /Users/hassanamad/Projects/ML/build/DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : DataFrame_build/DataStructures_build/tests/CMakeFiles/NightlyCoverage.dir/depend

