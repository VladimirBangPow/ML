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
include DataFrame_build/CMakeFiles/DataFrame.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.make

# Include the progress variables for this target.
include DataFrame_build/CMakeFiles/DataFrame.dir/progress.make

# Include the compile flags for this target's objects.
include DataFrame_build/CMakeFiles/DataFrame.dir/flags.make

DataFrame_build/CMakeFiles/DataFrame.dir/codegen:
.PHONY : DataFrame_build/CMakeFiles/DataFrame.dir/codegen

DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o: /Users/hassanamad/Projects/DataFrame/src/aggregate.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o -MF CMakeFiles/DataFrame.dir/src/aggregate.c.o.d -o CMakeFiles/DataFrame.dir/src/aggregate.c.o -c /Users/hassanamad/Projects/DataFrame/src/aggregate.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/aggregate.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/aggregate.c > CMakeFiles/DataFrame.dir/src/aggregate.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/aggregate.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/aggregate.c -o CMakeFiles/DataFrame.dir/src/aggregate.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o: /Users/hassanamad/Projects/DataFrame/src/combine.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o -MF CMakeFiles/DataFrame.dir/src/combine.c.o.d -o CMakeFiles/DataFrame.dir/src/combine.c.o -c /Users/hassanamad/Projects/DataFrame/src/combine.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/combine.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/combine.c > CMakeFiles/DataFrame.dir/src/combine.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/combine.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/combine.c -o CMakeFiles/DataFrame.dir/src/combine.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o: /Users/hassanamad/Projects/DataFrame/src/core.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o -MF CMakeFiles/DataFrame.dir/src/core.c.o.d -o CMakeFiles/DataFrame.dir/src/core.c.o -c /Users/hassanamad/Projects/DataFrame/src/core.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/core.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/core.c > CMakeFiles/DataFrame.dir/src/core.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/core.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/core.c -o CMakeFiles/DataFrame.dir/src/core.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o: /Users/hassanamad/Projects/DataFrame/src/date.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o -MF CMakeFiles/DataFrame.dir/src/date.c.o.d -o CMakeFiles/DataFrame.dir/src/date.c.o -c /Users/hassanamad/Projects/DataFrame/src/date.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/date.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/date.c > CMakeFiles/DataFrame.dir/src/date.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/date.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/date.c -o CMakeFiles/DataFrame.dir/src/date.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o: /Users/hassanamad/Projects/DataFrame/src/indexing.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o -MF CMakeFiles/DataFrame.dir/src/indexing.c.o.d -o CMakeFiles/DataFrame.dir/src/indexing.c.o -c /Users/hassanamad/Projects/DataFrame/src/indexing.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/indexing.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/indexing.c > CMakeFiles/DataFrame.dir/src/indexing.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/indexing.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/indexing.c -o CMakeFiles/DataFrame.dir/src/indexing.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o: /Users/hassanamad/Projects/DataFrame/src/io.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o -MF CMakeFiles/DataFrame.dir/src/io.c.o.d -o CMakeFiles/DataFrame.dir/src/io.c.o -c /Users/hassanamad/Projects/DataFrame/src/io.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/io.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/io.c > CMakeFiles/DataFrame.dir/src/io.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/io.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/io.c -o CMakeFiles/DataFrame.dir/src/io.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o: /Users/hassanamad/Projects/DataFrame/src/plot.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o -MF CMakeFiles/DataFrame.dir/src/plot.c.o.d -o CMakeFiles/DataFrame.dir/src/plot.c.o -c /Users/hassanamad/Projects/DataFrame/src/plot.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/plot.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/plot.c > CMakeFiles/DataFrame.dir/src/plot.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/plot.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/plot.c -o CMakeFiles/DataFrame.dir/src/plot.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o: /Users/hassanamad/Projects/DataFrame/src/print.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o -MF CMakeFiles/DataFrame.dir/src/print.c.o.d -o CMakeFiles/DataFrame.dir/src/print.c.o -c /Users/hassanamad/Projects/DataFrame/src/print.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/print.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/print.c > CMakeFiles/DataFrame.dir/src/print.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/print.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/print.c -o CMakeFiles/DataFrame.dir/src/print.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o: /Users/hassanamad/Projects/DataFrame/src/query.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o -MF CMakeFiles/DataFrame.dir/src/query.c.o.d -o CMakeFiles/DataFrame.dir/src/query.c.o -c /Users/hassanamad/Projects/DataFrame/src/query.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/query.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/query.c > CMakeFiles/DataFrame.dir/src/query.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/query.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/query.c -o CMakeFiles/DataFrame.dir/src/query.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o: /Users/hassanamad/Projects/DataFrame/src/reshape.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o -MF CMakeFiles/DataFrame.dir/src/reshape.c.o.d -o CMakeFiles/DataFrame.dir/src/reshape.c.o -c /Users/hassanamad/Projects/DataFrame/src/reshape.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/reshape.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/reshape.c > CMakeFiles/DataFrame.dir/src/reshape.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/reshape.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/reshape.c -o CMakeFiles/DataFrame.dir/src/reshape.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o: /Users/hassanamad/Projects/DataFrame/src/series.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o -MF CMakeFiles/DataFrame.dir/src/series.c.o.d -o CMakeFiles/DataFrame.dir/src/series.c.o -c /Users/hassanamad/Projects/DataFrame/src/series.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/series.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/series.c > CMakeFiles/DataFrame.dir/src/series.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/series.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/series.c -o CMakeFiles/DataFrame.dir/src/series.c.s

DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/flags.make
DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o: /Users/hassanamad/Projects/DataFrame/src/dftime.c
DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o: DataFrame_build/CMakeFiles/DataFrame.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o -MF CMakeFiles/DataFrame.dir/src/dftime.c.o.d -o CMakeFiles/DataFrame.dir/src/dftime.c.o -c /Users/hassanamad/Projects/DataFrame/src/dftime.c

DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/DataFrame.dir/src/dftime.c.i"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hassanamad/Projects/DataFrame/src/dftime.c > CMakeFiles/DataFrame.dir/src/dftime.c.i

DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/DataFrame.dir/src/dftime.c.s"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hassanamad/Projects/DataFrame/src/dftime.c -o CMakeFiles/DataFrame.dir/src/dftime.c.s

# Object files for target DataFrame
DataFrame_OBJECTS = \
"CMakeFiles/DataFrame.dir/src/aggregate.c.o" \
"CMakeFiles/DataFrame.dir/src/combine.c.o" \
"CMakeFiles/DataFrame.dir/src/core.c.o" \
"CMakeFiles/DataFrame.dir/src/date.c.o" \
"CMakeFiles/DataFrame.dir/src/indexing.c.o" \
"CMakeFiles/DataFrame.dir/src/io.c.o" \
"CMakeFiles/DataFrame.dir/src/plot.c.o" \
"CMakeFiles/DataFrame.dir/src/print.c.o" \
"CMakeFiles/DataFrame.dir/src/query.c.o" \
"CMakeFiles/DataFrame.dir/src/reshape.c.o" \
"CMakeFiles/DataFrame.dir/src/series.c.o" \
"CMakeFiles/DataFrame.dir/src/dftime.c.o"

# External object files for target DataFrame
DataFrame_EXTERNAL_OBJECTS =

DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/aggregate.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/combine.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/core.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/date.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/indexing.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/io.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/plot.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/print.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/query.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/reshape.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/series.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/src/dftime.c.o
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/build.make
DataFrame_build/libDataFrame.a: DataFrame_build/CMakeFiles/DataFrame.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/hassanamad/Projects/ML/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking C static library libDataFrame.a"
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && $(CMAKE_COMMAND) -P CMakeFiles/DataFrame.dir/cmake_clean_target.cmake
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DataFrame.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
DataFrame_build/CMakeFiles/DataFrame.dir/build: DataFrame_build/libDataFrame.a
.PHONY : DataFrame_build/CMakeFiles/DataFrame.dir/build

DataFrame_build/CMakeFiles/DataFrame.dir/clean:
	cd /Users/hassanamad/Projects/ML/build/DataFrame_build && $(CMAKE_COMMAND) -P CMakeFiles/DataFrame.dir/cmake_clean.cmake
.PHONY : DataFrame_build/CMakeFiles/DataFrame.dir/clean

DataFrame_build/CMakeFiles/DataFrame.dir/depend:
	cd /Users/hassanamad/Projects/ML/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hassanamad/Projects/ML /Users/hassanamad/Projects/DataFrame /Users/hassanamad/Projects/ML/build /Users/hassanamad/Projects/ML/build/DataFrame_build /Users/hassanamad/Projects/ML/build/DataFrame_build/CMakeFiles/DataFrame.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : DataFrame_build/CMakeFiles/DataFrame.dir/depend

