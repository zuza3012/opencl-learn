# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build

# Include any dependencies generated for this target.
include CMakeFiles/hello_v2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hello_v2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello_v2.dir/flags.make

CMakeFiles/hello_v2.dir/main.cpp.o: CMakeFiles/hello_v2.dir/flags.make
CMakeFiles/hello_v2.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hello_v2.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hello_v2.dir/main.cpp.o -c /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/main.cpp

CMakeFiles/hello_v2.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hello_v2.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/main.cpp > CMakeFiles/hello_v2.dir/main.cpp.i

CMakeFiles/hello_v2.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hello_v2.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/main.cpp -o CMakeFiles/hello_v2.dir/main.cpp.s

CMakeFiles/hello_v2.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/hello_v2.dir/main.cpp.o.requires

CMakeFiles/hello_v2.dir/main.cpp.o.provides: CMakeFiles/hello_v2.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/hello_v2.dir/build.make CMakeFiles/hello_v2.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/hello_v2.dir/main.cpp.o.provides

CMakeFiles/hello_v2.dir/main.cpp.o.provides.build: CMakeFiles/hello_v2.dir/main.cpp.o


CMakeFiles/hello_v2.dir/functions.cpp.o: CMakeFiles/hello_v2.dir/flags.make
CMakeFiles/hello_v2.dir/functions.cpp.o: ../functions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/hello_v2.dir/functions.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hello_v2.dir/functions.cpp.o -c /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/functions.cpp

CMakeFiles/hello_v2.dir/functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hello_v2.dir/functions.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/functions.cpp > CMakeFiles/hello_v2.dir/functions.cpp.i

CMakeFiles/hello_v2.dir/functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hello_v2.dir/functions.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/functions.cpp -o CMakeFiles/hello_v2.dir/functions.cpp.s

CMakeFiles/hello_v2.dir/functions.cpp.o.requires:

.PHONY : CMakeFiles/hello_v2.dir/functions.cpp.o.requires

CMakeFiles/hello_v2.dir/functions.cpp.o.provides: CMakeFiles/hello_v2.dir/functions.cpp.o.requires
	$(MAKE) -f CMakeFiles/hello_v2.dir/build.make CMakeFiles/hello_v2.dir/functions.cpp.o.provides.build
.PHONY : CMakeFiles/hello_v2.dir/functions.cpp.o.provides

CMakeFiles/hello_v2.dir/functions.cpp.o.provides.build: CMakeFiles/hello_v2.dir/functions.cpp.o


# Object files for target hello_v2
hello_v2_OBJECTS = \
"CMakeFiles/hello_v2.dir/main.cpp.o" \
"CMakeFiles/hello_v2.dir/functions.cpp.o"

# External object files for target hello_v2
hello_v2_EXTERNAL_OBJECTS =

hello_v2: CMakeFiles/hello_v2.dir/main.cpp.o
hello_v2: CMakeFiles/hello_v2.dir/functions.cpp.o
hello_v2: CMakeFiles/hello_v2.dir/build.make
hello_v2: /usr/lib/x86_64-linux-gnu/libOpenCL.so
hello_v2: CMakeFiles/hello_v2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable hello_v2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_v2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello_v2.dir/build: hello_v2

.PHONY : CMakeFiles/hello_v2.dir/build

CMakeFiles/hello_v2.dir/requires: CMakeFiles/hello_v2.dir/main.cpp.o.requires
CMakeFiles/hello_v2.dir/requires: CMakeFiles/hello_v2.dir/functions.cpp.o.requires

.PHONY : CMakeFiles/hello_v2.dir/requires

CMakeFiles/hello_v2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello_v2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello_v2.dir/clean

CMakeFiles/hello_v2.dir/depend:
	cd /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2 /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2 /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build /home/zuzanna/Documents/openCL/nauka/opencl-learn/hello_v2/build/CMakeFiles/hello_v2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hello_v2.dir/depend
