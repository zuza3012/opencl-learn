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
CMAKE_SOURCE_DIR = /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build

# Include any dependencies generated for this target.
include CMakeFiles/add_vectors.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/add_vectors.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/add_vectors.dir/flags.make

CMakeFiles/add_vectors.dir/main.cpp.o: CMakeFiles/add_vectors.dir/flags.make
CMakeFiles/add_vectors.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/add_vectors.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/add_vectors.dir/main.cpp.o -c /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/main.cpp

CMakeFiles/add_vectors.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/add_vectors.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/main.cpp > CMakeFiles/add_vectors.dir/main.cpp.i

CMakeFiles/add_vectors.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/add_vectors.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/main.cpp -o CMakeFiles/add_vectors.dir/main.cpp.s

CMakeFiles/add_vectors.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/add_vectors.dir/main.cpp.o.requires

CMakeFiles/add_vectors.dir/main.cpp.o.provides: CMakeFiles/add_vectors.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/add_vectors.dir/build.make CMakeFiles/add_vectors.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/add_vectors.dir/main.cpp.o.provides

CMakeFiles/add_vectors.dir/main.cpp.o.provides.build: CMakeFiles/add_vectors.dir/main.cpp.o


# Object files for target add_vectors
add_vectors_OBJECTS = \
"CMakeFiles/add_vectors.dir/main.cpp.o"

# External object files for target add_vectors
add_vectors_EXTERNAL_OBJECTS =

add_vectors: CMakeFiles/add_vectors.dir/main.cpp.o
add_vectors: CMakeFiles/add_vectors.dir/build.make
add_vectors: /usr/lib/x86_64-linux-gnu/libOpenCL.so
add_vectors: CMakeFiles/add_vectors.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable add_vectors"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add_vectors.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/add_vectors.dir/build: add_vectors

.PHONY : CMakeFiles/add_vectors.dir/build

CMakeFiles/add_vectors.dir/requires: CMakeFiles/add_vectors.dir/main.cpp.o.requires

.PHONY : CMakeFiles/add_vectors.dir/requires

CMakeFiles/add_vectors.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/add_vectors.dir/cmake_clean.cmake
.PHONY : CMakeFiles/add_vectors.dir/clean

CMakeFiles/add_vectors.dir/depend:
	cd /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build /home/zuzanna/Documents/openCL/nauka/opencl-learn/add_vectors/build/CMakeFiles/add_vectors.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/add_vectors.dir/depend
