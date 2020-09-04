#Minimal OpenCL CMakeLists.txt by StreamHPC

cmake_minimum_required (VERSION 3.1)
project(info)
set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (info main.cpp)
target_include_directories (info PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (info ${OpenCL_LIBRARY})
