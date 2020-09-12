cmake_minimum_required (VERSION 3.1)

project(hello_v2)

set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (
    hello_v2 
    main.cpp
    functions.cpp
    functions.h
    )
target_include_directories (hello_v2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (hello_v2 ${OpenCL_LIBRARY})
