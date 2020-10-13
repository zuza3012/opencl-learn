cmake_minimum_required (VERSION 3.1)

project(reduction)

set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (
    reduction 
    main.cpp
    )
target_include_directories (reduction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (reduction ${OpenCL_LIBRARY})
