cmake_minimum_required (VERSION 3.1)

project(add_vectors)

set(CMAKE_CXX_FLAGS "-std=c++0x")
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (
    add_vectors 
    main.cpp
    )
target_include_directories (add_vectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (add_vectors ${OpenCL_LIBRARY})
