cmake_minimum_required (VERSION 3.6)
 
# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()
 
# Define project name
project(dot)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")
 

set(CMAKE_CXX_FLAGS "-std=c++0x")
#find_package(OpenCL REQUIRED)
#find_package(BLAS REQUIRED)

find_package(BLAS)
find_package(LAPACK)
message(${BLAS_LIBRARIES})
message(${BLAS_INCLUDE_DIRS})
if(LAPACK_FOUND AND BLAS_FOUND)
   set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()



# Source code of application		
link_directories(${OpenCL_LIBRARY})
add_executable (
    dot 
    main.cpp
    )
 
# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
	elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
	endif ()
endif (CMAKE_COMPILER_IS_GNUCC)
 
# Set up executable

target_include_directories (dot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries (dot ${OpenCL_LIBRARY})
target_link_libraries(dot ${lapackblas_libraries})
include_directories(${BLAS_INCLUDE_DIRS})
target_link_libraries(dot ${BLAS_LIBRARIES})
