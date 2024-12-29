cmake_minimum_required(VERSION 3.22)
project(cuda_fast_voxel_traversal)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)

# Add the pybind11 module
pybind11_add_module(voxel_traversal
    src/voxel_traversal.cpp
)

# Set output directory for the built module
set_target_properties(voxel_traversal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Include directories
target_include_directories(voxel_traversal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)