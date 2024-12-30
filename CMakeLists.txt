cmake_minimum_required(VERSION 3.18)
project(cuda_fast_voxel_traversal CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 and CUDA
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)

add_library(cuda_kernels STATIC
    src/voxel_traversal.cu
)

set_target_properties(cuda_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cuda_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)
pybind11_add_module(voxel_traversal
    src/voxel_traversal.cpp
)

# Link the CUDA library to the pybind11 module
target_link_libraries(voxel_traversal PRIVATE
    cuda_kernels
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)

# Set output directory and other properties
set_target_properties(voxel_traversal PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Include directories for the pybind11 module
target_include_directories(voxel_traversal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)