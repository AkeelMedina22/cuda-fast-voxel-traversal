# CUDA Fast Voxel Traversal
## Overview
An implementation of Amanatides & Woo's "A Fast Voxel Traversal Algorithm for Ray Tracing" in CUDA. The algorithm performs parallel ray casting with orthographic projection to generate depth masks by intersecting rays with voxelized point clouds.

## Key Features
- CUDA-accelerated voxel traversal
- Binary search over morton-encoded point clouds
- Python interface via Pybind11
- Orthographic projection support

## Use Case
The primary purpose is intersection testing between two point clouds:
- Input: A voxelized point cloud (test case: cube)
- Target: A voxelized point cloud (test case: plane at frustum end)

## Dependencies
- CUDA
- CMake
- Python

## Building
The project can be built using the included Makefile

## Python Interface
The implementation uses Pybind11 to expose CUDA functionality to Python, allowing direct manipulation of NumPy arrays.

## Implementation Details
- Ray casting is performed in parallel across the view frustum
- Point clouds are voxelized and morton-encoded for efficient spatial search
- Binary search is used for intersection testing
- Output is a depth mask indicating intersection points

## Usage Examples
- `make build`
- `make rebuild`

## Performance Considerations
- The input point cloud must fit in VRAM as it is stored in texture memory for optimal access patterns (to-do)
- Morton encoding helps maintain spatial locality for better cache coherence
- Performance scales with:
    - Input point cloud size
    - View frustum resolution (number of rays) / Target point cloud size
    - Voxel resolution

- Memory access patterns are optimized through:
    - Texture memory for point cloud data
    - Binary search leveraging cache-friendly morton encoding

## References
- Amanatides, J., & Woo, A. (1987). A Fast Voxel Traversal Algorithm for Ray Tracing.