# Temp: Add build/lib dir to python path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
build_lib_path = os.path.join(os.path.dirname(current_dir), 'build', 'lib')
print(f"Adding {build_lib_path} to sys.path")
sys.path.append(build_lib_path)

import numpy as np
import matplotlib.pyplot as plt
import voxel_traversal


def morton_encode_points(points: np.ndarray, resolution: float = 0.025) -> np.ndarray:
    """Encode 3D points into Morton codes"""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 array of points")
    
    voxels = (points / resolution).astype(np.int64)
    
    offset = 100000 # negative value offset
    voxels += offset
    
    voxels = voxels.astype(np.uint64)
    voxels &= np.uint64(0x1FFFFF)  # 21-bit mask is enough
    
    def expand_bits(x):
        x = x.astype(np.uint64)
        x = (x | (x << 32)) & np.uint64(0x1f00000000ffff)
        x = (x | (x << 16)) & np.uint64(0x1f0000ff0000ff)
        x = (x | (x << 8))  & np.uint64(0x100f00f00f00f00f)
        x = (x | (x << 4))  & np.uint64(0x10c30c30c30c30c3)
        x = (x | (x << 2))  & np.uint64(0x1249249249249249)
        return x
    
    x_expanded = expand_bits(voxels[:, 0])
    y_expanded = expand_bits(voxels[:, 1])
    z_expanded = expand_bits(voxels[:, 2])
    
    return x_expanded | (y_expanded << 1) | (z_expanded << 2)


# For testing
def morton_decode_points(morton_codes: np.ndarray, resolution: float = 0.025) -> np.ndarray:
    """Decode Morton codes back to 3D points"""

    morton_codes = morton_codes.astype(np.uint64)
    
    def compact_bits(x):
        x = x.astype(np.uint64)
        x = x & np.uint64(0x1249249249249249)
        x = (x | (x >> 2))  & np.uint64(0x10c30c30c30c30c3)
        x = (x | (x >> 4))  & np.uint64(0x100f00f00f00f00f)
        x = (x | (x >> 8))  & np.uint64(0x1f0000ff0000ff)
        x = (x | (x >> 16)) & np.uint64(0x1f00000000ffff)
        x = (x | (x >> 32)) & np.uint64(0x1FFFFF)
        return x
    
    x_expanded = morton_codes & np.uint64(0x1249249249249249)
    y_expanded = (morton_codes >> 1) & np.uint64(0x1249249249249249)
    z_expanded = (morton_codes >> 2) & np.uint64(0x1249249249249249)
    
    x = compact_bits(x_expanded)
    y = compact_bits(y_expanded)
    z = compact_bits(z_expanded)
    
    points = np.column_stack([
        x.astype(np.int64),
        y.astype(np.int64),
        z.astype(np.int64)
    ])
    
    points -= 100000
    points = points * resolution
    
    return points


def generate_rays(height, width, camera_pos, pitch, yaw, roll, depth=7.0, cube_size=7.0):
    """In essence, this function is creating a point cloud (ray_end) 
    for visibility checking against the voxel grid"""

    # Creates Orthographic rays from the pos+pose in the direction of the test case pcd
    direction = np.array([0, 0, 1])
    
    x = np.linspace(-cube_size/2, cube_size/2, width)
    y = np.linspace(-cube_size/2, cube_size/2, height)
    X, Y = np.meshgrid(x, y)
    
    ray_starts = np.zeros((height, width, 3))
    ray_starts[..., 0] = X + camera_pos[0]
    ray_starts[..., 1] = Y + camera_pos[1]
    ray_starts[..., 2] = camera_pos[2]
    
    # just translate forward in the dir of pcd (orthographic) 
    ray_ends = ray_starts + direction * depth

    ray_starts_flat = np.ascontiguousarray(ray_starts).astype(np.float32).reshape(-1)
    ray_ends_flat = np.ascontiguousarray(ray_ends).astype(np.float32).reshape(-1)

    return ray_starts_flat, ray_ends_flat


def test():
    os.makedirs(os.path.join(os.path.dirname(current_dir), "output"), exist_ok=True)

    x = np.linspace(-0.25, 0.25, 100) 
    y = np.linspace(-0.25, 0.25, 100)
    z = np.linspace(0, 1, 50)

    X, Y, Z = np.meshgrid(x, y, z)

    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten())).astype(np.float32)

    voxels = morton_encode_points(points, resolution=0.025)

    voxels.sort()  
    voxels = np.unique(voxels)
    np.save(os.path.join(os.path.dirname(current_dir), "output/voxels.npy"), voxels)

    camera_position = np.array([0, 0, -3.5], dtype=np.float32)
    pose = np.array([0, 0, 0], dtype=np.float32) 

    ray_starts, ray_ends = generate_rays(640, 480, camera_position, *pose)

    # Pass to C++ using Pybind11
    result = voxel_traversal.trace(np.ascontiguousarray(voxels).astype(np.uint64), ray_starts, ray_ends, 0.025, 640, 480)

    plt.figure(figsize=(10, 8))
    
    plt.imshow(result, cmap='jet') 
    plt.colorbar(label='Depth')
    
    plt.title('Depth Map')
    plt.axis('image')
    
    plt.savefig(os.path.join(os.path.dirname(current_dir), "output/depth_map.png"))
    plt.show()

    print(f"Result shape: {result.shape}")
    print(f"Value range: [{np.min(result)}, {np.max(result)}]")

if __name__ == "__main__":
    test()