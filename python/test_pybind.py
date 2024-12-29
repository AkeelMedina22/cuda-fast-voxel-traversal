import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
sys.path.append('../build/lib')
import voxel_traversal

def test_point_cloud_roundtrip():
    points = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    camera_position = np.array([0, 0, 0], dtype=np.float32)
    
    # Pass to C++ using Pybind11
    result = voxel_traversal.create_point_cloud(points, camera_position, image_width=640, image_height=480)
    
    plt.figure(figsize=(10, 8))
    
    plt.imshow(result, cmap='jet') 
    plt.colorbar(label='Depth')
    
    plt.title('Depth Map')
    plt.axis('image') 
    
    plt.savefig('../output/depth_map.png')
    plt.show()

    print(f"Result shape: {result.shape}")
    print(f"Value range: [{np.min(result)}, {np.max(result)}]")

if __name__ == "__main__":
    test_point_cloud_roundtrip()