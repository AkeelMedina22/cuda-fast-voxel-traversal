#include "voxel_traversal.cuh"
#include <stdio.h>

const int BLOCK_SIZE = 16;
cudaTextureObject_t voxel_tex;
cudaArray_t voxel_array;

__device__ inline float norm_f3(const float3 &v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline bool any_lt_one_and_not_close(const float3 &t_for_next_voxel, const float epsilon = 1e-5f)
{
    bool x_valid = (t_for_next_voxel.x < 1.0f) && (fabsf(t_for_next_voxel.x - 1.0f) > epsilon);
    bool y_valid = (t_for_next_voxel.y < 1.0f) && (fabsf(t_for_next_voxel.y - 1.0f) > epsilon);
    bool z_valid = (t_for_next_voxel.z < 1.0f) && (fabsf(t_for_next_voxel.z - 1.0f) > epsilon);

    return x_valid || y_valid || z_valid;
}
__device__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator/(const float3 &a, const float &b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ inline float3 operator/(const float &a, const float3 &b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
}

__device__ inline float3 operator/(const float3 &a, const float3 &b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ inline int3 floor3(const float3 &a)
{
    return make_int3(
        (int)floorf(a.x),
        (int)floorf(a.y),
        (int)floorf(a.z));
}
__device__ inline float3 abs3(const float3 &a)
{
    return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}

__device__ inline int3 operator+(const int3 &a, const int3 &b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(const float3 &a, const float &b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ inline float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3 operator*(const float &a, const int3 &b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ inline int3 operator>(const float3 &a, const int &b)
{
    return make_int3(a.x > b, a.y > b, a.z > b);
}

__device__ inline int3 operator<(const float3 &a, const int &b)
{
    return make_int3(a.x < b, a.y < b, a.z < b);
}

__device__ inline int3 operator&&(const int3 &a, const int3 &b)
{
    return make_int3(a.x && b.x, a.y && b.y, a.z && b.z);
}

__device__ inline uint64_t expand_bits(uint64_t x)
{
    x = x & 0x1FFFFF;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
}

__device__ inline uint64_t encode_morton(const int3 &point, const float resolution)
{
    uint64_t x = static_cast<uint64_t>(static_cast<float>(point.x) / resolution) + 100000;
    uint64_t y = static_cast<uint64_t>(static_cast<float>(point.y) / resolution) + 100000;
    uint64_t z = static_cast<uint64_t>(static_cast<float>(point.z) / resolution) + 100000;

    x &= 0x1FFFFF;
    y &= 0x1FFFFF;
    z &= 0x1FFFFF;

    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

__device__ inline uint64_t encode_morton(const float3 &point, const float resolution)
{
    uint64_t x = static_cast<uint64_t>(point.x / resolution + 100000);
    uint64_t y = static_cast<uint64_t>(point.y / resolution + 100000);
    uint64_t z = static_cast<uint64_t>(point.z / resolution + 100000);

    x &= 0x1FFFFF;
    y &= 0x1FFFFF;
    z &= 0x1FFFFF;

    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

__device__ inline uint64_t compact_bits(uint64_t x)
{
    x = x & 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x | (x >> 4)) & 0x100f00f00f00f00f;
    x = (x | (x >> 8)) & 0x1f0000ff0000ff;
    x = (x | (x >> 16)) & 0x1f00000000ffff;
    x = (x | (x >> 32)) & 0x1FFFFF;
    return x;
}

__device__ inline float3 decode_morton_f3(uint64_t voxel, const float resolution)
{
    uint64_t x_expanded = voxel & 0x1249249249249249;
    uint64_t y_expanded = (voxel >> 1) & 0x1249249249249249;
    uint64_t z_expanded = (voxel >> 2) & 0x1249249249249249;

    uint64_t x = compact_bits(x_expanded);
    uint64_t y = compact_bits(y_expanded);
    uint64_t z = compact_bits(z_expanded);

    float3 point = make_float3(x, y, z);
    point.x -= 100000;
    point.y -= 100000;
    point.z -= 100000;

    point = point * resolution;

    return point;
}

__global__ void visibilityCheckKernel(
    float *depth_image,
    const uint64_t *voxels,
    const size_t num_voxels,
    const float *ray_starts,
    const float *ray_ends,
    const float resolution,
    const int image_width,
    const int image_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= image_width || y >= image_height)
        return;

    int idx = y * image_width + x;
    int base_idx = idx * 3; // 3 elements per index

    float3 ray_start = make_float3(
        ray_starts[base_idx],
        ray_starts[base_idx + 1],
        ray_starts[base_idx + 2]);

    float3 ray_end = make_float3(
        ray_ends[base_idx],
        ray_ends[base_idx + 1],
        ray_ends[base_idx + 2]);

    float3 ray_dir = normalize_f3(ray_end - ray_start);

    float3 start_voxel_f = ray_start / resolution;
    int3 start_voxel = floor3(start_voxel_f);

    float3 traversal_direction = ray_dir / abs3(ray_dir);
    float3 t_per_voxel = resolution / abs3(ray_dir);

    float3 t_for_next_voxel = (resolution * (start_voxel + (traversal_direction > 0)) - ray_start) / ray_dir;

    while (any_lt_one_and_not_close(t_for_next_voxel))
    {
        float min_t = t_for_next_voxel.x;
        int next_direction_index = 0;

        if (!isnan(t_for_next_voxel.y) && (isnan(min_t) || t_for_next_voxel.y < min_t))
        {
            min_t = t_for_next_voxel.y;
            next_direction_index = 1;
        }

        if (!isnan(t_for_next_voxel.z) && (isnan(min_t) || t_for_next_voxel.z < min_t))
        {
            min_t = t_for_next_voxel.z;
            next_direction_index = 2;
        }

        if (next_direction_index == 0)
        {
            t_for_next_voxel.x += t_per_voxel.x;
            start_voxel.x += traversal_direction.x;
        }
        else if (next_direction_index == 1)
        {
            t_for_next_voxel.y += t_per_voxel.y;
            start_voxel.y += traversal_direction.y;
        }
        else
        {
            t_for_next_voxel.z += t_per_voxel.z;
            start_voxel.z += traversal_direction.z;
        }

        uint64_t morton_code = encode_morton(start_voxel, resolution);

        int start = 0;
        int end = num_voxels - 1;

        while (start <= end)
        {
            int mid = start + (end - start) / 2;
            uint64_t voxel = voxels[mid];
            if (voxel == morton_code)
            {
                float3 hit = decode_morton_f3(voxel, resolution);
                float hit_distance = norm_f3(hit - ray_start);
                float end_distance = norm_f3(ray_end - ray_start);

                if (hit_distance <= end_distance)
                {
                    depth_image[idx] = hit_distance;
                    return;
                }
                depth_image[idx] = end_distance;
                break;
            }
            else if (voxel < morton_code)
            {
                start = mid + 1;
            }
            else
            {
                end = mid - 1;
            }
        }
    }
    float end_distance = norm_f3(ray_end - ray_start);
    depth_image[idx] = end_distance;
}

void launchVisibilityCheck(
    float *depth_image,
    const uint64_t *voxels,
    const size_t num_voxels,
    const float *ray_starts,
    const float *ray_ends,
    const float resolution,
    const int image_width,
    const int image_height)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(
        (image_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (image_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float *d_depth_image;
    float *d_ray_starts;
    float *d_ray_ends;
    uint64_t *d_voxels;

    cudaMalloc(&d_depth_image, image_width * image_height * sizeof(float));
    cudaMalloc(&d_ray_starts, 3 * image_width * image_height * sizeof(float));
    cudaMalloc(&d_ray_ends, 3 * image_width * image_height * sizeof(float));
    cudaMalloc(&d_voxels, num_voxels * sizeof(uint64_t));

    // Copy data to device
    cudaMemcpy(d_voxels, voxels, num_voxels * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_starts, ray_starts, 3 * image_width * image_height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ray_ends, ray_ends, 3 * image_width * image_height * sizeof(float),
               cudaMemcpyHostToDevice);

    visibilityCheckKernel<<<numBlocks, threadsPerBlock>>>(
        d_depth_image,
        d_voxels,
        num_voxels,
        d_ray_starts,
        d_ray_ends,
        resolution,
        image_width,
        image_height);

    cudaMemcpy(depth_image, d_depth_image, image_width * image_height * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_depth_image);
    cudaFree(d_voxels);
    cudaFree(d_ray_starts);
    cudaFree(d_ray_ends);
}