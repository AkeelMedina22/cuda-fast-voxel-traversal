#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

__device__ inline float3 normalize_f3(float3 v)
{
    float inv_len = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

__global__ void visibilityCheckKernel(
    float *depth_image,
    const uint64_t *voxels,
    const size_t num_voxels,
    const float *ray_starts,
    const float *ray_ends,
    const float resolution,
    const int image_width,
    const int image_height);

void launchVisibilityCheck(
    float *depth_image,
    const uint64_t *voxels,
    const size_t num_voxels,
    const float *ray_starts,
    const float *ray_ends,
    const float resolution,
    const int image_width,
    const int image_height);