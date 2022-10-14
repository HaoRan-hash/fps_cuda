#include "fps.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <stdio.h>

#define THREAD_PER_BLOCK 512

__device__ void __update(float* max_dists, int64_t* max_indices, int idx1, int idx2)
{
    float d1 = max_dists[idx1], d2 = max_dists[idx2];
    int64_t i1 = max_indices[idx1], i2 = max_indices[idx2];
    max_indices[idx1] = d1 > d2 ? i1 : i2;
    max_dists[idx1] = d1 > d2 ? d1 : d2;
}

__global__ void fps_kernel(torch::PackedTensorAccessor32<float, 3> points,
                           torch::PackedTensorAccessor32<float, 2> dists,
                           torch::PackedTensorAccessor32<int64_t, 2> indices)
{
    // 声明共享内存空间 (大小要和THREAD_PER_BLOCK一致!)
    __shared__ float max_dists[512];
    __shared__ int64_t max_indices[512];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (tid == 0)
    {
        for (int i = 0; i < THREAD_PER_BLOCK; i++)
        {
            max_dists[i] = 0;
            max_indices[i] = 0;
        }
    }
    __syncthreads();

    int block_size = THREAD_PER_BLOCK;
    int stride = block_size;
    int64_t old = indices[b][0];

    for (int i = 1; i < indices.size(1); i++)
    {
        float x1 = points[b][old][0];
        float y1 = points[b][old][1];
        float z1 = points[b][old][2];

        float max_dist = 0;
        int64_t max_idx = 0;
        for (int j = tid; j < points.size(1); j += stride)
        {
            float x2 = points[b][j][0];
            float y2 = points[b][j][1];
            float z2 = points[b][j][2];

            float temp = (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) + (z1-z2) * (z1-z2);
            temp = min(dists[b][j], temp);

            dists[b][j] = temp;
            max_idx = temp > max_dist ? j : max_idx;
            max_dist = temp > max_dist ? temp : max_dist;
        }
        max_dists[tid] = max_dist;
        max_indices[tid] = max_idx;
        __syncthreads();

        for (int j = block_size / 2; j > 0; j = j / 2)
        {
            if (tid < j)
                __update(max_dists, max_indices, tid, tid + j);
            
            __syncthreads();
        }

        old = max_indices[0];
        if (tid == 0)
            indices[b][i] = old;
    }
}

void fps_launcher(torch::Tensor points,
                  torch::Tensor dists,
                  torch::Tensor indices)
{
    const at::cuda::OptionalCUDAGuard device_guard(points.device());

    fps_kernel<<<points.size(0), THREAD_PER_BLOCK>>> (points.packed_accessor32<float, 3>(),
                                                      dists.packed_accessor32<float, 2>(),
                                                      indices.packed_accessor32<int64_t, 2>());
}