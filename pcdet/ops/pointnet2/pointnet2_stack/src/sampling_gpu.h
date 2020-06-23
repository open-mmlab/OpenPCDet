#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>


int furthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream);

#endif
