#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>


int farthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs);

int stack_farthest_point_sampling_wrapper(
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor idx_tensor, at::Tensor num_sampled_points_tensor);


void stack_farthest_point_sampling_kernel_launcher(int N, int batch_size,
    const float *dataset, float *temp, int *xyz_batch_cnt, int *idxs, int *num_sampled_points);

#endif
