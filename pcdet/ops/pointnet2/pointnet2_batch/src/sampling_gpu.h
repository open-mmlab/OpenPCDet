#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out, cudaStream_t stream);


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream);


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor);

void furthest_point_sampling_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, int *idxs, cudaStream_t stream);

#endif
