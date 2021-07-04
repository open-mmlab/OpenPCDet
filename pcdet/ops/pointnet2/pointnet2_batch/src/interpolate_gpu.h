#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
  at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown,
	const float *known, float *dist2, int *idx);


void three_interpolate_wrapper_fast(int b, int c, int m, int n, at::Tensor points_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out);


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m, at::Tensor grad_out_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_points_tensor);

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, float *grad_points);

#endif
