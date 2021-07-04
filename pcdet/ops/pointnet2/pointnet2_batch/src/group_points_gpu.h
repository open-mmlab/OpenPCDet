#ifndef _GROUP_POINTS_GPU_H
#define _GROUP_POINTS_GPU_H

#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>


int group_points_wrapper_fast(int b, int c, int n, int npoints, int nsample, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor);

void group_points_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out);

int group_points_grad_wrapper_fast(int b, int c, int n, int npoints, int nsample, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor);

void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *grad_out, const int *idx, float *grad_points);

#endif
