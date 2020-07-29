/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();

    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
}


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}
