/*
batch version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor, 
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {
    const float *unknown = unknown_tensor.data<float>();
    const float *known = known_tensor.data<float>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx);
}


void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    const float *points = points_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *out = out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out);
}


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();

    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points);
}
