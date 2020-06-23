/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <THC/THC.h>
#include "group_points_gpu.h"

extern THCState *state;
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int group_points_grad_wrapper_stack(int B, int M, int C, int N, int nsample,
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor,
    at::Tensor features_batch_cnt_tensor, at::Tensor grad_features_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(grad_features_tensor);

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    float *grad_features = grad_features_tensor.data<float>();

    group_points_grad_kernel_launcher_stack(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);
    return 1;
}


int group_points_wrapper_stack(int B, int M, int C, int nsample,
    at::Tensor features_tensor, at::Tensor features_batch_cnt_tensor,
    at::Tensor idx_tensor, at::Tensor idx_batch_cnt_tensor, at::Tensor out_tensor) {

    CHECK_INPUT(features_tensor);
    CHECK_INPUT(features_batch_cnt_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(idx_batch_cnt_tensor);
    CHECK_INPUT(out_tensor);

    const float *features = features_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const int *features_batch_cnt = features_batch_cnt_tensor.data<int>();
    const int *idx_batch_cnt = idx_batch_cnt_tensor.data<int>();
    float *out = out_tensor.data<float>();

    group_points_kernel_launcher_stack(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    return 1;
}