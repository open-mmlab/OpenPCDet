#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"

extern THCState *state;


void three_nn_wrapper_stack(at::Tensor unknown_tensor, 
    at::Tensor unknown_batch_cnt_tensor, at::Tensor known_tensor, 
    at::Tensor known_batch_cnt_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor){
    // unknown: (N1 + N2 ..., 3)
    // unknown_batch_cnt: (batch_size), [N1, N2, ...]
    // known: (M1 + M2 ..., 3)
    // known_batch_cnt: (batch_size), [M1, M2, ...]
    // Return:
    // dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
    // idx: (N1 + N2 ..., 3)  index of the three nearest neighbors

    int batch_size = unknown_batch_cnt_tensor.size(0);
    int N = unknown_tensor.size(0);
    int M = known_tensor.size(0);
    const float *unknown = unknown_tensor.data<float>();
    const int *unknown_batch_cnt = unknown_batch_cnt_tensor.data<int>();
    const float *known = known_tensor.data<float>();
    const int *known_batch_cnt = known_batch_cnt_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    three_nn_kernel_launcher_stack(batch_size, N, M, unknown, unknown_batch_cnt, known, known_batch_cnt, dist2, idx);
}


void three_interpolate_wrapper_stack(at::Tensor features_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor) {
    // features_tensor: (M1 + M2 ..., C)
    // idx_tensor: [N1 + N2 ..., 3]
    // weight_tensor: [N1 + N2 ..., 3]
    // Return:
    // out_tensor: (N1 + N2 ..., C)

    int N = out_tensor.size(0);
    int channels = features_tensor.size(1);
    const float *features = features_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    three_interpolate_kernel_launcher_stack(N, channels, features, idx, weight, out);
}


void three_interpolate_grad_wrapper_stack(at::Tensor grad_out_tensor, at::Tensor idx_tensor,
    at::Tensor weight_tensor, at::Tensor grad_features_tensor) {
    // grad_out_tensor: (N1 + N2 ..., C)
    // idx_tensor: [N1 + N2 ..., 3]
    // weight_tensor: [N1 + N2 ..., 3]
    // Return:
    // grad_features_tensor: (M1 + M2 ..., C)

    int N = grad_out_tensor.size(0);
    int channels = grad_out_tensor.size(1);
    const float *grad_out = grad_out_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_features = grad_features_tensor.data<float>();
    
    three_interpolate_grad_kernel_launcher_stack(N, channels, grad_out, idx, weight, grad_features);
}