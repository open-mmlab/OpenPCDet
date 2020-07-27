#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H

#include <torch/serialize/tensor.h>
#include<vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void three_nn_wrapper_stack(at::Tensor unknown_tensor, 
    at::Tensor unknown_batch_cnt_tensor, at::Tensor known_tensor, 
    at::Tensor known_batch_cnt_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);


void three_interpolate_wrapper_stack(at::Tensor features_tensor, 
    at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor out_tensor);



void three_interpolate_grad_wrapper_stack(at::Tensor grad_out_tensor, at::Tensor idx_tensor,
    at::Tensor weight_tensor, at::Tensor grad_features_tensor);


void three_nn_kernel_launcher_stack(int batch_size, int N, int M, const float *unknown, 
    const int *unknown_batch_cnt, const float *known, const int *known_batch_cnt,
    float *dist2, int *idx);


void three_interpolate_kernel_launcher_stack(int N, int channels,
    const float *features, const int *idx, const float *weight, float *out);



void three_interpolate_grad_kernel_launcher_stack(int N, int channels, const float *grad_out, 
    const int *idx, const float *weight, float *grad_features);



#endif