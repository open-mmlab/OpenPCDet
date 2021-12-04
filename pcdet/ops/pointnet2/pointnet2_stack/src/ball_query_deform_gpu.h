#ifndef _STACK_BALL_QUERY_DEFORM_GPU_H
#define _STACK_BALL_QUERY_DEFORM_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_deform_wrapper_stack(int B, int M, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_r_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor);


void ball_query_deform_kernel_launcher_stack(int B, int M, int nsample,
    const float *new_xyz, const float *new_xyz_r, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx);


#endif