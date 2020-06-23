/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#ifndef _STACK_BALL_QUERY_GPU_H
#define _STACK_BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_wrapper_stack(int B, int M, float radius, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor);


void ball_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx);


#endif
