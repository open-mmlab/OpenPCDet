/*
Vector-pool aggregation based local feature aggregation for point cloud.
PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
https://arxiv.org/abs/2102.00463

Written by Shaoshuai Shi
All Rights Reserved 2020.
*/


#ifndef _STACK_VECTOR_POOL_GPU_H
#define _STACK_VECTOR_POOL_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


int query_stacked_local_neighbor_idxs_kernel_launcher_stack(
    const float *support_xyz, const int *xyz_batch_cnt, const float *new_xyz, const int *new_xyz_batch_cnt,
    int *stack_neighbor_idxs, int *start_len, int *cumsum, int avg_length_of_neighbor_idxs,
    float max_neighbour_distance, int batch_size, int M, int nsample, int neighbor_type);

int query_stacked_local_neighbor_idxs_wrapper_stack(at::Tensor support_xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor stack_neighbor_idxs_tensor, at::Tensor start_len_tensor, at::Tensor cumsum_tensor,
    int avg_length_of_neighbor_idxs, float max_neighbour_distance, int nsample, int neighbor_type);


int query_three_nn_by_stacked_local_idxs_kernel_launcher_stack(
    const float *support_xyz, const float *new_xyz, const float *new_xyz_grid_centers,
    int *new_xyz_grid_idxs, float *new_xyz_grid_dist2,
    const int *stack_neighbor_idxs, const int *start_len,
    int M, int num_total_grids);

int query_three_nn_by_stacked_local_idxs_wrapper_stack(at::Tensor support_xyz_tensor,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_grid_centers_tensor,
    at::Tensor new_xyz_grid_idxs_tensor, at::Tensor new_xyz_grid_dist2_tensor,
    at::Tensor stack_neighbor_idxs_tensor, at::Tensor start_len_tensor,
    int M, int num_total_grids);


int vector_pool_wrapper_stack(at::Tensor support_xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor support_features_tensor, at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor new_features_tensor, at::Tensor new_local_xyz,
    at::Tensor point_cnt_of_grid_tensor, at::Tensor grouped_idxs_tensor,
    int num_grid_x, int num_grid_y, int num_grid_z, float max_neighbour_distance, int use_xyz,
    int num_max_sum_points, int nsample, int neighbor_type, int pooling_type);


int vector_pool_kernel_launcher_stack(
    const float *support_xyz, const float *support_features, const int *xyz_batch_cnt,
    const float *new_xyz, float *new_features, float * new_local_xyz, const int *new_xyz_batch_cnt,
    int *point_cnt_of_grid, int *grouped_idxs,
    int num_grid_x, int num_grid_y, int num_grid_z, float max_neighbour_distance,
    int batch_size, int N, int M, int num_c_in, int num_c_out, int num_total_grids, int use_xyz,
    int num_max_sum_points, int nsample, int neighbor_type, int pooling_type);


int vector_pool_grad_wrapper_stack(at::Tensor grad_new_features_tensor,
    at::Tensor point_cnt_of_grid_tensor, at::Tensor grouped_idxs_tensor,
    at::Tensor grad_support_features_tensor);


void vector_pool_grad_kernel_launcher_stack(
    const float *grad_new_features, const int *point_cnt_of_grid, const int *grouped_idxs,
    float *grad_support_features, int N, int M, int num_c_out, int num_c_in, int num_total_grids,
    int num_max_sum_points);

#endif
