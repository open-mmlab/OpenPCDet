/*
Vector-pool aggregation based local feature aggregation for point cloud.
PV-RCNN++: Point-Voxel Feature Set Abstraction With Local Vector Representation for 3D Object Detection
https://arxiv.org/abs/2102.00463

Written by Shaoshuai Shi
All Rights Reserved 2020.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vector_pool_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int query_stacked_local_neighbor_idxs_wrapper_stack(at::Tensor support_xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor stack_neighbor_idxs_tensor, at::Tensor start_len_tensor, at::Tensor cumsum_tensor,
    int avg_length_of_neighbor_idxs, float max_neighbour_distance, int nsample, int neighbor_type){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
    // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
    // num_grid_x, num_grid_y, num_grid_z: number of grids in each local area centered at new_xyz
    // nsample: find all (-1), find limited number(>0)
    // neighbor_type: 1: ball, others: cube

    CHECK_INPUT(support_xyz_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(new_xyz_batch_cnt_tensor);
    CHECK_INPUT(stack_neighbor_idxs_tensor);
    CHECK_INPUT(start_len_tensor);
    CHECK_INPUT(cumsum_tensor);

    const float *support_xyz = support_xyz_tensor.data<float>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
    int *stack_neighbor_idxs = stack_neighbor_idxs_tensor.data<int>();
    int *start_len = start_len_tensor.data<int>();
    int *cumsum = cumsum_tensor.data<int>();

    int batch_size = xyz_batch_cnt_tensor.size(0);
    int M = new_xyz_tensor.size(0);

    query_stacked_local_neighbor_idxs_kernel_launcher_stack(
        support_xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt,
        stack_neighbor_idxs, start_len, cumsum, avg_length_of_neighbor_idxs,
        max_neighbour_distance, batch_size, M, nsample, neighbor_type
    );
    return 0;
}


int query_three_nn_by_stacked_local_idxs_wrapper_stack(at::Tensor support_xyz_tensor,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_grid_centers_tensor,
    at::Tensor new_xyz_grid_idxs_tensor, at::Tensor new_xyz_grid_dist2_tensor,
    at::Tensor stack_neighbor_idxs_tensor, at::Tensor start_len_tensor,
    int M, int num_total_grids){
    // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
    // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
    // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
    // stack_neighbor_idxs: (max_length_of_neighbor_idxs)
    // start_len: (M1 + M2, 2)  [start_offset, neighbor_length]

    CHECK_INPUT(support_xyz_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(new_xyz_grid_centers_tensor);
    CHECK_INPUT(new_xyz_grid_idxs_tensor);
    CHECK_INPUT(new_xyz_grid_dist2_tensor);
    CHECK_INPUT(stack_neighbor_idxs_tensor);
    CHECK_INPUT(start_len_tensor);

    const float *support_xyz = support_xyz_tensor.data<float>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *new_xyz_grid_centers = new_xyz_grid_centers_tensor.data<float>();
    int *new_xyz_grid_idxs = new_xyz_grid_idxs_tensor.data<int>();
    float *new_xyz_grid_dist2 = new_xyz_grid_dist2_tensor.data<float>();
    int *stack_neighbor_idxs = stack_neighbor_idxs_tensor.data<int>();
    int *start_len = start_len_tensor.data<int>();

    query_three_nn_by_stacked_local_idxs_kernel_launcher_stack(
        support_xyz, new_xyz, new_xyz_grid_centers,
        new_xyz_grid_idxs, new_xyz_grid_dist2, stack_neighbor_idxs, start_len,
        M, num_total_grids
    );
    return 0;
}


int vector_pool_wrapper_stack(at::Tensor support_xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor support_features_tensor, at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor new_features_tensor, at::Tensor new_local_xyz_tensor,
    at::Tensor point_cnt_of_grid_tensor, at::Tensor grouped_idxs_tensor,
    int num_grid_x, int num_grid_y, int num_grid_z, float max_neighbour_distance, int use_xyz,
    int num_max_sum_points, int nsample, int neighbor_type, int pooling_type){
    // support_xyz_tensor: (N1 + N2 ..., 3) xyz coordinates of the features
    // support_features_tensor: (N1 + N2 ..., C)
    // xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // new_xyz_tensor: (M1 + M2 ..., 3) centers of new positions
    // new_features_tensor: (M1 + M2 ..., C)
    // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // point_cnt_of_grid: (M1 + M2 ..., num_total_grids)
    // grouped_idxs_tensor: (num_max_sum_points, 3)
    // num_grid_x, num_grid_y, num_grid_z: number of grids in each local area centered at new_xyz
    // use_xyz: whether to calculate new_local_xyz
    // neighbor_type: 1: ball, others: cube
    // pooling_type: 0: avg_pool, 1: random choice

    CHECK_INPUT(support_xyz_tensor);
    CHECK_INPUT(support_features_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(new_xyz_batch_cnt_tensor);
    CHECK_INPUT(new_features_tensor);
    CHECK_INPUT(new_local_xyz_tensor);
    CHECK_INPUT(point_cnt_of_grid_tensor);
    CHECK_INPUT(grouped_idxs_tensor);

    const float *support_xyz = support_xyz_tensor.data<float>();
    const float *support_features = support_features_tensor.data<float>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
    float *new_features = new_features_tensor.data<float>();
    float *new_local_xyz = new_local_xyz_tensor.data<float>();
    int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
    int *grouped_idxs = grouped_idxs_tensor.data<int>();

    int N = support_xyz_tensor.size(0);
    int batch_size = xyz_batch_cnt_tensor.size(0);
    int M = new_xyz_tensor.size(0);
    int num_c_out = new_features_tensor.size(1);
    int num_c_in = support_features_tensor.size(1);
    int num_total_grids = point_cnt_of_grid_tensor.size(1);

    int cum_sum = vector_pool_kernel_launcher_stack(
        support_xyz, support_features, xyz_batch_cnt,
        new_xyz, new_features, new_local_xyz, new_xyz_batch_cnt,
        point_cnt_of_grid, grouped_idxs,
        num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance,
        batch_size, N, M, num_c_in, num_c_out, num_total_grids, use_xyz, num_max_sum_points, nsample, neighbor_type, pooling_type
    );
    return cum_sum;
}


int vector_pool_grad_wrapper_stack(at::Tensor grad_new_features_tensor,
    at::Tensor point_cnt_of_grid_tensor, at::Tensor grouped_idxs_tensor,
    at::Tensor grad_support_features_tensor) {
    // grad_new_features_tensor: (M1 + M2 ..., C_out)
    // point_cnt_of_grid_tensor: (M1 + M2 ..., num_total_grids)
    // grouped_idxs_tensor: (num_max_sum_points, 3) [idx of support_xyz, idx of new_xyz, idx of grid_idx in new_xyz]
    // grad_support_features_tensor: (N1 + N2 ..., C_in)

    CHECK_INPUT(grad_new_features_tensor);
    CHECK_INPUT(point_cnt_of_grid_tensor);
    CHECK_INPUT(grouped_idxs_tensor);
    CHECK_INPUT(grad_support_features_tensor);

    int M = grad_new_features_tensor.size(0);
    int num_c_out = grad_new_features_tensor.size(1);
    int N = grad_support_features_tensor.size(0);
    int num_c_in = grad_support_features_tensor.size(1);
    int num_total_grids = point_cnt_of_grid_tensor.size(1);
    int num_max_sum_points = grouped_idxs_tensor.size(0);

    const float *grad_new_features = grad_new_features_tensor.data<float>();
    const int *point_cnt_of_grid = point_cnt_of_grid_tensor.data<int>();
    const int *grouped_idxs = grouped_idxs_tensor.data<int>();
    float *grad_support_features = grad_support_features_tensor.data<float>();

    vector_pool_grad_kernel_launcher_stack(
        grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features,
        N, M, num_c_out, num_c_in, num_total_grids, num_max_sum_points
    );
    return 1;
}
