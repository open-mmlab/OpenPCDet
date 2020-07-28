/*
Stacked-batch-data version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "interpolate_gpu.h"


__global__ void three_nn_kernel_stack(int batch_size, int N, int M, const float *unknown, 
    const int *unknown_batch_cnt, const float *known, const int *known_batch_cnt,
    float *dist2, int *idx) {
    // unknown: (N1 + N2 ..., 3)
    // unknown_batch_cnt: (batch_size), [N1, N2, ...]
    // known: (M1 + M2 ..., 3)
    // known_batch_cnt: (batch_size), [M1, M2, ...]
    // Return:
    // dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
    // idx: (N1 + N2 ..., 3)  index of the three nearest neighbors

    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N) return;

    int bs_idx = 0, pt_cnt = unknown_batch_cnt[0];
    for (int k = 1; k < batch_size; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += unknown_batch_cnt[k];
        bs_idx = k;
    }

    int cur_num_known_points = known_batch_cnt[bs_idx];

    int known_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) known_batch_start_idx += known_batch_cnt[k];

    known += known_batch_start_idx * 3;
    unknown += pt_idx * 3;
    dist2 += pt_idx * 3;
    idx += pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < cur_num_known_points; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        } 
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        } 
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
    idx[0] = besti1 + known_batch_start_idx; 
    idx[1] = besti2 + known_batch_start_idx; 
    idx[2] = besti3 + known_batch_start_idx;
}


void three_nn_kernel_launcher_stack(int batch_size, int N, int M, const float *unknown, 
    const int *unknown_batch_cnt, const float *known, const int *known_batch_cnt,
    float *dist2, int *idx) {
    // unknown: (N1 + N2 ..., 3)
    // unknown_batch_cnt: (batch_size), [N1, N2, ...]
    // known: (M1 + M2 ..., 3)
    // known_batch_cnt: (batch_size), [M1, M2, ...]
    // Return:
    // dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
    // idx: (N1 + N2 ..., 3)  index of the three nearest neighbors

    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    three_nn_kernel_stack<<<blocks, threads>>>(
        batch_size, N, M, unknown, unknown_batch_cnt, 
        known, known_batch_cnt, dist2, idx
    );

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



__global__ void three_interpolate_kernel_stack(int N, int channels, const float *features, 
    const int *idx, const float *weight, float *out) {
    // features: (M1 + M2 ..., C)
    // idx: [N1 + N2 ..., 3]
    // weight: [N1 + N2 ..., 3]
    // Return:
    // out: (N1 + N2 ..., C)

    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N || c_idx >= channels) return;

    weight += pt_idx * 3;
    idx += pt_idx * 3;
    out += pt_idx * channels + c_idx;

    out[0] = weight[0] * features[idx[0] * channels + c_idx] + 
        weight[1] * features[idx[1] * channels + c_idx] + 
        weight[2] * features[idx[2] * channels + c_idx];
}



void three_interpolate_kernel_launcher_stack(int N, int channels,
    const float *features, const int *idx, const float *weight, float *out) {
    // features: (M1 + M2 ..., C)
    // idx: [N1 + N2 ..., 3]
    // weight: [N1 + N2 ..., 3]
    // Return:
    // out: (N1 + N2 ..., C)

    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), channels);
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_kernel_stack<<<blocks, threads>>>(N, channels, features, idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void three_interpolate_grad_kernel_stack(int N, int channels, const float *grad_out, 
    const int *idx, const float *weight, float *grad_features) {
    // grad_out_tensor: (N1 + N2 ..., C)
    // idx_tensor: [N1 + N2 ..., 3]
    // weight_tensor: [N1 + N2 ..., 3]
    // Return:
    // grad_features_tensor: (M1 + M2 ..., C)

    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N || c_idx >= channels) return;

    grad_out += pt_idx * channels + c_idx;
    weight += pt_idx * 3;
    idx += pt_idx * 3;
    
    // printf("pt_idx=%d, c_idx=%d, idx=(%d, %d, %d), grad_out=%f\n", pt_idx, c_idx, idx[0], idx[1], idx[2], grad_out[0]);

    atomicAdd(grad_features + idx[0] * channels + c_idx, grad_out[0] * weight[0]);
    atomicAdd(grad_features + idx[1] * channels + c_idx, grad_out[0] * weight[1]);
    atomicAdd(grad_features + idx[2] * channels + c_idx, grad_out[0] * weight[2]);
}


void three_interpolate_grad_kernel_launcher_stack(int N, int channels, const float *grad_out, 
    const int *idx, const float *weight, float *grad_features) {
    // grad_out_tensor: (N1 + N2 ..., C)
    // idx_tensor: [N1 + N2 ..., 3]
    // weight_tensor: [N1 + N2 ..., 3]
    // Return:
    // grad_features_tensor: (M1 + M2 ..., C)

    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), channels);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_grad_kernel_stack<<<blocks, threads>>>(
        N, channels, grad_out, idx, weight, grad_features
    );

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}