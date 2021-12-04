/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_bin_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_bin_kernel_stack(int B, int M, float radius, int nsample, int bin_nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx, int *bin_idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, 8, nsample) filled with -1
    //      bin_idx: (M, bin_nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * 8 * nsample;
    bin_idx += pt_idx * bin_nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt0 = 0;
    int cnt1 = 0;
    int cnt2 = 0;
    int cnt3 = 0;
    int cnt4 = 0;
    int cnt5 = 0;
    int cnt6 = 0;
    int cnt7 = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float diff_x = new_x - x;
        float diff_y = new_y - y;
        float diff_z = new_z - z;
        float d2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        if (d2 < radius2){
            if (diff_x >=0 && diff_y >= 0 && diff_z >= 0) {
                if (cnt0 >= nsample) continue;
                idx[0*nsample + cnt0] = k;
                ++cnt0;
            } else if (diff_x >=0 && diff_y >= 0 && diff_z < 0) {
                if (cnt1 >= nsample) continue;
                idx[1*nsample + cnt1] = k;
                ++cnt1;
            } else if (diff_x >=0 && diff_y < 0 && diff_z >= 0) {
                if (cnt2 >= nsample) continue;
                idx[2*nsample + cnt2] = k;
                ++cnt2;                
            } else if (diff_x >=0 && diff_y < 0 && diff_z < 0) {
                if (cnt3 >= nsample) continue;
                idx[3*nsample + cnt3] = k;
                ++cnt3;                
            } else if (diff_x < 0 && diff_y >= 0 && diff_z >= 0) {
                if (cnt4 >= nsample) continue;
                idx[4*nsample + cnt4] = k;
                ++cnt4;                
            } else if (diff_x < 0 && diff_y >= 0 && diff_z < 0) {
                if (cnt5 >= nsample) continue;
                idx[5*nsample + cnt5] = k;
                ++cnt5;                
            } else if (diff_x < 0 && diff_y < 0 && diff_z >= 0) {
                if (cnt6 >= nsample) continue;
                idx[6*nsample + cnt6] = k;
                ++cnt6;                
            } else {
                if (cnt7 >= nsample) continue;
                idx[7*nsample + cnt7] = k;
                ++cnt7;                
            }
        }
    }

    int cnt_sample = 0;
    for (int level_idx = 0; level_idx < nsample; ++level_idx){
        if (cnt_sample >= bin_nsample) break;
        for (int split_idx = 0; split_idx < 8; ++split_idx){
            if (cnt_sample >= bin_nsample) break;
            int real_idx = idx[split_idx * nsample + level_idx];
            if (real_idx < 0) continue; // -1
            if (cnt_sample == 0) {
                for (int i = 0; i < bin_nsample; ++i){
                    bin_idx[i] = real_idx;
                }
            } else {
                bin_idx[cnt_sample] = real_idx;
            }
            ++cnt_sample;
        }
    }
    if (cnt_sample == 0) bin_idx[0] = -1;
}


void ball_query_bin_kernel_launcher_stack(int B, int M, float radius, int nsample, int bin_nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx, int *bin_idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_bin_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, bin_nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx, bin_idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}