/*
RoI-aware point cloud feature pooling
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__device__ inline int check_pt_in_box_bev(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1];
    float cx = box3d[0], cy = box3d[1];
    float dx = box3d[3], dy = box3d[4], rz = box3d[6];

    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__global__ void generate_pts_mask_for_box3d(int boxes_num, int pts_num, int out_x, int out_y, int out_z,
    const float *rois, const float *pts, int *pts_mask){
    // params rois: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (npoints, 3) [x, y, z]
    // params pts_mask: (N, npoints): -1 means point doesnot in this box, otherwise: encode (x_idxs, y_idxs, z_idxs) by binary bit
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    if (pt_idx >= pts_num || box_idx >= boxes_num) return;

    pts += pt_idx * 3;
    rois += box_idx * 7;
    pts_mask += box_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(pts, rois, local_x, local_y);

    pts_mask[0] = -1;
    if (cur_in_flag > 0){
        float local_z = pts[2] - rois[2];
        float dx = rois[3], dy = rois[4], dz = rois[5];

        float x_res = dx / out_x;
        float y_res = dy / out_y;
        float z_res = dz / out_z;

        unsigned int x_idx = int((local_x + dx / 2) / x_res);
        unsigned int y_idx = int((local_y + dy / 2) / y_res);
        unsigned int z_idx = int((local_z + dz / 2) / z_res);

        x_idx = min(max(x_idx, 0), out_x - 1);
        y_idx = min(max(y_idx, 0), out_y - 1);
        z_idx = min(max(z_idx, 0), out_z - 1);

        unsigned int idx_encoding = (x_idx << 16) + (y_idx << 8) + z_idx;
        pts_mask[0] = idx_encoding;
    }
}


__global__ void collect_inside_pts_for_box3d(int boxes_num, int pts_num, int max_pts_each_voxel,
    int out_x, int out_y, int out_z, const int *pts_mask, int *pts_idx_of_voxels){
    // params pts_mask: (N, npoints)  0 or 1
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)

    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (box_idx >= boxes_num) return;

    int max_num_pts = max_pts_each_voxel - 1;  // index 0 is the counter
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel;

    for (int k = 0; k < pts_num; k++){
        if (pts_mask[box_idx * pts_num + k] != -1){
            unsigned int idx_encoding = pts_mask[box_idx * pts_num + k];
            unsigned int x_idx = (idx_encoding >> 16) & 0xFF;
            unsigned int y_idx = (idx_encoding >> 8) & 0xFF;
            unsigned int z_idx = idx_encoding & 0xFF;
            unsigned int base_offset = x_idx * out_y * out_z * max_pts_each_voxel + y_idx * out_z * max_pts_each_voxel + z_idx * max_pts_each_voxel;
            unsigned int cnt = pts_idx_of_voxels[base_offset];
            if (cnt < max_num_pts){
                pts_idx_of_voxels[base_offset + cnt + 1] = k;
                pts_idx_of_voxels[base_offset]++;
            }
#ifdef DEBUG
        printf("collect: pts_%d, idx(%d, %d, %d), idx_encoding=%x\n",
            k, x_idx, y_idx, z_idx, idx_encoding);
#endif

        }
    }
}


__global__ void roiaware_maxpool3d(int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
    int out_y, int out_z, const float *pts_feature, const int *pts_idx_of_voxels, float *pooled_features, int *argmax){
    // params pts_feature: (npoints, C)
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel), index 0 is the counter
    // params pooled_features: (N, out_x, out_y, out_z, C)
    // params argmax: (N, out_x, out_y, out_z, C)

    int box_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels|| x_idx >= out_x || y_idx >= out_y || z_idx >= out_z) return;

#ifdef DEBUG
    printf("src pts_idx_of_voxels: (%p, ), argmax: %p\n", pts_idx_of_voxels, argmax);
#endif

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel + offset_base * max_pts_each_voxel;
    pooled_features += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;
    argmax += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;

    int argmax_idx = -1;
    float max_val = -1e50;

    int total_pts = pts_idx_of_voxels[0];

    for (int k = 1; k <= total_pts; k++){
        if (pts_feature[pts_idx_of_voxels[k] * channels + channel_idx] > max_val){
            max_val = pts_feature[pts_idx_of_voxels[k] * channels + channel_idx];
            argmax_idx = pts_idx_of_voxels[k];
        }
    }

    if (argmax_idx != -1){
        pooled_features[0] = max_val;
    }
    argmax[0] = argmax_idx;

#ifdef DEBUG
    printf("channel_%d idx(%d, %d, %d), argmax_idx=(%d, %.3f), total=%d, after pts_idx: %p, argmax: (%p, %d)\n",
        channel_idx, x_idx, y_idx, z_idx, argmax_idx, max_val, total_pts, pts_idx_of_voxels, argmax, argmax_idx);
#endif
}


__global__ void roiaware_avgpool3d(int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x,
    int out_y, int out_z, const float *pts_feature, const int *pts_idx_of_voxels, float *pooled_features){
    // params pts_feature: (npoints, C)
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel), index 0 is the counter
    // params pooled_features: (N, out_x, out_y, out_z, C)
    // params argmax: (N, out_x, out_y, out_z, C)

    int box_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels|| x_idx >= out_x || y_idx >= out_y || z_idx >= out_z) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel + offset_base * max_pts_each_voxel;
    pooled_features += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;

    float sum_val = 0;
    int total_pts = pts_idx_of_voxels[0];

    for (int k = 1; k <= total_pts; k++){
        sum_val += pts_feature[pts_idx_of_voxels[k] * channels + channel_idx];
    }

    if (total_pts > 0){
        pooled_features[0] = sum_val / total_pts;
    }
}


void roiaware_pool3d_launcher(int boxes_num, int pts_num, int channels, int max_pts_each_voxel, int out_x, int out_y, int out_z,
    const float *rois, const float *pts, const float *pts_feature, int *argmax, int *pts_idx_of_voxels, float *pooled_features, int pool_method){
    // params rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (npoints, 3) [x, y, z]
    // params pts_feature: (npoints, C)
    // params argmax: (N, out_x, out_y, out_z, C)
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
    // params pooled_features: (N, out_x, out_y, out_z, C)
    // params pool_method: 0: max_pool 1: avg_pool

    int *pts_mask = NULL;
    cudaMalloc(&pts_mask, boxes_num * pts_num * sizeof(int));  // (N, M)
    cudaMemset(pts_mask, -1, boxes_num * pts_num * sizeof(int));

    dim3 blocks_mask(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    generate_pts_mask_for_box3d<<<blocks_mask, threads>>>(boxes_num, pts_num, out_x, out_y, out_z, rois, pts, pts_mask);

    // TODO: Merge the collect and pool functions, SS

    dim3 blocks_collect(DIVUP(boxes_num, THREADS_PER_BLOCK));
    collect_inside_pts_for_box3d<<<blocks_collect, threads>>>(boxes_num, pts_num, max_pts_each_voxel,
        out_x, out_y, out_z, pts_mask, pts_idx_of_voxels);

    dim3 blocks_pool(DIVUP(out_x * out_y * out_z, THREADS_PER_BLOCK), channels, boxes_num);
    if (pool_method == 0){
        roiaware_maxpool3d<<<blocks_pool, threads>>>(boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
            pts_feature, pts_idx_of_voxels, pooled_features, argmax);
    }
    else if (pool_method == 1){
        roiaware_avgpool3d<<<blocks_pool, threads>>>(boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
            pts_feature, pts_idx_of_voxels, pooled_features);
    }


    cudaFree(pts_mask);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}


__global__ void roiaware_maxpool3d_backward(int boxes_num, int channels, int out_x, int out_y, int out_z,
    const int *argmax, const float *grad_out, float *grad_in){
    // params argmax: (N, out_x, out_y, out_z, C)
    // params grad_out: (N, out_x, out_y, out_z, C)
    // params grad_in: (npoints, C), return value

    int box_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels|| x_idx >= out_x || y_idx >= out_y || z_idx >= out_z) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    argmax += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;
    grad_out += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;

    if (argmax[0] == -1) return;

    atomicAdd(grad_in + argmax[0] * channels + channel_idx, grad_out[0] * 1);
}


__global__ void roiaware_avgpool3d_backward(int boxes_num, int channels, int out_x, int out_y, int out_z,
    int max_pts_each_voxel, const int *pts_idx_of_voxels, const float *grad_out, float *grad_in){
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
    // params grad_out: (N, out_x, out_y, out_z, C)
    // params grad_in: (npoints, C), return value

    int box_idx = blockIdx.z;
    int channel_idx = blockIdx.y;
    int voxel_idx_flat = blockIdx.x * blockDim.x + threadIdx.x;

    int x_idx = voxel_idx_flat / (out_y * out_z);
    int y_idx = (voxel_idx_flat - x_idx * (out_y * out_z)) / out_z;
    int z_idx = voxel_idx_flat % out_z;
    if (box_idx >= boxes_num || channel_idx >= channels|| x_idx >= out_x || y_idx >= out_y || z_idx >= out_z) return;

    int offset_base = x_idx * out_y * out_z + y_idx * out_z + z_idx;
    pts_idx_of_voxels += box_idx * out_x * out_y * out_z * max_pts_each_voxel + offset_base * max_pts_each_voxel;
    grad_out += box_idx * out_x * out_y * out_z * channels + offset_base * channels + channel_idx;


    int total_pts = pts_idx_of_voxels[0];
    float cur_grad = 1 / fmaxf(float(total_pts), 1.0);
    for (int k = 1; k <= total_pts; k++){
        atomicAdd(grad_in + pts_idx_of_voxels[k] * channels + channel_idx, grad_out[0] * cur_grad);
    }
}


void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y, int out_z, int channels, int max_pts_each_voxel,
    const int *pts_idx_of_voxels, const int *argmax, const float *grad_out, float *grad_in, int pool_method){
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
    // params argmax: (N, out_x, out_y, out_z, C)
    // params grad_out: (N, out_x, out_y, out_z, C)
    // params grad_in: (npoints, C), return value
    // params pool_method: 0: max_pool, 1: avg_pool

    dim3 blocks(DIVUP(out_x * out_y * out_z, THREADS_PER_BLOCK), channels, boxes_num);
    dim3 threads(THREADS_PER_BLOCK);
    if (pool_method == 0){
        roiaware_maxpool3d_backward<<<blocks, threads>>>(
            boxes_num, channels, out_x, out_y, out_z, argmax, grad_out, grad_in
        );
    }
    else if (pool_method == 1){
        roiaware_avgpool3d_backward<<<blocks, threads>>>(
            boxes_num, channels, out_x, out_y, out_z, max_pts_each_voxel, pts_idx_of_voxels, grad_out, grad_in
        );
    }

}


__global__ void points_in_boxes_kernel(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
    // params boxes_idx_of_points: (B, npoints), default -1

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; k++){
        cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
        if (cur_in_flag){
            box_idx_of_points[0] = k;
            break;
        }
    }
}


void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    points_in_boxes_kernel<<<blocks, threads>>>(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}


__global__ void points_in_boxes_bev_kernel(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 2) [x, y] in LiDAR coordinate
    // params boxes_idx_of_points: (B, npoints), default -1

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 2 + pt_idx * 2;
    box_idx_of_points += bs_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; k++){
        cur_in_flag = check_pt_in_box_bev(pts, boxes + k * 7, local_x, local_y);
        if (cur_in_flag){
            box_idx_of_points[0] = k;
            break;
        }
    }
}


void points_in_boxes_bev_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    points_in_boxes_bev_kernel<<<blocks, threads>>>(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

__global__ void bev_in_boxes_kernel(float x_min, float x_max, float y_min, float y_max,
                                     int boxes_num, int batch_size, int x_inds_length, int y_inds_length,
                                     const float *boxes, const float *bev_coords, int *bev_indices){
    /*
    Args:
        boxes: [B, N, 7]
        bev_coords: [X, Y, 2]
        bev_indices: [B, X, Y]
    */

    int bs_idx = blockIdx.x;
    int box_idx = threadIdx.x;
    if (bs_idx >= batch_size || box_idx >= boxes_num) return;

    boxes += bs_idx * boxes_num * 7;
    bev_indices += bs_idx * x_inds_length * y_inds_length;

    float cx = boxes[box_idx * 7 + 0];
    float cy = boxes[box_idx * 7 + 1];
    float dx = boxes[box_idx * 7 + 3];
    float dy = boxes[box_idx * 7 + 4];
    float rz = boxes[box_idx * 7 + 6];
    if (dx == 0 || dy == 0) return;

    const float MARGIN = 0.001;
    float x_length = x_max - x_min;
    float y_length = y_max - y_min;

    float r = dx * 0.5 + dy * 0.5;
    float cosa = cos(-rz), sina = sin(-rz);

    int search_x_min = floor(x_inds_length * ((cx - r - x_min) / x_length));
    if (search_x_min < 0) search_x_min = 0;
    if (search_x_min >= x_inds_length) search_x_min = x_inds_length - 1;
    int search_x_max = ceil(x_inds_length * ((cx + r - x_min) / x_length));
    if (search_x_max < 0) search_x_max = 0;
    if (search_x_max >= x_inds_length) search_x_max = x_inds_length - 1;

    int search_y_min = floor(y_inds_length * ((cy - r - y_min) / y_length));
    if (search_y_min < 0) search_y_min = 0;
    if (search_y_min >= y_inds_length) search_y_min = y_inds_length - 1;
    int search_y_max = ceil(y_inds_length * ((cy + r - y_min) / y_length));
    if (search_y_max < 0) search_y_max = 0;
    if (search_y_max >= y_inds_length) search_y_max = y_inds_length - 1;

    for (int xi = search_x_min; xi <= search_x_max; ++xi){
        for (int yi = search_y_min; yi <= search_y_max; ++yi){
            float x_coords = bev_coords[xi * y_inds_length * 2 + yi * 2 + 0];
            float y_coords = bev_coords[xi * y_inds_length * 2 + yi * 2 + 1];
            float local_x = (x_coords - cx) * cosa + (y_coords - cy) * (-sina);
            float local_y = (x_coords - cx) * sina + (y_coords - cy) * cosa;
            float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
            if (in_flag) bev_indices[xi * y_inds_length + yi] = box_idx;
        }
    }

}

void bev_in_boxes_launcher(float x_min, float x_max, float y_min, float y_max,
                            int boxes_num, int batch_size, int x_inds_length, int y_inds_length,
                            const float *boxes, const float *bev_coords, int *bev_indices){
    cudaError_t err;

    dim3 blocks(batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    bev_in_boxes_kernel<<<blocks, threads>>>(x_min, x_max, y_min, y_max,
        boxes_num, batch_size, x_inds_length, y_inds_length,
        boxes, bev_coords, bev_indices);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}