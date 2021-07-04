/*
Center assignments
Written by Jiageng Mao
*/

#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

__device__ float limit_period(float val, float offset, float period){
    float rval = val - floor(val / period + offset) * period;
    return rval;
}

__device__ float gaussian_radius(float height, float width, float min_overlap){
    float a1 = 1;
    float b1 = (height + width);
    float c1 = width * height * (1 - min_overlap) / (1 + min_overlap);
    float sq1 = sqrt(b1 * b1 - 4 * a1 * c1);
    float r1 = (b1 + sq1) / 2;

    float a2 = 4;
    float b2 = 2 * (height + width);
    float c2 = (1 - min_overlap) * width * height;
    float sq2 = sqrt(b2 * b2 - 4 * a2 * c2);
    float r2 = (b2 + sq2) / 2;

    float a3 = 4 * min_overlap;
    float b3 = -2 * min_overlap * (height + width);
    float c3 = (min_overlap - 1) * width * height;
    float sq3 = sqrt(b3 * b3 - 4 * a3 * c3);
    float r3 = (b3 + sq3) / 2;
    return min(min(r1, r2), r3);
}

__global__ void draw_center_kernel(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size, int min_radius,
                                    float voxel_x, float voxel_y, float range_x, float range_y, float out_factor, float gaussian_overlap,
                                    const float *gt_boxes, float *heatmap, int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt){

    /*
        Args:
            gt_boxes: (B, max_boxes, 8 or 10) with class labels
            heatmap: (B, num_cls, H, W)
            gt_ind: (B, num_cls, max_objs)
            gt_mask: (B, num_cls, max_objs)
            gt_cat: (B, num_cls, max_objs)
            gt_box_encoding: (B, num_cls, max_objs, code_size) sin/cos
            gt_cnt: (B, num_cls)
    */
    int bs_idx = blockIdx.x;
    int box_idx = threadIdx.x;
    if (bs_idx >= batch_size || box_idx >= max_boxes) return;

    // move pointer
    gt_boxes += bs_idx * max_boxes * code_size;
    heatmap += bs_idx * num_cls * H * W;
    gt_ind += bs_idx * num_cls * max_objs;
    gt_mask += bs_idx * num_cls * max_objs;
    gt_cat += bs_idx * num_cls * max_objs;
    gt_box_encoding += bs_idx * num_cls * max_objs * code_size;
    gt_cnt += bs_idx * num_cls;

    // gt box parameters
    float x = gt_boxes[box_idx * code_size + 0];
    float y = gt_boxes[box_idx * code_size + 1];
    float z = gt_boxes[box_idx * code_size + 2];
    float dx = gt_boxes[box_idx * code_size + 3];
    float dy = gt_boxes[box_idx * code_size + 4];
    float dz = gt_boxes[box_idx * code_size + 5];
    // origin dx/dy/dz is for box_encodings
    float origin_dx = gt_boxes[box_idx * code_size + 3];
    float origin_dy = gt_boxes[box_idx * code_size + 4];
    float origin_dz = gt_boxes[box_idx * code_size + 5];
    float rot = gt_boxes[box_idx * code_size + 6];
    float vel_x = 0;
    float vel_y = 0;
    float cls = 0;
    if (code_size == 10) {
        vel_x = gt_boxes[box_idx * code_size + 7];
        vel_y = gt_boxes[box_idx * code_size + 8];
        cls = gt_boxes[box_idx * code_size + 9];
    } else if (code_size == 8) {
        cls = gt_boxes[box_idx * code_size + 7];
    } else {
        return;
    }

    // box not defined
    if (dx == 0 || dy == 0 || dz == 0) return;

    // cls begin from 1
    int cls_idx = (int) cls - 1;
    heatmap += cls_idx * H * W;
    gt_ind += cls_idx * max_objs;
    gt_mask += cls_idx * max_objs;
    gt_cat += cls_idx * max_objs;
    gt_box_encoding += cls_idx * max_objs * code_size;
    gt_cnt += cls_idx;

    // transform to bev map coords
    float offset = 0.5;
    float period = 6.283185307179586;
    rot = limit_period(rot, offset, period);
    dx = dx / voxel_x / out_factor;
    dy = dy / voxel_y / out_factor;
    float radius = gaussian_radius(dy, dx, gaussian_overlap);
    int radius_int = max(min_radius, (int) radius);
    float coor_x = (x - range_x) / voxel_x / out_factor;
    float coor_y = (y - range_y) / voxel_y / out_factor;
    int coor_x_int = (int) coor_x;
    int coor_y_int = (int) coor_y;
    if (coor_x_int >= W || coor_x_int < 0) return;
    if (coor_y_int >= H || coor_y_int < 0) return;

    // draw gaussian map
    float div_factor = 6.0;
    float sigma = (2 * radius_int + 1) / div_factor;
    for (int scan_y = -radius_int; scan_y < radius_int + 1; scan_y++){
        if (coor_y_int + scan_y < 0 || coor_y_int + scan_y >= H) continue;
        for (int scan_x = -radius_int; scan_x < radius_int + 1; scan_x++){
            if (coor_x_int + scan_x < 0 || coor_x_int + scan_x >= W) continue;
            float weight = exp(-(scan_x * scan_x + scan_y * scan_y) / (2 * sigma * sigma)); // force convert float sigma
            float eps = 0.0000001;
            if (weight < eps) weight = 0;
            float *w_addr = heatmap + (coor_y_int + scan_y) * W + (coor_x_int + scan_x);
            float old_weight = atomicExch(w_addr, weight);
            if (old_weight > weight) weight = atomicExch(w_addr, old_weight);
        }
    }
    int obj_idx = atomicAdd(gt_cnt, 1);
    if (obj_idx >= max_objs) return;
    gt_ind[obj_idx] = coor_y_int * W + coor_x_int;
    gt_mask[obj_idx] = 1;
    gt_cat[obj_idx] = cls_idx + 1; // begin from 1
    gt_box_encoding[obj_idx * code_size + 0] = coor_x - coor_x_int;
    gt_box_encoding[obj_idx * code_size + 1] = coor_y - coor_y_int;
    gt_box_encoding[obj_idx * code_size + 2] = z;
    gt_box_encoding[obj_idx * code_size + 3] = origin_dx;
    gt_box_encoding[obj_idx * code_size + 4] = origin_dy;
    gt_box_encoding[obj_idx * code_size + 5] = origin_dz;
    gt_box_encoding[obj_idx * code_size + 6] = sin(rot);
    gt_box_encoding[obj_idx * code_size + 7] = cos(rot);
    if (code_size == 10) {
        gt_box_encoding[obj_idx * code_size + 8] = vel_x;
        gt_box_encoding[obj_idx * code_size + 9] = vel_y;
    }
    return;
}

void draw_center_kernel_launcher(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size, int min_radius,
                                    float voxel_x, float voxel_y, float range_x, float range_y, float out_factor, float gaussian_overlap,
                                    const float *gt_boxes, float *heatmap, int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt){
    cudaError_t err;

    dim3 blocks(batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    draw_center_kernel<<<blocks, threads>>>(batch_size, max_boxes, max_objs, num_cls, H, W, code_size, min_radius,
                                            voxel_x, voxel_y, range_x, range_y, out_factor, gaussian_overlap,
                                            gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}