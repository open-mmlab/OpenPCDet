/*
RoI-aware point cloud feature pooling
Reference paper:  https://arxiv.org/abs/1907.03670
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>


//#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
//#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
//#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void roiaware_pool3d_launcher(int boxes_num, int pts_num, int channels, int max_pts_each_voxel,
    int out_x, int out_y, int out_z, const float *rois, const float *pts, const float *pts_feature,
    int *argmax, int *pts_idx_of_voxels, float *pooled_features, int pool_method);

void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y, int out_z, int channels, int max_pts_each_voxel,
    const int *pts_idx_of_voxels, const int *argmax, const float *grad_out, float *grad_in, int pool_method);

void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

void points_in_boxes_bev_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

void bev_in_boxes_launcher(float x_min, float x_max, float y_min, float y_max,
    int boxes_num, int batch_size, int x_inds_length, int y_inds_length,
    const float *boxes, const float *bev_coords, int *bev_indices);

int roiaware_pool3d_gpu(at::Tensor rois, at::Tensor pts, at::Tensor pts_feature, at::Tensor argmax,
    at::Tensor pts_idx_of_voxels, at::Tensor pooled_features, int pool_method){
    // params rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (npoints, 3) [x, y, z]
    // params pts_feature: (npoints, C)
    // params argmax: (N, out_x, out_y, out_z, C)
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
    // params pooled_features: (N, out_x, out_y, out_z, C)
    // params pool_method: 0: max_pool 1: avg_pool

//    CHECK_INPUT(rois);
//    CHECK_INPUT(pts);
//    CHECK_INPUT(pts_feature);
//    CHECK_INPUT(argmax);
//    CHECK_INPUT(pts_idx_of_voxels);
//    CHECK_INPUT(pooled_features);

    int boxes_num = rois.size(0);
    int pts_num = pts.size(0);
    int channels = pts_feature.size(1);
    int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
    int out_x = pts_idx_of_voxels.size(1);
    int out_y = pts_idx_of_voxels.size(2);
    int out_z = pts_idx_of_voxels.size(3);
    assert ((out_x < 256) && (out_y < 256) && (out_z < 256));  // we encode index with 8bit

    const float *rois_data = rois.data<float>();
    const float *pts_data = pts.data<float>();
    const float *pts_feature_data = pts_feature.data<float>();
    int *argmax_data = argmax.data<int>();
    int *pts_idx_of_voxels_data = pts_idx_of_voxels.data<int>();
    float *pooled_features_data = pooled_features.data<float>();

    roiaware_pool3d_launcher(boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
        rois_data, pts_data, pts_feature_data, argmax_data, pts_idx_of_voxels_data, pooled_features_data, pool_method);

    return 1;
}

int roiaware_pool3d_gpu_backward(at::Tensor pts_idx_of_voxels, at::Tensor argmax, at::Tensor grad_out, at::Tensor grad_in, int pool_method){
    // params pts_idx_of_voxels: (N, out_x, out_y, out_z, max_pts_each_voxel)
    // params argmax: (N, out_x, out_y, out_z, C)
    // params grad_out: (N, out_x, out_y, out_z, C)
    // params grad_in: (npoints, C), return value
    // params pool_method: 0: max_pool 1: avg_pool

//    CHECK_INPUT(pts_idx_of_voxels);
//    CHECK_INPUT(argmax);
//    CHECK_INPUT(grad_out);
//    CHECK_INPUT(grad_in);

    int boxes_num = pts_idx_of_voxels.size(0);
    int out_x = pts_idx_of_voxels.size(1);
    int out_y = pts_idx_of_voxels.size(2);
    int out_z = pts_idx_of_voxels.size(3);
    int max_pts_each_voxel = pts_idx_of_voxels.size(4);  // index 0 is the counter
    int channels = grad_out.size(4);

    const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data<int>();
    const int *argmax_data = argmax.data<int>();
    const float *grad_out_data = grad_out.data<float>();
    float *grad_in_data = grad_in.data<float>();

    roiaware_pool3d_backward_launcher(boxes_num, out_x, out_y, out_z, channels, max_pts_each_voxel,
        pts_idx_of_voxels_data, argmax_data, grad_out_data, grad_in_data, pool_method);

    return 1;
}

int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

int points_in_boxes_bev_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 2) [x, y]
    // params boxes_idx_of_points: (B, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_bev_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}


inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


inline int check_pt_in_box3d_cpu(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    const float MARGIN = 1e-2;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor pts_indices_tensor){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    // params pts: (num_points, 3) [x, y, z]
    // params pts_indices: (N, num_points)

//    CHECK_CONTIGUOUS(boxes_tensor);
//    CHECK_CONTIGUOUS(pts_tensor);
//    CHECK_CONTIGUOUS(pts_indices_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *pts_indices = pts_indices_tensor.data<int>();

    float local_x = 0, local_y = 0;
    for (int i = 0; i < boxes_num; i++){
        for (int j = 0; j < pts_num; j++){
            int cur_in_flag = check_pt_in_box3d_cpu(pts + j * 3, boxes + i * 7, local_x, local_y);
            pts_indices[i * pts_num + j] = cur_in_flag;
        }
    }

    return 1;
}


int bev_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor bev_tensor, at::Tensor bev_coords_tensor, float x_min, float x_max, float y_min, float y_max){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    // params bev: (X, Y)
    // params bev_coords: (X, Y, 2)

    int boxes_num = boxes_tensor.size(0);
    int x_inds_length = bev_tensor.size(0);
    int y_inds_length = bev_tensor.size(1);

    const float MARGIN = 0.04;
    float x_length = x_max - x_min;
    float y_length = y_max - y_min;

    const float *boxes = boxes_tensor.data<float>();
    const float *bev_coords = bev_coords_tensor.data<float>();
    int *bev = bev_tensor.data<int>();

    for (int i = 0; i < boxes_num; i++){
        float cx = boxes[i * 7 + 0];
        float cy = boxes[i * 7 + 1];
        float dx = boxes[i * 7 + 3];
        float dy = boxes[i * 7 + 4];
        float rz = boxes[i * 7 + 6];
        if (dx == 0 || dy == 0) continue;
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
                if (in_flag) bev[xi * y_inds_length + yi] = i;
            }
        }

    }

    return 1;
}


int bev_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor bev_coords_tensor, at::Tensor bev_indices_tensor,
    float x_min, float x_max, float y_min, float y_max){
    /*
    Args:
        boxes: [B, N, 7]
        bev_coords: [X, Y, 2]
        bev_indices: [B, X, Y]
    */

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int x_inds_length = bev_indices_tensor.size(1);
    int y_inds_length = bev_indices_tensor.size(2);

    const float *boxes = boxes_tensor.data<float>();
    const float *bev_coords = bev_coords_tensor.data<float>();
    int *bev_indices = bev_indices_tensor.data<int>();

    bev_in_boxes_launcher(x_min, x_max, y_min, y_max,
        boxes_num, batch_size, x_inds_length, y_inds_length,
        boxes, bev_coords, bev_indices);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &roiaware_pool3d_gpu, "roiaware pool3d forward (CUDA)");
    m.def("backward", &roiaware_pool3d_gpu_backward, "roiaware pool3d backward (CUDA)");
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");
    m.def("points_in_boxes_bev_gpu", &points_in_boxes_bev_gpu, "points_in_boxes_bev_gpu forward (CUDA)");
    m.def("bev_in_boxes_cpu", &bev_in_boxes_cpu, "boxes cover bev map CPU");
    m.def("bev_in_boxes_gpu", &bev_in_boxes_gpu, "boxes cover bev map CUDA");
}
