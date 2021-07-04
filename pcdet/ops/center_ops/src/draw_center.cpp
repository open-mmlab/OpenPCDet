/*
Center assignments
Written by Jiageng Mao
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>

void draw_center_kernel_launcher(int batch_size, int max_boxes, int max_objs, int num_cls, int H, int W, int code_size, int min_radius,
                                    float voxel_x, float voxel_y, float range_x, float range_y, float out_factor, float gaussian_overlap,
                                    const float *gt_boxes, float *heatmap, int *gt_ind, int *gt_mask, int *gt_cat, float *gt_box_encoding, int *gt_cnt);

int draw_center_gpu(at::Tensor gt_boxes_tensor, at::Tensor heatmap_tensor, at::Tensor gt_ind_tensor,
                        at::Tensor gt_mask_tensor, at::Tensor gt_cat_tensor,
                        at::Tensor gt_box_encoding_tensor, at::Tensor gt_cnt_tensor,
                        int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                        float out_factor, float gaussian_overlap){
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
    int batch_size = gt_boxes_tensor.size(0);
    int max_boxes = gt_boxes_tensor.size(1);
    int code_size = gt_boxes_tensor.size(2);
    int num_cls = heatmap_tensor.size(1);
    int H = heatmap_tensor.size(2);
    int W = heatmap_tensor.size(3);
    int max_objs = gt_ind_tensor.size(2);

    const float *gt_boxes = gt_boxes_tensor.data<float>();
    float *heatmap = heatmap_tensor.data<float>();
    int *gt_ind = gt_ind_tensor.data<int>();
    int *gt_mask = gt_mask_tensor.data<int>();
    int *gt_cat = gt_cat_tensor.data<int>();
    float *gt_box_encoding = gt_box_encoding_tensor.data<float>();
    int *gt_cnt = gt_cnt_tensor.data<int>();

    draw_center_kernel_launcher(batch_size, max_boxes, max_objs, num_cls, H, W, code_size, min_radius,
                                    voxel_x, voxel_y, range_x, range_y, out_factor, gaussian_overlap,
                                    gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt);

    return 1;
}
