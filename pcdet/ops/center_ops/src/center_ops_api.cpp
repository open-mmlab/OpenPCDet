#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int draw_center_gpu(at::Tensor gt_boxes_tensor, at::Tensor heatmap_tensor, at::Tensor gt_ind_tensor,
                        at::Tensor gt_mask_tensor, at::Tensor gt_cat_tensor,
                        at::Tensor gt_box_encoding_tensor, at::Tensor gt_cnt_tensor,
                        int min_radius, float voxel_x, float voxel_y, float range_x, float range_y,
                        float out_factor, float gaussian_overlap);
int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou);
int center_rotate_nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);
int center_rotate_nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("draw_center_gpu", &draw_center_gpu, "centerpoint assignment creation");
  m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu, "oriented boxes overlap");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
  m.def("center_rotate_nms_gpu", &center_rotate_nms_gpu, "oriented nms gpu");
  m.def("center_rotate_nms_normal_gpu", &center_rotate_nms_normal_gpu, "nms gpu");
}