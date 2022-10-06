#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>
#include <vector>

using namespace torch::indexing;
namespace py = pybind11;

// Returns the projected pred_boxes and the mask to get them
//std::vector<at::Tensor> project_past_detections(
std::vector<torch::Tensor> project_past_detections(
        torch::Tensor chosen_tile_coords, // [num_chosen_tiles] x*w+y notation
        torch::Tensor pred_tile_coords, // [num_objects] x*w+y notation
        torch::Tensor pred_boxes, // [num_objects, 9]
        torch::Tensor past_pose_indexes, // [num_objects]
        torch::Tensor past_poses, // [num_past_poses, 14]
        torch::Tensor cur_pose, // [14]
        torch::Tensor past_timestamps, // [num_past_poses]
        long cur_timestamp // [1]
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_past_detections", &project_past_detections, "Detection projector CUDA");
}
