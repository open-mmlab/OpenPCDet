#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>

using namespace torch::indexing;
namespace py = pybind11;

// Returns the boolean mask that can be used to filter tile_coords
torch::Tensor point_tile_mask(
        torch::Tensor tile_coords, // [num_points] x*w+y notation
        torch::Tensor chosen_tile_coords // [num_chosen_tiles] x*w+y notation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("point_tile_mask", &point_tile_mask, "Point tile mask CUDA");
}
