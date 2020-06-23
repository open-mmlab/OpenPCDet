#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_stack, "ball_query_wrapper_stack");

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper, "furthest_point_sampling_wrapper");

    m.def("group_points_wrapper", &group_points_wrapper_stack, "group_points_wrapper_stack");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_stack, "group_points_grad_wrapper_stack");
}
