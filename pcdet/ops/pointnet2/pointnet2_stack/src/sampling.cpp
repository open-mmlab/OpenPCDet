#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;
#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


int farthest_point_sampling_wrapper(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(temp_tensor);
    CHECK_INPUT(idx_tensor);

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    farthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}


int stack_farthest_point_sampling_wrapper(at::Tensor points_tensor,
  at::Tensor temp_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor,
  at::Tensor num_sampled_points_tensor) {

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(temp_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);
    CHECK_INPUT(num_sampled_points_tensor);

    int batch_size = xyz_batch_cnt_tensor.size(0);
    int N = points_tensor.size(0);
    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    int *idx = idx_tensor.data<int>();
    int *num_sampled_points = num_sampled_points_tensor.data<int>();

    stack_farthest_point_sampling_kernel_launcher(N, batch_size, points, temp, xyz_batch_cnt, idx, num_sampled_points);
    return 1;
}