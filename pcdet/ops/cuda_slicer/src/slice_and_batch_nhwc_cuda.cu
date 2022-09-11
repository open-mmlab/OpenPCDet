#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cassert>
#include <iostream>

// inds is two dimensionals, (num_slices, 3). 3 is batch_id,x,y
template <typename scalar_t>
__global__ void slice_and_batch_nhwc_kernel(
        const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> inp,
        const torch::PackedTensorAccessor32<int16_t,2,torch::RestrictPtrTraits> inds,
        torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> outp){
  // blockDim.x is the number of threads in a block

  const auto slice_idx = blockIdx.x;
  const auto C_per_thread = inp.size(3) / blockDim.z;

  const auto slice_h = threadIdx.x;
  const auto slice_w = threadIdx.y;
  const auto c_offset = threadIdx.z * C_per_thread;

  const auto n = inds[slice_idx][0];
  const auto h = inds[slice_idx][1] + slice_h;
  const auto w = inds[slice_idx][2] + slice_w;

  for(auto c=0; c < C_per_thread; ++c){
    outp[slice_idx][c_offset + c][slice_h][slice_w] = inp[n][h][w][c_offset + c];
  }

}


// inp has size (N x H x W x C)
// outp has size (len(slice_indices) x C x slice_size x slice_size)
// slice indices are the corner indices (not center)
torch::Tensor slice_and_batch_nhwc(
        torch::Tensor inp,
        torch::Tensor slice_indices,
        const int64_t slice_size) 
{
  // Each block is going to read a slice and write it
  const auto max_threads_per_block = 256;
  const auto C = inp.size(3);
  auto num_active_threads = slice_size * slice_size * C;
  uint32_t C_per_thread = 1;
  while(num_active_threads > max_threads_per_block){
    num_active_threads /= 2;
    C_per_thread *= 2;
  }
  // If conditions doesn't hold, error!
  assert(C % C_per_thread == 0);

  const auto num_slices = slice_indices.size(0);
  const dim3 grid_dims(num_slices);
  dim3 block_dims(slice_size, slice_size, C / C_per_thread);
 
  auto tensor_options = torch::TensorOptions()
      .layout(torch::kStrided)
      .dtype(inp.dtype())
      .device(inp.device().type())
      .requires_grad(inp.requires_grad());

  torch::Tensor outp = torch::empty({num_slices, C, slice_size, slice_size},
      tensor_options);

  AT_DISPATCH_FLOATING_TYPES(outp.type(), "slice_and_batch_nhwc", ([&] {
    slice_and_batch_nhwc_kernel<scalar_t><<<grid_dims, block_dims>>>(
      inp.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      slice_indices.packed_accessor32<int16_t,2,torch::RestrictPtrTraits>(),
      outp.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return outp;
}
