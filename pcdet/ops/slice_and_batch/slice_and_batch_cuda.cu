#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// Jetson AGX Orin has 16 SMs, each can run 1536 threads

// Each CUDA thread copies a channel of a slice

// input and output has the same number of channels
// input= (1 x C x H x W) output (len(slice_indices) x C x slice_size x slice_size)
// This kernel can only run on a single frame (batch size = 1)
// slice_indices are indexed for HxW

// Each thread is responsible for copying a single channel of a slice
template <typename scalar_t>
__global__ void slice_and_batch_cuda_kernel(
		scalar_t* __restrict__ inp,
		const int64_t inp_H,
		const int64_t inp_W,
		const int64_t* __restrict__ slice_indices,
		const int64_t slice_size,
		scalar_t* __restrict__ outp) {
  // blockIdx.x is the channel index
  // blockDim.x is the number of threads in a block
  // threadIdx.x is the output slice index

  const auto inp_slice_idx = slice_indices[threadIdx.x];
  const auto outp_slice_idx = threadIdx.x;
  const auto channel_idx = blockIdx.x;
  const auto num_channels = gridDim.x;

  scalar_t* slice_bgn = inp + channel_idx * inp_H * inp_W + inp_slice_idx;
  scalar_t* outp_bgn = outp + (outp_slice_idx * num_channels + channel_idx) 
	  * slice_size * slice_size;


  //printf("%d %d %d %f %d\n", blockIdx.x, gridDim.x, threadIdx.x, *slice_bgn, step);

  // copy a channel of a slice
  for(auto i=0; i < slice_size; ++i){
    // copy a row of slice
    for(auto j=0; j < slice_size; ++j){
      outp_bgn[j] = slice_bgn[j];
    }
    slice_bgn += inp_W;
    outp_bgn += slice_size;
  } 
}


void slice_and_batch_cuda_inplace(torch::Tensor inp, torch::Tensor slice_indices,
		const int64_t slice_size, torch::Tensor outp){

  const auto inp_H = inp.size(1);
  const auto inp_W = inp.size(2);
  const auto blocks = inp.size(0); // channel size, for example 64
  const auto threads_per_block = slice_indices.size(0); // number of slices, max 500

  AT_DISPATCH_FLOATING_TYPES(outp.type(), "slice_and_batch_cuda", ([&] {
    slice_and_batch_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
      inp.data<scalar_t>(),
      inp_H,
      inp_W,
      slice_indices.data<int64_t>(),
      slice_size,
      outp.data<scalar_t>());
  }));
}

// inp has size (C x H x W)
// outp has size (len(slice_indices) x C x slice_size x slice_size)
// slice indices are the corner indices (not center)
torch::Tensor slice_and_batch_cuda(torch::Tensor inp, torch::Tensor slice_indices,
                const int64_t slice_size) {

  const auto inp_H = inp.size(1);
  const auto inp_W = inp.size(2);
  const auto blocks = inp.size(0); // channel size, for example 64
  const auto threads_per_block = slice_indices.size(0); // number of slices, max 500
 
  auto tensor_options = torch::TensorOptions()
      .layout(torch::kStrided)
      .dtype(inp.dtype())
      .device(inp.device().type())
      .requires_grad(inp.requires_grad());

  torch::Tensor outp = torch::empty({slice_indices.size(0), blocks, slice_size, slice_size},
      tensor_options);

  AT_DISPATCH_FLOATING_TYPES(outp.type(), "slice_and_batch_cuda", ([&] {
    slice_and_batch_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
      inp.data<scalar_t>(),
      inp_H,
      inp_W,
      slice_indices.data<int64_t>(),
      slice_size,
      outp.data<scalar_t>());
  }));

  return outp;
}
