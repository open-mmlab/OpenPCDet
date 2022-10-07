#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
// Each CUDA thread copies a channel of a slice
// input and output has the same number of channels
// This kernel can only run on a single frame (batch size = 1)

// This kernel assumes input and heatmap has same H and W
template <typename scalar_t>
__global__ void slice_and_batch_cuda_kernel_v2(
		scalar_t* __restrict__ inp,
		scalar_t* __restrict__ heatmap,
		const int64_t hm_C,
		const int64_t slice_size,
		scalar_t* __restrict__ outp,
		int32_t* slice_idx,
		const int32_t slice_limit,
		const scalar_t score_threshold){
  const auto inp_C = gridDim.x;
  const auto hm_H = gridDim.y;
  const auto hm_W = blockDim.x;
  const auto c_idx = blockIdx.x;
  const auto h_idx = blockIdx.y;
  const auto w_idx = threadIdx.x;

  auto max_channel_score = 0.;

  // Computation here is redundant in channel dimension but no big deal I think
  const auto hm_HxW = hm_H * hm_W;
  auto idx = h_idx * hm_W + w_idx;
  for(auto c=0; c < hm_C; ++c){
    max_channel_score = max(max_channel_score, *(heatmap + idx));
    idx += hm_HxW;
  }

  if(max_channel_score < score_threshold)
    return;
  
  // atomically acquire slice index and start copying the slice to outp
  auto slice_ch_idx = atomicAdd(slice_idx, 1);
  if(slice_ch_idx >= slice_limit){
    atomicSub(slice_idx, 1);
    return;
  }

  auto half_slice_size = slice_size/2;
  scalar_t* slice_bgn = inp + (c_idx * hm_HxW) + 
      (hm_H - half_slice_size) * hm_W + (w_idx - half_slice_size);
  scalar_t* outp_bgn = outp + slice_ch_idx * slice_size * slice_size;

  for(auto h=0; h < slice_size; ++h){
    for(auto w=0; w < slice_size; ++w){
      outp_bgn[w] = slice_bgn[w];
    }
    slice_bgn += hm_W;
    outp_bgn += slice_size;
  }
}

// heatmap has size (C_1 x H x W)
// inp has size (C_2 x H x W)
// outp has size (len(slice_indices) x C x slice_size x slice_size)
// slice indices are the corner indices (not center)
torch::Tensor slice_and_batch_v2(torch::Tensor inp, torch::Tensor heatmap,
                const int64_t slice_size, const float score_threshold,
	       	torch::Tensor outp) {
  const auto inp_C = inp.size(0);
  const auto hm_C = heatmap.size(0);
  const auto hm_H = heatmap.size(1);
  const auto hm_W = heatmap.size(2);
  const dim3 blocks(inp_C, hm_H);
  const auto threads_per_block = hm_W;
  
  auto tensor_options = torch::TensorOptions().dtype(torch::kInt32)
        .device(torch::kCUDA).requires_grad(false);
  torch::Tensor num_slices = torch::zeros(1, tensor_options);

  AT_DISPATCH_FLOATING_TYPES(outp.type(), "slice_and_batch_cuda_v2", ([&] {
    slice_and_batch_cuda_kernel_v2<scalar_t><<<blocks, threads_per_block>>>(
      inp.data<scalar_t>(),
      heatmap.data<scalar_t>(),
      hm_C,
      slice_size,
      outp.data<scalar_t>(),
      num_slices.data<int32_t>(),
      outp.size(0),
      static_cast<scalar_t>(score_threshold));
  }));

  return num_slices / inp.size(0); // divide by num channels
}
