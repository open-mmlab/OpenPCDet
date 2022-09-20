#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>

using namespace torch::indexing;
namespace py = pybind11;

// ONLY SUPPORTS BATCH SIZE 1 FOR NOW
// First two inputs may have batch size higher than 1
// inp tensor will be in the shape of BxCxHxW where C is channels
std::pair<torch::Tensor, torch::Tensor> slice_and_batch(torch::Tensor inp, torch::Tensor heatmap,
		const int64_t K, const int64_t slice_width, const float score_thres) {
    auto hsw = slice_width/2;
    inp = pad(inp, {hsw, hsw, hsw, hsw});
    heatmap = pad(heatmap, {hsw, hsw, hsw, hsw});
    auto hm_flat = heatmap.flatten(2,3);
    auto scrs_and_indices = hm_flat.topk(K, -1, true, false); // params: K, dim, largest, sorted
    auto& scores = std::get<0>(scrs_and_indices);
    auto& indices= std::get<1>(scrs_and_indices);
    auto s_i = scores.topk(1, 1, true, false); // Take the best class score for each 2D position
    auto max_indices = indices.gather(1, std::get<1>(s_i));
    auto max_scores = scores.gather(1, std::get<1>(s_i));
    auto score_mask = (max_scores > score_thres);
    auto num_slices_per_batch = score_mask.sum(-1); 
   // masked_indices contain the indicies of all slices for all batches
    auto masked_indices = max_indices.masked_select(score_mask).cpu();
    auto num_slices = masked_indices.size(0);
    auto tensor_options = torch::TensorOptions()
        .layout(torch::kStrided)
        .dtype(torch::kFloat32)
        .device(torch::kCUDA)
        .requires_grad(false); // maybe true?
    torch::Tensor outp = torch::empty({num_slices, inp.size(1),
            slice_width, slice_width}, tensor_options);
    // The overhead up here is 1.7 ms
    // We need to pay the overhead of topk anyway
    const auto width = inp.size(3);
    //std::cout << "width:" << width << std::endl; 
    //std::cout << "hsw:" << hsw << std::endl; 
    auto h = masked_indices.div(width, "trunc");
    auto w = masked_indices % width;
    //std::cout << "Type of h: " << h.dtype() << std::endl;
    //std::cout << "Type of w: " << w.dtype() << std::endl;
    auto h_a = h.accessor<int64_t,1>();
    auto w_a = w.accessor<int64_t,1>();
    // This part is the bottleneck
    for(auto i=0; i< num_slices; ++i){
        // BxCxHxW <- BxCxHxW
        //std::cout << "h_a[" << i << "]=" << h_a[i] << std::endl;
        //std::cout << "w_a[" << i << "]=" << w_a[i] << std::endl;
        outp.index_put_({i}, inp.index({"...", Slice(h_a[i]-hsw, h_a[i]+hsw+1), 
                    Slice(w_a[i]-hsw, w_a[i]+hsw+1)}));
    }
    
    return std::make_pair(outp, num_slices_per_batch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("slice_and_batch", &slice_and_batch, "Slice and Batch");
}
