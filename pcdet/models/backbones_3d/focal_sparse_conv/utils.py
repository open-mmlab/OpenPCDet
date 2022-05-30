import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).to(index.device)

        index = index.view(*view)
        ones = 1.

        if isinstance(index, Variable):
            ones = Variable(torch.Tensor(index.size()).fill_(1).to(index.device))
            mask = Variable(mask, volatile=index.volatile)

        return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

def sort_by_indices(features, indices, features_add=None):
    """
        To sort the sparse features with its indices in a convenient manner.
        Args:
            features: [N, C], sparse features
            indices: [N, 4], indices of sparse features
            features_add: [N, C], additional features to sort
    """
    idx = indices[:, 1:]
    idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() + idx.select(1, 1) * idx[:, 2].max() + idx.select(1, 2)
    _, ind = idx_sum.sort()
    features = features[ind]
    indices = indices[ind]
    if not features_add is None:
        features_add = features_add[ind]
    return features, indices, features_add

def check_repeat(features, indices, features_add=None, sort_first=True, flip_first=True):
    """
        Check that whether there are replicate indices in the sparse features, 
        remove the replicate features if any.
    """
    if sort_first:
        features, indices, features_add = sort_by_indices(features, indices, features_add)

    if flip_first:
        features, indices = features.flip([0]), indices.flip([0])

    if not features_add is None:
        features_add=features_add.flip([0])

    idx = indices[:, 1:].int()
    idx_sum = torch.add(torch.add(idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max(), idx.select(1, 1) * idx[:, 2].max()), idx.select(1, 2))
    _unique, inverse, counts = torch.unique_consecutive(idx_sum, return_inverse=True, return_counts=True, dim=0)
    
    if _unique.shape[0] < indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
        features_new.index_add_(0, inverse.long(), features)
        features = features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        indices = indices[perm_].int()

        if not features_add is None:
            features_add_new = torch.zeros((_unique.shape[0],), device=features_add.device)
            features_add_new.index_add_(0, inverse.long(), features_add)
            features_add = features_add_new / counts
    return features, indices, features_add


def split_voxels(x, b, imps_3d, voxels_3d, kernel_offsets, mask_multi=True, topk=True, threshold=0.5):
    """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            x: [N, C], input sparse features
            b: int, batch size id
            imps_3d: [N, kernelsize**3], the prediced importance values
            voxels_3d: [N, 3], the 3d positions of voxel centers 
            kernel_offsets: [kernelsize**3, 3], the offset coords in an kernel
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
    """
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    mask_voxel = imps_3d[batch_index, -1].sigmoid()
    mask_kernel = imps_3d[batch_index, :-1].sigmoid()

    if mask_multi:
        features_ori *= mask_voxel.unsqueeze(-1)

    if topk:
        _, indices = mask_voxel.sort(descending=True)
        indices_fore = indices[:int(mask_voxel.shape[0]*threshold)]
        indices_back = indices[int(mask_voxel.shape[0]*threshold):]
    else:
        indices_fore = mask_voxel > threshold
        indices_back = mask_voxel <= threshold

    features_fore = features_ori[indices_fore]
    coords_fore = indices_ori[indices_fore]

    mask_kernel_fore = mask_kernel[indices_fore]
    mask_kernel_bool = mask_kernel_fore>=threshold
    voxel_kerels_imp = kernel_offsets.unsqueeze(0).repeat(mask_kernel_bool.shape[0],1, 1)
    mask_kernel_fore = mask_kernel[indices_fore][mask_kernel_bool]
    indices_fore_kernels = coords_fore[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
    indices_with_imp = indices_fore_kernels + voxel_kerels_imp
    selected_indices = indices_with_imp[mask_kernel_bool]
    spatial_indices = (selected_indices[:, 0] >0) * (selected_indices[:, 1] >0) * (selected_indices[:, 2] >0)  * \
                        (selected_indices[:, 0] < x.spatial_shape[0]) * (selected_indices[:, 1] < x.spatial_shape[1]) * (selected_indices[:, 2] < x.spatial_shape[2])
    selected_indices = selected_indices[spatial_indices]
    mask_kernel_fore = mask_kernel_fore[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_fore.device)*b, selected_indices], dim=1)

    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_fore.device)

    features_fore_cat = torch.cat([features_fore, selected_features], dim=0)
    coords_fore = torch.cat([coords_fore, selected_indices], dim=0)
    mask_kernel_fore = torch.cat([torch.ones(features_fore.shape[0], device=features_fore.device), mask_kernel_fore], dim=0)

    features_fore, coords_fore, mask_kernel_fore = check_repeat(features_fore_cat, coords_fore, features_add=mask_kernel_fore)

    features_back = features_ori[indices_back]
    coords_back = indices_ori[indices_back]

    return features_fore, coords_fore, features_back, coords_back, mask_kernel_fore
