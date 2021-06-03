import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import List

from . import pointnet2_stack_cuda as pointnet2
from . import pointnet2_utils

class VoxelQuery(Function):

    @staticmethod
    def forward(ctx, max_range: int, radius: float, nsample: int, xyz: torch.Tensor, \
                    new_xyz: torch.Tensor, new_coords: torch.Tensor, point_indices: torch.Tensor):
        """
        Args:
            ctx:
            max_range: int, max range of voxels to be grouped
            nsample: int, maximum number of features in the balls
            new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert new_coords.is_contiguous()
        assert point_indices.is_contiguous()

        M = new_coords.shape[0]
        B, Z, Y, X = point_indices.shape
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        z_range, y_range, x_range = max_range
        pointnet2.voxel_query_wrapper(M, Z, Y, X, nsample, radius, z_range, y_range, x_range, \
                    new_xyz, xyz, new_coords, point_indices, idx)

        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

voxel_query = VoxelQuery.apply


class VoxelQueryAndGrouping(nn.Module):
    def __init__(self, max_range: int, radius: float, nsample: int):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample

    def forward(self, new_coords: torch.Tensor, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor, voxel2point_indices: torch.Tensor):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0]
        
        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)

        idx1 = idx1.view(batch_size, -1, self.nsample)
        count = 0
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = idx1.view(-1, self.nsample)
        idx1[empty_ball_mask1] = 0

        idx = idx1
        empty_ball_mask = empty_ball_mask1
        
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        # grouped_features: (M1 + M2, C, nsample)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  
        
        return grouped_features, grouped_xyz, empty_ball_mask
