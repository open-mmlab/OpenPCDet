import torch
import torch.nn as nn
from torch.autograd import Function
from . import roiaware_pool3d_cuda


class RoIAwarePool3d(nn.Module):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature,
            self.out_size, self.max_pts_each_voxel, pool_method)


class RoIAwarePool3dFunction(Function):
    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        :param rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate, (x, y, z) is the bottom center of rois
        :param pts: (npoints, 3)
        :param pts_feature: (npoints, C)
        :param out_size: int or tuple, like 7 or (7, 7, 7)
        :param pool_method: 'max' or 'avg'
        :return
            pooled_features: (N, out_x, out_y, out_z, C)
        """

        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)

        return None, None, grad_in, None, None, None


def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 8), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts


def points_in_boxes_cpu(points, boxes):
    """
    :param points: (npoints, 3)
    :param boxes: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center, each box DO NOT overlaps
    :return point_indices: (N, npoints)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices


if __name__ == '__main__':
    pass
