import torch
import torch.nn as nn
from .plot_utils import plot_pc, plot_pc_with_gt, map_plot_with_gt, plot_pc_with_gt_threshold, analyze


class PointGather(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.foreground_threshold = model_cfg.FOREGROUND_THRESHOLD
        self.mode = 'train' if self.training else 'test'

    def forward(self, batch_dict, **kwargs):
        """

        Args:
            batch_dict:
                range_features: (B, C, H, W)
                seg_mask:(B, H, W)
                points:(N, batch_idx+xyz+channels)
                ri_indices:(N, batch_idx+indices)
                voxels:(voxel_num, max_point_num, features+indices)
                voxel_coords:(voxel_num, xyz)
            **kwargs:

        Returns:
            filtered points and voxels

        """
        batch_dict = self.foreground_points_voxels_filter_and_feature_gather(batch_dict)
        return batch_dict

    def foreground_points_voxels_filter_and_feature_gather(self, batch_dict):
        range_features = batch_dict['range_features'].permute((0, 2, 3, 1))
        seg_mask = batch_dict['seg_pred']
        batch_size, height, width = batch_dict['seg_pred'].shape
        points = batch_dict['points']
        ri_indices = batch_dict['ri_indices']
        voxels = batch_dict['voxels']
        voxel_coords = batch_dict['voxel_coords']
        foreground_points = []
        foreground_voxels = []
        foreground_voxel_coords = []
        foreground_voxel_num_points = []
        # import pudb
        # pudb.set_trace()
        # analyze(batch_dict)

        for batch_idx in range(batch_size):
            this_range_features = range_features[batch_idx].reshape((height * width, -1))
            cur_seg_mask = seg_mask[batch_idx] >= self.foreground_threshold
            cur_seg_mask = torch.flatten(cur_seg_mask)

            # points
            batch_points_mask = points[:, 0] == batch_idx
            this_points = points[batch_points_mask, :]
            this_ri_indices = ri_indices[batch_points_mask, :]
            this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
            this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
            this_points = this_points[this_points_mask]
            this_points_features = this_range_features[this_ri_indexes]
            this_points_features = this_points_features[this_points_mask]
            this_points = torch.cat((this_points, this_points_features), dim=1)
            foreground_points.append(this_points)

            # voxels
            batch_voxels_mask = voxel_coords[:, 0] == batch_idx
            this_voxels = voxels[batch_voxels_mask]
            this_voxel_coords = voxel_coords[batch_voxels_mask]
            this_voxels_indices = this_voxels[..., -2:]
            this_voxels = this_voxels[..., :-2]
            num_voxels, max_num_points, num_points_features = this_voxels.shape
            # (num_voxels, max_num_points)
            this_voxels_points_indexes = (
                    this_voxels_indices[..., 0] * width + this_voxels_indices[..., 1]).long().flatten()
            # index 0 means empty, but get some value
            this_voxels_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_voxels_points_indexes) * (
                    this_voxels_points_indexes > 0).float()
            this_voxels_points_mask = this_voxels_points_mask.reshape((num_voxels, max_num_points)).long()
            this_voxels_points_features = this_range_features[this_voxels_points_indexes].reshape(
                (num_voxels, max_num_points, -1)).float()
            this_voxels = torch.cat((this_voxels, this_voxels_points_features), dim=-1)
            this_voxels = this_voxels * this_voxels_points_mask.unsqueeze(dim=2)
            # (num_voxels,)
            this_voxel_num_points = this_voxels_points_mask.sum(dim=1)
            this_voxels_mask = this_voxel_num_points > 0
            this_voxels = this_voxels[this_voxels_mask]
            this_voxel_coords = this_voxel_coords[this_voxels_mask]
            this_voxel_num_points = this_voxel_num_points[this_voxels_mask]
            foreground_voxels.append(this_voxels)
            foreground_voxel_coords.append(this_voxel_coords)
            foreground_voxel_num_points.append(this_voxel_num_points)

        foreground_points = torch.cat(foreground_points, dim=0)
        batch_dict['points'] = foreground_points

        foreground_voxels = torch.cat(foreground_voxels, dim=0)
        foreground_voxel_coords = torch.cat(foreground_voxel_coords, dim=0)
        foreground_voxel_num_points = torch.cat(foreground_voxel_num_points, dim=0)
        batch_dict['voxels'] = foreground_voxels
        batch_dict['voxel_coords'] = foreground_voxel_coords
        batch_dict['voxel_num_points'] = foreground_voxel_num_points

        return batch_dict
