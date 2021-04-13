import torch
import torch.nn as nn
import torch.nn.functional as F
import pudb


class PointGather(nn.Module):
    def __init__(self, model_cfg, point_cloud_range, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.foreground_threshold = model_cfg.FOREGROUND_THRESHOLD

    def forward(self, batch_dict, **kwargs):
        # TODO: point features
        pudb.set_trace()
        batch_dict = self.foreground_points_filter_and_feature_gather(batch_dict)
        batch_dict = self.transform_points_to_voxels(batch_dict, self.model_cfg)
        return batch_dict

    def foreground_points_filter_and_feature_gather(self, batch_dict):
        range_features = batch_dict['range_features'].permute((0, 2, 3, 1))
        seg_mask = batch_dict['seg_pred']
        batch_size, height, width = batch_dict['seg_pred'].shape
        points = batch_dict['points']
        ri_indices = batch_dict['ri_indices']
        foreground_points = []
        for batch_idx in range(batch_size):
            this_range_features = range_features[batch_idx].reshape((height * width, -1))
            cur_seg_mask = seg_mask[batch_idx] >= self.foreground_threshold
            cur_seg_mask = torch.flatten(cur_seg_mask)
            batch_mask = points[:, 0] == batch_idx
            this_points = points[batch_mask, :]
            this_ri_indices = ri_indices[batch_mask, :]
            this_ri_indexes = (this_ri_indices[:, 0] * width + this_ri_indices[:, 1]).long()
            this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
            this_points = this_points[this_points_mask]
            this_points_features = this_range_features[this_ri_indexes]
            this_points_features = this_points_features[this_points_mask]
            this_points = torch.cat((this_points, this_points_features), dim=1)
            foreground_points.append(this_points)

        foreground_points = torch.cat(foreground_points, dim=0)
        batch_dict['points'] = foreground_points
        return batch_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            from spconv.utils import VoxelGenerator

        voxel_generator = VoxelGenerator(
            voxel_size=config.VOXEL_SIZE,
            point_cloud_range=self.point_cloud_range,
            max_num_points=config.MAX_POINTS_PER_VOXEL,
            max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
        )
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
        self.grid_size = torch.round(grid_size).int()
        self.voxel_size = config.VOXEL_SIZE

        points = data_dict['points']
        batch_size = points[:, 0].max().int().item() + 1
        voxels_list, voxel_coords_list, voxel_num_points_list = [], [], []
        for batch_idx in range(batch_size):
            batch_mask = points[:, 0] == batch_idx
            this_points = points[batch_mask, :]
            voxel_output = voxel_generator.generate(this_points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
            coordinates = F.pad(coordinates, (1, 0), mode='constant', value=batch_idx)
            voxels_list.append(voxels)
            voxel_coords_list.append(coordinates)
            voxel_num_points_list.append(num_points)

        data_dict['voxels'] = torch.cat(voxels_list, dim=0)
        data_dict['voxel_coords'] = torch.cat(voxel_coords_list, dim=0)
        data_dict['voxel_num_points'] = torch.cat(voxel_num_points_list, dim=0)
        return data_dict
