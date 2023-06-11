import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2


class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict
