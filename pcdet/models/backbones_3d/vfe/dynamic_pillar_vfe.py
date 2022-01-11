import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        assert len(self.model_cfg.NUM_FILTERS) == 1
        self.num_filters = self.model_cfg.NUM_FILTERS[0]

        self.linear1 = nn.Sequential(
            nn.Linear(num_point_features, self.num_filters//2, bias=False),
            nn.BatchNorm1d(self.num_filters//2, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(self.num_filters, self.num_filters, bias=False),
            nn.BatchNorm1d(self.num_filters, eps=1e-3, momentum=0.01),
            nn.ReLU()
        ) 

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.point_cloud_range = point_cloud_range
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]


    def get_output_feature_dim(self):
        return self.num_filters

    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_xyz = points[:, [1,2,3]].contiguous()
        # points_coords = (points_xyz[:, [0,1]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]
        points_coords_x = (points_xyz[:, 0] - self.point_cloud_range[0]) / self.voxel_x
        points_coords_y = (points_xyz[:, 1] - self.point_cloud_range[1]) / self.voxel_y

        points_coords_x = points_coords_x.floor()
        points_coords_y = points_coords_y.floor()
        points_coords = torch.stack([points_coords_x, points_coords_y], dim=-1)

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        features = self.linear1(features)
        features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        features = torch.cat([features, features_max[unq_inv, :]], dim=1)
        features = self.linear2(features)
        features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict
