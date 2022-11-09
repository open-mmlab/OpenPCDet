from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


def build_local_aggregation_module(input_channels, config):
    local_aggregation_name = config.get('NAME', 'StackSAModuleMSG')

    if local_aggregation_name == 'StackSAModuleMSG':
        mlps = config.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]
        cur_layer = StackSAModuleMSG(
            radii=config.POOL_RADIUS, nsamples=config.NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool',
        )
        num_c_out = sum([x[-1] for x in mlps])
    elif local_aggregation_name == 'VectorPoolAggregationModuleMSG':
        cur_layer = VectorPoolAggregationModuleMSG(input_channels=input_channels, config=config)
        num_c_out = config.MSG_POST_MLPS[-1]
    else:
        raise NotImplementedError

    return cur_layer, num_c_out


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolLocalInterpolateModule(nn.Module):
    def __init__(self, mlp, num_voxels, max_neighbour_distance, nsample, neighbor_type, use_xyz=True,
                 neighbour_distance_multiplier=1.0, xyz_encoding_type='concat'):
        """
        Args:
            mlp:
            num_voxels:
            max_neighbour_distance:
            neighbor_type: 1: ball, others: cube
            nsample: find all (-1), find limited number(>0)
            use_xyz:
            neighbour_distance_multiplier:
            xyz_encoding_type:
        """
        super().__init__()
        self.num_voxels = num_voxels  # [num_grid_x, num_grid_y, num_grid_z]: number of grids in each local area centered at new_xyz
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz
        self.xyz_encoding_type = xyz_encoding_type

        if mlp is not None:
            if self.use_xyz:
                mlp[0] += 9 if self.xyz_encoding_type == 'concat' else 0
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp[k + 1]),
                    nn.ReLU()
                ])
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None

        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz, support_features, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            support_features: (N1 + N2 ..., C) point-wise features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        with torch.no_grad():
            dist, idx, num_avg_length_of_neighbor_idxs = pointnet2_utils.three_nn_for_vector_pool_by_two_step(
                support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                self.max_neighbour_distance, self.nsample, self.neighbor_type,
                self.num_avg_length_of_neighbor_idxs, self.num_total_grids, self.neighbor_distance_multiplier
            )
        self.num_avg_length_of_neighbor_idxs = max(self.num_avg_length_of_neighbor_idxs, num_avg_length_of_neighbor_idxs.item())

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-8)

        empty_mask = (idx.view(-1, 3)[:, 0] == -1)
        idx.view(-1, 3)[empty_mask] = 0

        interpolated_feats = pointnet2_utils.three_interpolate(support_features, idx.view(-1, 3), weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(idx.shape[0], idx.shape[1], -1)  # (M1 + M2 ..., num_total_grids, C)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(-1, 3, 3)  # ( (M1 + M2 ...)*num_total_grids, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) - near_known_xyz).view(-1, idx.shape[1], 9)
            if self.xyz_encoding_type == 'concat':
                interpolated_feats = torch.cat((interpolated_feats, local_xyz), dim=-1)  # ( M1 + M2 ..., num_total_grids, 9+C)
            else:
                raise NotImplementedError

        new_features = interpolated_feats.view(-1, interpolated_feats.shape[-1])  # ((M1 + M2 ...) * num_total_grids, C)
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
            new_features = self.mlp(new_features)

            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolAggregationModule(nn.Module):
    def __init__(
            self, input_channels, num_local_voxel=(3, 3, 3), local_aggregation_type='local_interpolation',
            num_reduced_channels=30, num_channels_of_local_aggregation=32, post_mlps=(128,),
            max_neighbor_distance=None, neighbor_nsample=-1, neighbor_type=0, neighbor_distance_multiplier=2.0):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.total_voxels = self.num_local_voxel[0] * self.num_local_voxel[1] * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice']
        self.input_channels = input_channels
        self.num_reduced_channels = input_channels if num_reduced_channels is None else num_reduced_channels
        self.num_channels_of_local_aggregation = num_channels_of_local_aggregation
        self.max_neighbour_distance = max_neighbor_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type  # 1: ball, others: cube

        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = VectorPoolLocalInterpolateModule(
                mlp=None, num_voxels=self.num_local_voxel,
                max_neighbour_distance=self.max_neighbour_distance,
                nsample=self.neighbor_nsample,
                neighbor_type=self.neighbor_type,
                neighbour_distance_multiplier=neighbor_distance_multiplier,
            )
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels

        num_c_out = self.total_voxels * self.num_channels_of_local_aggregation

        self.separate_local_aggregation_layer = nn.Sequential(
            nn.Conv1d(num_c_in, num_c_out, kernel_size=1, groups=self.total_voxels, bias=False),
            nn.BatchNorm1d(num_c_out),
            nn.ReLU()
        )

        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list)

        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) -> str:
        ret = f'radius={self.max_neighbour_distance}, local_voxels=({self.num_local_voxel}, ' \
              f'local_aggregation_type={self.local_aggregation_type}, ' \
              f'num_c_reduction={self.input_channels}->{self.num_reduced_channels}, ' \
              f'num_c_local_aggregation={self.num_channels_of_local_aggregation}'
        return ret

    def vector_pool_with_voxel_query(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        use_xyz = 1
        pooling_type = 0 if self.local_aggregation_type == 'voxel_avg_pool' else 1

        new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid = pointnet2_utils.vector_pool_with_voxel_query_op(
            xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt,
            self.num_local_voxel[0], self.num_local_voxel[1], self.num_local_voxel[2],
            self.max_neighbour_distance, self.num_reduced_channels, use_xyz,
            self.num_mean_points_per_grid, self.neighbor_nsample, self.neighbor_type,
            pooling_type
        )
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid, num_mean_points_per_grid.item())

        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1, 3)  # (N, num_voxel, 3)
        new_features = new_features.view(num_new_pts, -1, self.num_reduced_channels)  # (N, num_voxel, C)
        new_features = torch.cat((new_local_xyz, new_features), dim=-1).view(num_new_pts, -1)

        return new_features, point_cnt_of_grid

    @staticmethod
    def get_dense_voxels_by_center(point_centers, max_neighbour_distance, num_voxels):
        """
        Args:
            point_centers: (N, 3)
            max_neighbour_distance: float
            num_voxels: [num_x, num_y, num_z]

        Returns:
            voxel_centers: (N, total_voxels, 3)
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(-R + R / num_voxels[0], R - R / num_voxels[0] + 1e-5, 2 * R / num_voxels[0], device=device)
        y_grids = torch.arange(-R + R / num_voxels[1], R - R / num_voxels[1] + 1e-5, 2 * R / num_voxels[1], device=device)
        z_grids = torch.arange(-R + R / num_voxels[2], R - R / num_voxels[2] + 1e-5, 2 * R / num_voxels[2], device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(x_grids, y_grids, z_grids)  # shape: [num_x, num_y, num_z]
        xyz_offset = torch.cat((
            x_offset.contiguous().view(-1, 1),
            y_offset.contiguous().view(-1, 1),
            z_offset.contiguous().view(-1, 1)), dim=-1
        )
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            xyz: (N, 3)
            xyz_batch_cnt: (batch_size)
            features: (N, C)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size)
        Returns:
            new_features: (M, total_voxels * C)
        """
        voxel_centers = self.get_dense_voxels_by_center(
            point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel
        )  # (M1 + M2 + ..., total_voxels, 3)
        voxel_features = self.local_interpolate_module.forward(
            support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt
        )  # ((M1 + M2 ...) * total_voxels, C)

        voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, **kwargs):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        N, C = features.shape

        assert C % self.num_reduced_channels == 0, \
            f'the input channels ({C}) should be an integral multiple of num_reduced_channels({self.num_reduced_channels})'

        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)

        if self.local_aggregation_type in ['voxel_avg_pool', 'voxel_random_choice']:
            vector_features, point_cnt_of_grid = self.vector_pool_with_voxel_query(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )
        elif self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )  # (M1 + M2 + ..., total_voxels * C)
        else:
            raise NotImplementedError

        vector_features = vector_features.permute(1, 0)[None, :, :]  # (1, num_voxels * C, M1 + M2 ...)

        new_features = self.separate_local_aggregation_layer(vector_features)

        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


class VectorPoolAggregationModuleMSG(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_groups = self.model_cfg.NUM_GROUPS

        self.layers = []
        c_in = 0
        for k in range(self.num_groups):
            cur_config = self.model_cfg[f'GROUP_CFG_{k}']
            cur_vector_pool_module = VectorPoolAggregationModule(
                input_channels=input_channels, num_local_voxel=cur_config.NUM_LOCAL_VOXEL,
                post_mlps=cur_config.POST_MLPS,
                max_neighbor_distance=cur_config.MAX_NEIGHBOR_DISTANCE,
                neighbor_nsample=cur_config.NEIGHBOR_NSAMPLE,
                local_aggregation_type=self.model_cfg.LOCAL_AGGREGATION_TYPE,
                num_reduced_channels=self.model_cfg.get('NUM_REDUCED_CHANNELS', None),
                num_channels_of_local_aggregation=self.model_cfg.NUM_CHANNELS_OF_LOCAL_AGGREGATION,
                neighbor_distance_multiplier=2.0
            )
            self.__setattr__(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config.POST_MLPS[-1]

        c_in += 3  # use_xyz

        shared_mlps = []
        for cur_num_c in self.model_cfg.MSG_POST_MLPS:
            shared_mlps.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def forward(self, **kwargs):
        features_list = []
        for k in range(self.num_groups):
            cur_xyz, cur_features = self.__getattr__(f'layer_{k}')(**kwargs)
            features_list.append(cur_features)

        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]  # (1, C, N)
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)  # (N, C)

        return cur_xyz, new_features
