from typing import List
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils


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

class StackSAModulePyramid(nn.Module):

    def __init__(self, *, mlps: List[List[int]], nsamples, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

"""
Dynamic Conv with learnable convolutional weights
"""
class StackSAModulePyramidDynamicConvV1(nn.Module):
    def __init__(self, *, mlps: List[List[int]], nsamples, grid_sizes, kernel_sizes, activation = None, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.kernel_sizes = kernel_sizes
        self.grid_sizes = grid_sizes

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.conv_weights = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.activation = activation
        if self.activation not in ['softmax', 'layernorm']:
            raise NotImplementedError

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

            kernel_size = kernel_sizes[i]
            conv_weight = []
            conv_weight.extend([
                nn.Linear(grid_size ** 3 * mlp_spec[-1], kernel_size ** 3 * mlp_spec[-1]),
                nn.ReLU(),
                nn.Linear(kernel_size ** 3 * mlp_spec[-1], kernel_size ** 3 * mlp_spec[-1])
            ])
            self.conv_weights.append(nn.Sequential(*conv_weight))

            if self.activation == 'layernorm':
                self.layer_norms.append(nn.LayerNorm(kernel_size ** 3))

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            kernel_size = self.kernel_sizes[i]
            grid_size = self.grid_sizes[i]
            conv_weight = new_features.reshape(batch_size * num_rois, -1)
            conv_weight = self.conv_weights[i](conv_weight).reshape(batch_size * num_rois, num_features, kernel_size ** 3)
            if self.activation == 'softmax':
                conv_weight = F.softmax(conv_weight, dim=-1)
            elif self.activation == 'layernorm':
                conv_weight = self.layer_norms[i](conv_weight)
            else:
                assert False, 'activation has to be softmax or layernorm'
            conv_weight = conv_weight.reshape(batch_size * num_rois * num_features, 1, kernel_size, kernel_size, kernel_size)
            conv_features = new_features.transpose(1, 2).contiguous() # (BN, C, grid_size^3)
            conv_features = conv_features.reshape(1, batch_size * num_rois * num_features, grid_size, grid_size, grid_size)
            conv_features = F.conv3d(conv_features, conv_weight, stride=1, padding=0, groups=batch_size * num_rois * num_features)
            conv_features = conv_features.reshape(batch_size * num_rois, num_features, -1)
            conv_features = conv_features.transpose(1, 2).contiguous()
            new_features_list.append(conv_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

"""
Dynamic Conv with dynamic selection of convolutional kernels
"""
class StackSAModulePyramidDynamicConvV2(nn.Module):
    def __init__(self, *, mlps: List[List[int]], nsamples, grid_sizes, kernel_sizes, K, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.kernel_sizes = kernel_sizes
        self.grid_sizes = grid_sizes

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.conv_weights = []
        self.K = K
        self.attention_modules = nn.ModuleList()

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

            kernel_size = kernel_sizes[i]
            conv_weight = nn.Parameter(
                torch.zeros((self.K, mlp_spec[-1], 1, kernel_size, kernel_size, kernel_size)).cuda()
            )
            nn.init.kaiming_normal_(conv_weight)
            self.conv_weights.append(conv_weight)

            grid_size = grid_sizes[i]
            attention_module = []
            attention_module.extend([
                nn.Linear(grid_size ** 3 * mlp_spec[-1], mlp_spec[-1]),
                nn.ReLU(),
                nn.Linear(mlp_spec[-1], self.K)
            ])
            self.attention_modules.append(nn.Sequential(
                *attention_module
            ))

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            kernel_size = self.kernel_sizes[i]
            grid_size = self.grid_sizes[i]
            attention_features = new_features.reshape(batch_size * num_rois, -1)
            attention_k = self.attention_modules[i](attention_features) # (BN, K)
            attention_k = F.softmax(attention_k, dim=-1)
            conv_weight = self.conv_weights[i].reshape(self.K, -1)
            conv_weight = torch.matmul(attention_k, conv_weight) # (BN, C * 1 * kernel_size^3)
            conv_weight = conv_weight.reshape(batch_size * num_rois * num_features, 1, kernel_size, kernel_size, kernel_size)

            conv_features = new_features.transpose(1, 2).contiguous() # (BN, C, grid_size^3)
            conv_features = conv_features.reshape(1, batch_size * num_rois * num_features, grid_size, grid_size, grid_size)
            conv_features = F.conv3d(conv_features, conv_weight, stride=1, padding=0, groups=batch_size * num_rois * num_features)
            conv_features = conv_features.reshape(batch_size * num_rois, num_features, -1)
            conv_features = conv_features.transpose(1, 2).contiguous()
            new_features_list.append(conv_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

"""
Transformer encoder for each pyramid level
"""
class StackSAModulePyramidTransformerV1(nn.Module):
    def __init__(self, *, mlps, nsamples, grid_sizes, use_pos_emb = True, use_xyz = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.use_pos_emb = use_pos_emb

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.transformer_encoders = nn.ModuleList()
        self.pos_embeddings = []

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

            num_out_ch = mlp_spec[-1]

            encoder_layer = TransformerEncoderLayer(
                d_model=num_out_ch,
                nhead=4,
                dim_feedforward=num_out_ch,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
            )
            self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))
            if self.use_pos_emb:
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features = new_features.permute(1, 0, 2).contiguous() # (L, B, C)
            if self.use_pos_emb:
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos = pos_emb)
            else:
                new_features = self.transformer_encoders[i](new_features)
            new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features

"""
SiblingHeads for disentagled classification and regression
"""
class StackSAModuleSib(nn.Module):
    def __init__(self, *, mlps, nsamples, grid_sizes, use_pos_emb = True, use_xyz = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.use_pos_emb = use_pos_emb

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

            num_out_ch = mlp_spec[-1]

            encoder_layer = TransformerEncoderLayer(
                d_model=num_out_ch,
                nhead=4,
                dim_feedforward=num_out_ch,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
            )
            self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

            if self.use_pos_emb:
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features = new_features.permute(1, 0, 2).contiguous() # (L, B, C)
            if self.use_pos_emb:
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos = pos_emb)
            else:
                new_features = self.transformer_encoders[i](new_features)
            new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)

            cls_weights = self.cls_weight_module[i](new_features) # (B, L, 1)
            cls_features = new_features * cls_weights
            cls_features_list.append(cls_features)

            reg_weights = self.reg_weight_module[i](new_features) # (B, L, 1)
            reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return cls_features, reg_features

"""
Transformer encoder for all grid points
"""
class StackSAModulePyramidTransformerV2(nn.Module):
    def __init__(self, *, mlps, nsamples, grid_sizes, use_pos_emb = True, use_xyz = True, pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.use_pos_emb = use_pos_emb

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.num_grid_points = 0

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.num_grid_points += grid_size ** 3
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

        num_out_ch = mlps[0][-1]
        encoder_layer = TransformerEncoderLayer(
            d_model=num_out_ch,
            nhead=2,
            dim_feedforward=num_out_ch,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=1, norm=None)
        if self.use_pos_emb:
            self.pos_embedding = nn.Parameter(torch.zeros((self.num_grid_points, num_out_ch)).cuda())
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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
        if self.use_pos_emb:
            pos_emb = self.pos_embedding.unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
            new_features = self.transformer_encoder(new_features, pos=pos_emb)
        else:
            new_features = self.transformer_encoder(new_features)
        new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)

        return new_features

"""
Attention ops
"""
class StackSAModulePyramidAttention(nn.Module):

    def __init__(self, input_channels, nsamples, num_heads, head_dims, attention_op, mlp_specs = None):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))

            if attention_op in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']:
                pass

            elif attention_op in ['v10', 'v11', 'v12']:
                self.pos_proj.append(nn.Sequential(
                    nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.key_proj.append(nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False))
                self.value_proj.append(nn.Sequential(
                    nn.Conv1d(input_channels, self.output_dims // 2, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.output_dims // 2, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.attention_proj.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                ))
                if attention_op == 'v10':
                    self.norm_layer.append(nn.Softmax(dim=-1))
                elif attention_op == 'v11':
                    self.norm_layer.append(nn.LayerNorm(nsample))
                elif attention_op == 'v12':
                    self.norm_layer.append(nn.Sequential())
                else:
                    raise NotImplementedError

            elif attention_op in ['v13']:
                self.pos_proj.append(nn.Sequential(
                    nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.key_proj.append(nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False))
                self.value_proj.append(nn.Sequential(
                    nn.Conv1d(input_channels, self.output_dims // 2, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.output_dims // 2, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.attention_proj.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                ))
                self.norm_layer.append(nn.Softmax(dim=-1))
                self.k_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, 1, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.q_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, 1, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.qk_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, 1, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.v_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, 1, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))

            elif attention_op in ['v14']:
                self.pos_proj.append(nn.Sequential(
                    nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.key_proj.append(nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False))
                self.value_proj.append(nn.Sequential(
                    nn.Conv1d(input_channels, self.output_dims // 2, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.output_dims // 2, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.attention_proj.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                ))
                self.norm_layer.append(nn.Softmax(dim=-1))
                self.k_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.q_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.qk_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))
                self.v_coef.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.Sigmoid()
                ))

            elif attention_op in ['vmh_1']:
                self.pos_proj.append(nn.Sequential(
                    nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.ReLU(),
                ))
                self.key_proj.append(nn.Sequential(
                    nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                    nn.ReLU()
                ))
                self.value_proj.append(nn.Sequential(
                    nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                    nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                    nn.BatchNorm1d(self.output_dims),
                    nn.ReLU(),
                ))
                self.attention_proj.append(nn.Sequential(
                    nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
                ))
                self.norm_layer.append(nn.Softmax(dim=-1))
                self.k_coef.append(nn.Sequential(
                    nn.Linear(self.output_dims, self.output_dims, bias=False),
                    nn.Sigmoid()
                ))
                self.q_coef.append(nn.Sequential(
                    nn.Linear(self.output_dims, self.output_dims, bias=False),
                    nn.Sigmoid()
                ))
                self.qk_coef.append(nn.Sequential(
                    nn.Linear(self.output_dims, self.output_dims, bias=False),
                    nn.Sigmoid()
                ))
                self.v_coef.append(nn.Sequential(
                    nn.Linear(self.output_dims, self.output_dims, bias=False),
                    nn.Sigmoid()
                ))

            else:
                raise NotImplementedError

        self.attention_op = attention_op
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )

            if self.attention_op in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']:
                assert False, 'attention ops v1-v9 are legacy code'

            elif self.attention_op in ['v10', 'v11', 'v12']:
                pos_embedding = self.pos_proj[i](grouped_xyz)
                key_embedding = self.key_proj[i](grouped_features)
                value_embedding = self.value_proj[i](grouped_features)
                value_embedding = value_embedding + pos_embedding
                attention_embedding = pos_embedding + key_embedding + pos_embedding * key_embedding
                attention_map = self.attention_proj[i](attention_embedding)
                attention_map = self.norm_layer[i](attention_map)
                attend_features = (attention_map * value_embedding).sum(-1)
                # new_features = attend_features.squeeze(dim=0).permute(1, 0).contiguous()
                new_features = attend_features
            elif self.attention_op in ['v13', 'v14']:
                pos_embedding = self.pos_proj[i](grouped_xyz)
                key_embedding = self.key_proj[i](grouped_features)
                value_embedding = self.value_proj[i](grouped_features)
                v_coef = self.v_coef[i](pos_embedding)
                value_embedding = value_embedding + pos_embedding * v_coef
                q_coef = self.q_coef[i](pos_embedding)
                k_coef = self.k_coef[i](key_embedding)
                qk_coef = self.qk_coef[i](pos_embedding * key_embedding)
                attention_embedding = pos_embedding * q_coef + key_embedding * k_coef + pos_embedding * key_embedding * qk_coef
                attention_map = self.attention_proj[i](attention_embedding)
                attention_map = self.norm_layer[i](attention_map)
                attend_features = (attention_map * value_embedding).sum(-1)
                # new_features = attend_features.squeeze(dim=0).permute(1, 0).contiguous()
                new_features = attend_features
            elif self.attention_op in ['vmh_1']:
                pos_embedding = self.pos_proj[i](grouped_xyz)
                key_embedding = self.key_proj[i](grouped_features)
                value_embedding = self.value_proj[i](grouped_features)
                pos_key_embedding = pos_embedding * key_embedding

                v_coef = self.v_coef[i](pos_embedding.mean(2))
                q_coef = self.q_coef[i](pos_embedding.mean(2))
                k_coef = self.k_coef[i](key_embedding.mean(2))
                qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

                value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
                attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

                attention_map = self.attention_proj[i](attention_embedding)
                attention_map = self.norm_layer[i](attention_map)
                # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
                attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])
                attend_features = (attention_map * value_embedding).sum(-1)
                new_features = attend_features
            else:
                raise NotImplementedError

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features.contiguous()

class StackSAModulePyramidAdaptiveAttention(nn.Module):

    def __init__(self, input_channels, nsamples, num_heads, head_dims, attention_op, predict_radii, predict_ns, norm_factors, pre_weights):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims

        self.pre_weights = pre_weights
        self.predict_groupers = nn.ModuleList()
        self.predict_modules = nn.ModuleList()
        self.predict_radii = predict_radii
        self.predict_ns = predict_ns
        self.num_predict_levels = len(predict_radii)
        for i in range(self.num_predict_levels):
            self.predict_groupers.append(pointnet2_utils.QueryAndGroup(radius=self.predict_radii[i], nsample=self.predict_ns[i]))
            self.predict_modules.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
        predict_feat_dims = self.output_dims * self.num_predict_levels

        self.radius_modules = nn.ModuleList()
        self.norm_factors = norm_factors

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            self.radius_modules.append(nn.Sequential(
                nn.Linear(predict_feat_dims, predict_feat_dims // 4, bias=False),
                nn.ReLU(),
                nn.Linear(predict_feat_dims // 4, 1, bias=False),
            ))
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

        self.attention_op = attention_op
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, anchor_xyz, anchor_batch_cnt, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list,
                features=None, batch_size=None, num_rois=None, temperature=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        predict_feature_list = []
        for i in range(self.num_predict_levels):
            predict_feature, _ = self.predict_groupers[i](
                xyz, xyz_batch_cnt, anchor_xyz, anchor_batch_cnt, features
            )
            predict_feature = self.predict_modules[i](predict_feature)
            predict_feature = predict_feature.max(dim = 2, keepdim = False)[0]
            predict_feature_list.append(predict_feature)
        predict_features = torch.cat(predict_feature_list, dim=1)

        new_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            delta_r = self.radius_modules[i](predict_features) # (BN, 1)
            div_factor, clamp_min, clamp_max = self.norm_factors[i]
            delta_r = torch.clamp(delta_r / div_factor, min = clamp_min, max=clamp_max)
            num_grid_points = new_xyz_r.shape[0] // delta_r.shape[0]
            delta_r = delta_r.unsqueeze(1).repeat(1, num_grid_points, 1).view(-1, 1)

            new_xyz_r = new_xyz_r + delta_r
            ex_new_xyz_r = new_xyz_r + temperature * 5

            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, ex_new_xyz_r, new_xyz_batch_cnt, features
            )

            grouped_dist = torch.sqrt((grouped_xyz ** 2).sum(dim = 1, keepdim = False))
            r_weights = 1 - torch.sigmoid((grouped_dist - new_xyz_r) / temperature)  # (N, nsample)
            r_weights = r_weights.unsqueeze(1)

            if self.pre_weights:
                grouped_features = r_weights * grouped_features

            pos_embedding = self.pos_proj[i](grouped_xyz)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2))
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

            attention_map = self.attention_proj[i](attention_embedding)
            attention_map = self.norm_layer[i](attention_map)
            # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            if self.pre_weights:
                attend_features = (attention_map * value_embedding).sum(-1)
            else:
                attend_features = (attention_map * r_weights * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return new_features.contiguous()

class StackSAModuleAll(nn.Module):

    def __init__(self, input_channels, nsamples, grid_sizes, num_heads, head_dims, attention_op, predict_radii, predict_ns, norm_factors, pre_weights):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims
        self.grid_sizes = grid_sizes


        self.pre_weights = pre_weights
        self.predict_groupers = nn.ModuleList()
        self.predict_modules = nn.ModuleList()
        self.predict_radii = predict_radii
        self.predict_ns = predict_ns
        self.num_predict_levels = len(predict_radii)
        for i in range(self.num_predict_levels):
            self.predict_groupers.append(pointnet2_utils.QueryAndGroup(radius=self.predict_radii[i], nsample=self.predict_ns[i]))
            self.predict_modules.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
        predict_feat_dims = self.output_dims * self.num_predict_levels

        self.radius_modules = nn.ModuleList()
        self.norm_factors = norm_factors

        self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.radius_modules.append(nn.Sequential(
                nn.Linear(predict_feat_dims, predict_feat_dims // 4, bias=False),
                nn.ReLU(),
                nn.Linear(predict_feat_dims // 4, 1, bias=False),
            ))
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

            num_out_ch = self.output_dims

            encoder_layer = TransformerEncoderLayer(
                d_model=num_out_ch,
                nhead=4,
                dim_feedforward=num_out_ch,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
            )
            self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

            self.pos_embeddings.append(
                nn.Parameter(
                    torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                )
            )



        self.attention_op = attention_op
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, anchor_xyz, anchor_batch_cnt, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list,
                features=None, batch_size=None, num_rois=None, temperature=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        predict_feature_list = []
        for i in range(self.num_predict_levels):
            predict_feature, _ = self.predict_groupers[i](
                xyz, xyz_batch_cnt, anchor_xyz, anchor_batch_cnt, features
            )
            predict_feature = self.predict_modules[i](predict_feature)
            predict_feature = predict_feature.max(dim = 2, keepdim = False)[0]
            predict_feature_list.append(predict_feature)
        predict_features = torch.cat(predict_feature_list, dim=1)

        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            delta_r = self.radius_modules[i](predict_features) # (BN, 1)
            div_factor, clamp_min, clamp_max = self.norm_factors[i]
            # delta_r = torch.clamp(delta_r / div_factor, min = clamp_min, max=clamp_max)
            delta_r = delta_r / div_factor
            num_grid_points = new_xyz_r.shape[0] // delta_r.shape[0]
            delta_r = delta_r.unsqueeze(1).repeat(1, num_grid_points, 1).view(-1, 1)

            new_xyz_r = new_xyz_r + delta_r
            new_xyz_r = torch.clamp(new_xyz_r, min=clamp_min, max=clamp_max)
            ex_new_xyz_r = new_xyz_r + temperature * 5

            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, ex_new_xyz_r, new_xyz_batch_cnt, features
            )

            grouped_dist = torch.sqrt((grouped_xyz ** 2).sum(dim = 1, keepdim = False))
            r_weights = 1 - torch.sigmoid((grouped_dist - new_xyz_r) / temperature)  # (N, nsample)
            r_weights = r_weights.unsqueeze(1)

            if self.pre_weights:
                grouped_features = r_weights * grouped_features

            pos_embedding = self.pos_proj[i](grouped_xyz)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2))
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

            attention_map = self.attention_proj[i](attention_embedding)
            attention_map = self.norm_layer[i](attention_map)
            # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            if self.pre_weights:
                attend_features = (attention_map * value_embedding).sum(-1)
            else:
                attend_features = (attention_map * r_weights * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)

            new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
            pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
            new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
            new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)

            cls_weights = self.cls_weight_module[i](new_features)  # (B, L, 1)
            cls_features = new_features * cls_weights
            cls_features_list.append(cls_features)

            reg_weights = self.reg_weight_module[i](new_features)  # (B, L, 1)
            reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return cls_features.contiguous(), reg_features.contiguous()

class StackSAModuleAllNoTemp(nn.Module):

    def __init__(self, input_channels, nsamples, grid_sizes, num_heads, head_dims, attention_op, dp_value = 0.1, tr_mode = 'Normal'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims
        self.grid_sizes = grid_sizes

        self.tr_mode = tr_mode
        assert self.tr_mode in ['NoTr', 'Normal', 'Residual']

        if self.tr_mode != 'NoTr':
            self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

            num_out_ch = self.output_dims

            if self.tr_mode != 'NoTr':
                encoder_layer = TransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=4,
                    dim_feedforward=num_out_ch,
                    dropout=dp_value,
                    activation="relu",
                    normalize_before=False,
                )
                self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

        self.attention_op = attention_op
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list,
                features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )

            pos_embedding = self.pos_proj[i](grouped_xyz)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2))
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

            attention_map = self.attention_proj[i](attention_embedding)
            attention_map = self.norm_layer[i](attention_map)
            # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            attend_features = (attention_map * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)

            if self.tr_mode == 'NoTr':
                pass
            elif self.tr_mode == 'Normal':
                new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
                new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
            elif self.tr_mode == 'Residual':
                tr_new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, tr_new_features.shape[1], 1)  # (L, B, C)
                tr_new_features = self.transformer_encoders[i](tr_new_features, pos=pos_emb)
                tr_new_features = tr_new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
                new_features = new_features + tr_new_features
            else:
                raise NotImplementedError

            cls_weights = self.cls_weight_module[i](new_features)  # (B, L, 1)
            cls_features = new_features * cls_weights
            cls_features_list.append(cls_features)

            reg_weights = self.reg_weight_module[i](new_features)  # (B, L, 1)
            reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return cls_features.contiguous(), reg_features.contiguous()

class StackSAModuleMSGDeform(nn.Module):
    """
    Set abstraction with single radius prediction for each roi
    """

    def __init__(self, *, temperatures: List[float], div_coefs: List[float], radii: List[float],
                 nsamples: List[int], predict_nsamples: List[int],
                 mlps: List[List[int]], pmlps: List[List[int]], pfcs: List[List[int]],
                 grid_size: int, use_xyz: bool = True):
        """
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.grid_size = grid_size
        self.MIN_R = 0.01

        self.radii_list = radii
        self.div_coef_list = div_coefs

        self.norm_groupers = nn.ModuleList()
        self.deform_groupers = nn.ModuleList()

        self.feat_mlps = nn.ModuleList()

        self.predict_mlps = nn.ModuleList()
        self.predict_fcs = nn.ModuleList()

        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            predict_nsample = predict_nsamples[i]
            temperature = temperatures[i]

            self.norm_groupers.append(
                pointnet2_utils.QueryAndGroup(radius, predict_nsample, use_xyz=use_xyz)
            )
            self.deform_groupers.append(
                pointnet2_utils.QueryAndGroupDeform(temperature, nsample, use_xyz=use_xyz)
            )

            mlp_spec = mlps[i]
            predict_mlp_spec = pmlps[i]
            if use_xyz:
                mlp_spec[0] += 3
                predict_mlp_spec[0] += 3

            self.feat_mlps.append(self._make_mlp_layer(mlp_spec))

            self.predict_mlps.append(self._make_mlp_layer(predict_mlp_spec))

            fc_spec = pfcs[i]
            self.predict_fcs.append(self._make_fc_layer(fc_spec))

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

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_mlp_layer(self, mlp_spec):
        mlps = []
        for i in range(len(mlp_spec) - 1):
            mlps.extend([
                nn.Conv2d(mlp_spec[i], mlp_spec[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[i + 1]),
                nn.ReLU()
            ])
        return nn.Sequential(*mlps)

    def _make_fc_layer(self, fc_spec):
        assert len(fc_spec) == 2
        return nn.Linear(fc_spec[0], fc_spec[1], bias = True)

    def forward(self, xyz, xyz_batch_cnt, rois, roi_features, features=None, temperature_decay=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param rois: (B, num_rois, grid_size^3, 3) roi grid points
        :param roi_features: (B, num_rois, C) roi features
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        batch_size = rois.shape[0]
        num_rois = rois.shape[1]
        new_xyz = rois.view(batch_size, -1, 3).contiguous()
        new_xyz_batch_cnt = new_xyz.new_full((batch_size), new_xyz.shape[1]).int()
        new_xyz = new_xyz.view(-1, 3).contiguous()
        new_features_list = []

        for k in range(len(self.norm_groupers)):
            # radius prediction
            predicted_features, ball_idxs = self.norm_groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M, C, nsample)
            predicted_features = predicted_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M, nsample)
            predicted_features = self.predict_mlps[k](predicted_features)  # (1, C, M, nsample)

            predicted_features = F.max_pool2d(
                predicted_features, kernel_size=[1, predicted_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M)

            # M = batch_size * num_rois * grid_size^3
            predicted_features = predicted_features.squeeze(0).permute(0, 1).contiguous() # (M, C)
            num_predicted_features = predicted_features.shape[1]
            predicted_features = predicted_features.view(batch_size, num_rois, self.grid_size ** 3, num_predicted_features)
            predicted_features = predicted_features.view(batch_size, num_rois, -1).contiguous()

            predicted_residual_r = self.predict_fcs[k](torch.cat([predicted_features, roi_features], dim = 2))  # (batch_size, num_rois, C -> 1)

            new_xyz_r = predicted_residual_r / self.div_coef_list[k] + self.radii_list[k]
            # constrain predicted radius above MIN_R
            new_xyz_r = torch.clamp(new_xyz_r, min = self.MIN_R)

            new_xyz_r = new_xyz_r.unsqueeze(2).repeat(1, 1, self.grid_size ** 3, 1) # (batch_size, num_rois, grid_size^3, 1)
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            # feature extraction
            # new_features (M, C, nsample) weights (M, nsample)
            new_features, new_weights, ball_idxs = self.deform_groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features, temperature_decay
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M, nsample)
            new_features = self.feat_mlps[k](new_features)  # (1, C, M, nsample)

            # multiply after mlps
            new_weights = new_weights.unsqueeze(0).unsqueeze(0) # (1, 1, M, nsample)
            new_features = new_weights * new_features
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
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

"""
Transformer encoder + decoder for each pyramid level
"""
class TransformerROI(nn.Module):
    def __init__(self, *, mlps,
                    nsamples,
                    grid_sizes,
                    num_encoder_layers,
                    num_cls_layers,
                    num_reg_layers,
                    num_encoder_heads,
                    num_decoder_heads,
                    reg_num_q,
                    cls_num_q,
                    use_pos_emb = True,
                    use_xyz = True,
                    no_zero_query = True,
                    no_dropout_norm = True,
                    pool_method='max_pool'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        self.num_pyramid_levels = len(nsamples)
        assert len(nsamples) == len(mlps)
        self.use_pos_emb = use_pos_emb
        self.reg_num_q = reg_num_q
        self.cls_num_q = cls_num_q

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.transformer_encoders = nn.ModuleList()
        self.cls_decoders = nn.ModuleList()
        self.reg_decoders = nn.ModuleList()
        self.pos_embeddings = []
        self.reg_pos_embeddings = []
        self.cls_pos_embeddings = []

        self.no_dropout_norm = no_dropout_norm
        self.no_zero_query = no_zero_query

        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramid(nsample, use_xyz=use_xyz))
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

            num_out_ch = mlp_spec[-1]
            reg_num = self.reg_num_q[i]
            cls_num = self.cls_num_q[i]

            if self.no_dropout_norm:
                encoder_layer = ReTransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=num_encoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

                reg_decoder_layer = ReTransformerDecoderLayer(
                    d_model=num_out_ch,
                    nhead=num_decoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

                cls_decoder_layer = ReTransformerDecoderLayer(
                    d_model=num_out_ch,
                    nhead=num_decoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

            else:
                encoder_layer = TransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=num_encoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

                reg_decoder_layer = TransformerDecoderLayer(
                    d_model=num_out_ch,
                    nhead=num_decoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

                cls_decoder_layer = TransformerDecoderLayer(
                    d_model=num_out_ch,
                    nhead=num_decoder_heads,
                    dim_feedforward=num_out_ch,
                    dropout=0.1,
                    activation="relu",
                    normalize_before=False,
                )

            self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_encoder_layers, norm = None))
            self.cls_decoders.append(TransformerDecoder(decoder_layer = cls_decoder_layer, num_layers = num_cls_layers, norm = None))
            self.reg_decoders.append(TransformerDecoder(decoder_layer = reg_decoder_layer, num_layers = num_reg_layers, norm = None))

            if self.use_pos_emb:
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )
                self.cls_pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((cls_num, num_out_ch)).cuda()
                    )
                )
                self.reg_pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((reg_num, num_out_ch)).cuda()
                    )
                )

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

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list, features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size)
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()
            new_features, _ = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[i](new_features)  # (1, C, M1 + M2 ..., nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)

            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            num_features = new_features.shape[1]
            new_features = new_features.view(batch_size * num_rois, -1, num_features)

            new_features = new_features.permute(1, 0, 2).contiguous() # (L, B, C)
            cls_num = self.cls_num_q[i]
            reg_num = self.reg_num_q[i]

            if self.no_zero_query:
                max_feats = torch.max(new_features, dim = 0, keepdim=True)[0]
                #mean_feats = torch.mean(new_features, dim = 0, keepdim=True)
                #cls_emb = max_feats + mean_feats
                #reg_emb = max_feats + mean_feats
                cls_emb = max_feats
                reg_emb = max_feats
                cls_emb = cls_emb.repeat(cls_num, 1, 1)
                reg_emb = reg_emb.repeat(reg_num, 1, 1)
            else:
                cls_emb = torch.zeros((cls_num, new_features.shape[1], new_features.shape[2])).to(new_features.device)
                reg_emb = torch.zeros((reg_num, new_features.shape[1], new_features.shape[2])).to(new_features.device)

            if self.use_pos_emb:
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                cls_pos_emb = self.cls_pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                reg_pos_emb = self.reg_pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
                cls_features = self.cls_decoders[i](tgt=cls_emb, memory=new_features, pos=pos_emb, query_pos=cls_pos_emb)
                reg_features = self.reg_decoders[i](tgt=reg_emb, memory=new_features, pos=pos_emb, query_pos=reg_pos_emb)
            else:
                new_features = self.transformer_encoders[i](new_features)
                cls_features = self.cls_decoders[i](tgt=cls_emb, memory=new_features)
                reg_features = self.reg_decoders[i](tgt=reg_emb, memory=new_features)

            cls_features = cls_features.permute(1, 0, 2).contiguous()  # (B, L, C)
            reg_features = reg_features.permute(1, 0, 2).contiguous()  # (B, L, C)

            cls_features_list.append(cls_features)
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(Li), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(Li), C)

        return cls_features, reg_features

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class ReTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + src2
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class ReTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + tgt2
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")