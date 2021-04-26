import torch.nn as nn
import spconv
from functools import partial


class B1Block(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, stride=1, norm_fn=None, indice_key=None):
        super(B1Block, self).__init__()

        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        assert norm_fn is not None
        bias = norm_fn is not None  # bool type
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias,
            indice_key=indice_key)
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias,
            indice_key=indice_key)
        self.bn2 = norm_fn(out_channels)

    def forward(self, x):
        identity = x  # type: spconv.SparseConvTensor

        out = self.conv1(x)  # type: spconv.SparseConvTensor
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)  # type: spconv.SparseConvTensor
        out.features = self.bn2(out.features)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class B0Block(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, stride=1, norm_fn=None, indice_key=None):
        super(B0Block, self).__init__()

        # norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        assert norm_fn is not None
        bias = norm_fn is not None  # bool type
        indice_key_sc = indice_key if stride == 1 else None
        self.conv0 = spconv.SparseConv3d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key_sc)
        self.bn0 = norm_fn(out_channels)
        self.relu0 = nn.ReLU()
        self.B1Block = B1Block(in_channels, out_channels, stride=1, norm_fn=norm_fn, indice_key=indice_key)

    def forward(self, x):
        out = self.conv0(x)  # type: spconv.SparseConvTensor
        out.features = self.bn0(out.features)
        out.features = self.relu0(out.features)

        out = self.B1Block(out)

        return out


class CarS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(CarS, self).__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        out_channels = 96
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=input_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(out_channels),
            nn.ReLU()
        )

        self.conv1 = B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_1")
        self.conv2 = B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                             indice_key="subm_2")
        self.conv3 = B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_3")
        self.conv4 = B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                             indice_key="subm_4")
        self.conv5 = B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_5")
        self.conv6 = B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_6")

        self.num_point_features = 96

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv_input = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x_conv_input)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # out = self.block(input_sp_tensor)
        batch_dict.update({
            'encoded_spconv_tensor': x_conv6,
            'encoded_spconv_tensor_stride': 4
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #         'x_conv5': x_conv5,
        #         'x_conv6': x_conv6,
        #     }
        # })
        return batch_dict


class CarL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(CarL, self).__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        out_channels = 64
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels=input_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(out_channels),
            nn.ReLU()
        )

        self.conv1 = B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_1")
        self.conv2 = B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                             indice_key="subm_2")
        self.conv3 = B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_3")
        self.conv4 = B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                             indice_key="subm_4")
        self.conv5 = B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_5")
        self.conv6 = B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_5")

        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv_input = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x_conv_input)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        # out = self.block(input_sp_tensor)
        batch_dict.update({
            'encoded_spconv_tensor': x_conv6,
            'encoded_spconv_tensor_stride': 4
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
                'x_conv6': x_conv6,
            }
        })
        return batch_dict


class CarXL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(CarXL, self).__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        out_channels = 64
        self.conv = spconv.SparseSequential(
            conv_input=spconv.SparseSequential(
                spconv.SubMConv3d(in_channels=input_channels, out_channels=out_channels,
                                  kernel_size=3, padding=1, bias=False, indice_key='subm_input'),
                norm_fn(out_channels),
                nn.ReLU()
            ),

            conv1=B1Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_1"),
            conv2=B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                          indice_key="subm_2"),
            conv3=B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_3"),
            conv4=B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_3"),
            conv5=B0Block(in_channels=out_channels, out_channels=out_channels, stride=2, norm_fn=norm_fn,
                          indice_key="subm_5"),
            conv6=B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_6"),
            conv7=B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_6"),
            conv8=B0Block(in_channels=out_channels, out_channels=out_channels, norm_fn=norm_fn, indice_key="subm_6")
        )

        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv(input_sp_tensor)

        # out = self.block(input_sp_tensor)
        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 4
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #         'x_conv5': x_conv5,
        #         'x_conv6': x_conv6,
        #     }
        # })
        return batch_dict
