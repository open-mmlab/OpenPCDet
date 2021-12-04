from functools import partial

from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn

class B1Block(spconv.SparseModule):
    def __init__(self, inplanes, planes, norm_fn=None, indice_key=None):
        super(B1Block, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None

        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        out.features += identity.features
        out = replace_feature(out, self.relu(out.features))

        return out

class B0Block(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, indice_key=None, spconv_key=None):
        super(B0Block, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None

        self.conv0 = spconv.SparseConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=spconv_key)
        self.bn0 = norm_fn(planes)
        self.conv1 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.stride = stride

    def forward(self, x):
        sp_out = self.conv0(x)
        sp_out = replace_feature(sp_out, self.bn0(sp_out.features))
        sp_out = replace_feature(sp_out, self.relu(sp_out.features))

        identity = sp_out

        out = self.conv1(sp_out)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class CarS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv1 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_1')
        self.conv2 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_2', spconv_key='spc_2')
        self.conv3 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_3')
        self.conv4 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_4', spconv_key='spc_4')
        self.conv5 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_5')
        self.conv6 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_6')

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
                'x_conv6': x_conv6
            }
        })

        return batch_dict

class CarL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv1 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_1')
        self.conv2 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_2', spconv_key='spc_2')
        self.conv3 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_3', spconv_key='spc_3')
        self.conv4 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_4', spconv_key='spc_4')
        self.conv5 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_5', spconv_key='spc_5')
        self.conv6 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_6', spconv_key='spc_6')

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
                'x_conv6': x_conv6
            }
        })

        return batch_dict

class CarXL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv1 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_1')
        self.conv2 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_2', spconv_key='spc_2')
        self.conv3 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_3', spconv_key='spc_3')
        self.conv4 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_4', spconv_key='spc_4')
        self.conv5 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_5', spconv_key='spc_5')
        self.conv6 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_6', spconv_key='spc_6')
        self.conv7 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_7', spconv_key='spc_7')
        self.conv8 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_8', spconv_key='spc_8')

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
        x_conv7 = self.conv7(x_conv6)
        x_conv8 = self.conv8(x_conv7)


        batch_dict.update({
            'encoded_spconv_tensor': x_conv8,
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
                'x_conv7': x_conv7,
                'x_conv8': x_conv8
            }
        })

        return batch_dict

class PedS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv1 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_1')
        self.conv2 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_2', spconv_key='spc_2')
        self.conv3 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_3')
        self.conv4 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_4')

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

        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 2
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict

class PedL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm_input'),
            norm_fn(64),
            nn.ReLU(),
        )

        self.conv1 = B1Block(inplanes=64, planes=64, norm_fn=norm_fn, indice_key='subm_1')
        self.conv2 = B0Block(inplanes=64, planes=64, stride=2, norm_fn=norm_fn, indice_key='subm_2', spconv_key='spc_2')
        self.conv3 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_3', spconv_key='spc_3')
        self.conv4 = B0Block(inplanes=64, planes=64, stride=1, norm_fn=norm_fn, indice_key='subm_4', spconv_key='spc_4')

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

        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 2
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict