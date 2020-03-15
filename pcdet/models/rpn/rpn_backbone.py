import torch.nn as nn
import spconv
from functools import partial
from ...config import cfg


class BackBone8x(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        ret = {'spatial_features': spatial_features}
        return ret

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m
