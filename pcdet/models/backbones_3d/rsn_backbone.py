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

        out.features = self.conv2(out.features)  # type: spconv.SparseConvTensor
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

        self.conv0 = spconv.SparseConv3d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key)
        self.bn0 = norm_fn(out_channels)
        self.relu0 = nn.ReLU()
        self.B1Block = B1Block(in_channels, out_channels, stride=1, norm_fn=norm_fn, indice_key=indice_key)

    def forward(self, x):
        out = self.conv0(x)  # type: spconv.SparseConvTensor
        out.features = self.bn0(out.features)
        out.features = self.relu0(out.features)

        out = self.B1Block(out)


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
        self.block = nn.Sequential(self.conv_input, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
                                   self.conv6)

        self.num_point_features = 96

    def forward(self, x):
        """
        Too many things need to complete, mark.
        Args:
            x:

        Returns:

        """
        out = self.block(x)
        return out
