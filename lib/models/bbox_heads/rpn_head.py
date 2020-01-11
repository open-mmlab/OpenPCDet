import torch
import torch.nn as nn
import numpy as np
from functools import partial
from ..utils import Empty


class RPNV2(nn.Module):
    def __init__(self, use_norm=True, num_class=2, layer_nums=(3, 5, 5), layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256), upsample_strides=(1, 2, 4), num_upsample_filters=(256, 256, 256),
                 num_input_features=128, num_anchor_per_loc=2, encode_background_as_zeros=True,
                 use_direction_classifier=True, box_code_size=7,
                 num_direction_bins=2, concat_input=False):
        super(RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._concat_input = concat_input
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)

        if use_norm:
            BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block = nn.Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = nn.Sequential(
                ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i], stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(num_upsample_filters)
        if self._concat_input:
            c_in += num_input_features

        if len(upsample_strides)>len(num_filters):
            deblock = nn.Sequential(
                ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(c_in, num_cls, 1)
        reg_channels = num_anchor_per_loc * box_code_size
        self.conv_box = nn.Conv2d(c_in, reg_channels, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(c_in, num_anchor_per_loc * num_direction_bins, 1)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, x_in, bev=None):
        ups = []
        x = x_in
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(x_in.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)
        ret_dict['spatial_features_last'] = x

        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict.update({
            'box_preds': box_preds,
            'cls_preds': cls_preds,
        })
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict['dir_cls_preds'] = dir_cls_preds
        return ret_dict
