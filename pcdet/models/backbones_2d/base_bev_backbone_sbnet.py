import numpy as np
import torch
import torch.nn as nn

from sbnet.layers import SparseBlock_ConvChain, SparseBlock_Conv2d_BN_ReLU

class BaseBEVBackboneSbnet(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        kernel_size=3
        self.tcount=self.model_cfg.TILE_COUNT
        chain_masks = self.model_cfg.get('SBNET_BLOCK_CHAIN_MASKS', [None]*num_levels)
        for idx in range(num_levels):
            cur_layers = [
                SparseBlock_Conv2d_BN_ReLU(c_in_list[idx], num_filters[idx], kernel_size,
                    stride=layer_strides[idx], bias=False, bn_eps=1e-3, bn_momentum=0.01,
                    bcount=self.tcount, transpose=True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.append(
                    SparseBlock_Conv2d_BN_ReLU(num_filters[idx], num_filters[idx], kernel_size,
                        bias=False, bn_eps=1e-3, bn_momentum=0.01,
                        bcount=self.tcount, transpose=True)
                )
            sbcc = SparseBlock_ConvChain(chain_masks[idx])
            for l in cur_layers:
                sbcc.append_conv(l)
            self.blocks.append(sbcc)
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        SparseBlock_Conv2d_BN_ReLU(num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx], stride=upsample_strides[idx], bias=False,
                            bn_eps=1e-3, bn_momentum=0.01,
                            bcount=self.tcount, deconv=True, transpose=True)
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        SparseBlock_Conv2d_BN_ReLU(num_filters[idx], num_upsample_filters[idx],
                            stride, stride=stride, bias=False, bn_eps=1e-3, bn_momentum=0.01,
                            bcount=self.tcount, transpose=True)
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                SparseBlock_Conv2d_BN_ReLU(c_in, c_in,
                    upsample_strides[-1], stride=upsample_strides[-1], bias=False,
                    bn_eps=1e-3, bn_momentum=0.01,
                    bcount=self.tcount, deconv=True, transpose=True)
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        ups = []
        #ret_dict = {}
        data_dict['sbnet_x'] = data_dict['spatial_features']
        for i in range(len(self.blocks)):
            #torch.cuda.nvtx.range_push(f'Block_{i+1}')
            data_dict = self.blocks[i](data_dict)
            data_dict['sbnet_x'] = data_dict['sbnet_y']
            #if self.training:
            #    stride = int(spatial_features.shape[2] / x.shape[2])
            #    ret_dict['spatial_features_%dx' % stride] = x
            #torch.cuda.nvtx.range_pop()
            #torch.cuda.nvtx.range_push(f'Deblock_{i+1}')
            if len(self.deblocks) > 0:
                data_dict = self.deblocks[i](data_dict)
            ups.append(data_dict['sbnet_y'])
            #torch.cuda.nvtx.range_pop()

        if len(ups) > 1:
            data_dict['sbnet_x'] = torch.cat(ups, dim=-1)
        else:
            data_dict['sbnet_x'] = ups[0]

        if len(self.deblocks) > len(self.blocks):
            data_dict = self.deblocks[-1](data_dict)
        else:
            data_dict['sbnet_y'] = data_dict['sbnet_x']

        data_dict['spatial_features_2d'] = data_dict['sbnet_y']

        return data_dict
