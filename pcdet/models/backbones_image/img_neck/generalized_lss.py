import torch
import torch.nn as nn
import torch.nn.functional as F
from ...model_utils.basic_block_2d import BasicBlock2D


class GeneralizedLSSFPN(nn.Module):
    """
        This module implements FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL

        self.in_channels = in_channels

        if end_level == -1:
            self.backbone_end_level = num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = BasicBlock2D(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels, kernel_size=1, bias = False
            )
            fpn_conv = BasicBlock2D(out_channels,out_channels, kernel_size=3, padding=1, bias = False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        # upsample -> cat -> conv1x1 -> conv3x3
        inputs = batch_dict['image_features']
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        batch_dict['image_fpn'] = tuple(outs)
        return batch_dict
