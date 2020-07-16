import numpy as np
import torch.nn as nn
from .anchor_head_template import AnchorHeadTemplate
from ..backbones_2d import BaseBEVBackbone
import torch


class SingleHead(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, encode_conv_cfg=None):
        super().__init__(encode_conv_cfg, input_channels)

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get('USE_MULTI_HEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                       self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class).unsqueeze(-1)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4, 2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds

        return ret_dict


class AnchorHeadMulti(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training
        )
        self.model_cfg = model_cfg
        self.make_multihead(input_channels)

    def make_multihead(self, input_channels):
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        rpn_heads = []
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])
        for rpn_head_cfg in rpn_head_cfgs:
            num_anchors_per_location = sum([self.num_anchors_per_location[class_names.index(head_cls)] for head_cls in rpn_head_cfg['HEAD_CLS_NAME']])
            rpn_head = SingleHead(self.model_cfg, input_channels, self.num_class, num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg)
            rpn_heads.append(rpn_head)
        self.rpn_heads = nn.ModuleList(rpn_heads)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        ret_dicts = []
        for rpn_head in self.rpn_heads:
            ret_dicts.append(rpn_head(spatial_features_2d))

        cls_preds = torch.cat([ret_dict['cls_preds'] for ret_dict in ret_dicts], dim=1)
        box_preds = torch.cat([ret_dict['box_preds'] for ret_dict in ret_dicts], dim=1)
        ret = {
            'cls_preds': cls_preds,
            'box_preds': box_preds,

        }
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
            dir_cls_preds = torch.cat([ret_dict['dir_cls_preds'] for ret_dict in ret_dicts], dim=1)
            ret['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
 
        self.forward_ret_dict.update(ret)
       
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
