from typing import Iterable, List

import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadDualClass(AnchorHeadTemplate):
    def __init__(
        self,
        model_cfg,
        input_channels: int,
        num_class: int,
        num_aux_class: int,
        class_names: List[List[str]],
        grid_size: Iterable[int],
        point_cloud_range: Iterable[int],
        predict_boxes_when_training=True,
        **kwargs,
    ):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_aux_class = num_aux_class
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_aux_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_aux_class,
            kernel_size=1,
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.constant_(self.conv_aux_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        aux_cls_preds = self.conv_aux_cls(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        aux_cls_preds = aux_cls_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['aux_cls_preds'] = aux_cls_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            # get the labels for all anchor boxes. targets_dict will contain the following
            # keys: box_cls_labels, box_reg_targets, reg_weights
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            batch_aux_cls_preds, _ = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=aux_cls_preds, box_preds=box_preds, dir_cls_preds=None,
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_aux_cls_preds'] = batch_aux_cls_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_cls_layer_loss(self):
        cls_fdi_loss = self.get_cls_layer_loss_one_set(
            cls_preds=self.forward_ret_dict["cls_preds"],
            box_cls_labels=self.forward_ret_dict["box_cls_labels"],
            num_class=self.num_class,
        )
        aux_cls_loss = self.get_cls_layer_loss_one_set(
            cls_preds=self.forward_ret_dict["aux_cls_preds"],
            box_cls_labels=self.forward_ret_dict["box_aux_cls_labels"],
            num_class=self.num_aux_class,
        )
        # combine losses
        cls_loss = cls_fdi_loss + aux_cls_loss
        tb_dict = {
            'rpn_loss_aux_cls': aux_cls_loss.item(),
            'rpn_loss_fdi_cls': cls_fdi_loss.item(),
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
