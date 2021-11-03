import numpy as np
import torch.nn as nn
import torch

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingleImprecise(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls_alternatives = nn.ModuleList()
        self.conv_box_alternatives = nn.ModuleList()
        # input channels correspond to num_upsample_filters
        for i in range(len(input_channels)):
            self.conv_cls_alternatives.append(nn.Conv2d(
                sum(input_channels[:(i+1)]), self.num_anchors_per_location * self.num_class,
                kernel_size=1
            ))
            self.conv_box_alternatives.append(nn.Conv2d(
                sum(input_channels[:(i+1)]), self.num_anchors_per_location * self.box_coder.code_size,
                kernel_size=1
            ))

        self.conv_dir_cls_alternatives = None
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_alternatives = nn.ModuleList()
            for i in range(len(input_channels)):
                self.conv_dir_cls_alternatives.append(nn.Conv2d(
                    sum(input_channels[:(i+1)]),
                    self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                    kernel_size=1
                ))
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        for conv_cls in self.conv_cls_alternatives:
            nn.init.constant_(conv_cls.bias, -np.log((1 - pi) / pi))
        for conv_box in self.conv_box_alternatives:
            nn.init.normal_(conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        return self.forward_remaining_preds(self.forward_cls_preds(data_dict))

    def forward_cls_preds(self, data_dict):
        cur_stg = data_dict["stages_executed"]
    
        if f'spatial_features_2d_{cur_stg}' not in data_dict:
            uplist = [data_dict[f"up{i}"] for i in range(1, cur_stg+1)]
            if len(uplist) == 1:
                data_dict[f'spatial_features_2d_{cur_stg}'] = uplist[0]
            else:
                data_dict[f'spatial_features_2d_{cur_stg}'] = torch.cat(uplist, dim=1)

        cls_preds = self.conv_cls_alternatives[cur_stg-1](data_dict[f'spatial_features_2d_{cur_stg}'])
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        self.forward_ret_dict['cls_preds'] = cls_preds # needed for training
        data_dict['cls_preds'] = cls_preds
    
        return data_dict
       
    def forward_remaining_preds(self, data_dict):
        cur_stg = data_dict["stages_executed"]

        box_preds = self.conv_box_alternatives[cur_stg-1](data_dict[f'spatial_features_2d_{cur_stg}'])
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        self.forward_ret_dict['box_preds'] = box_preds # needed for training
        data_dict['box_preds'] = box_preds

        if self.conv_dir_cls_alternatives is not None:
            dir_cls_preds = self.conv_dir_cls_alternatives[cur_stg-1]( \
                    data_dict[f'spatial_features_2d_{cur_stg}'])
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
            data_dict['dir_cls_preds'] = dir_cls_preds
        else:
            data_dict['dir_cls_preds'] = None

        # Only do it once for each model forward, no need to repeat
        if self.training and cur_stg == 1:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if self.training and cur_stg == 3 and self.predict_boxes_when_training:
            data_dict = self.gen_pred_boxes(data_dict)

        return data_dict

    def gen_pred_boxes(self, data_dict):
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=data_dict['cls_preds'],
            box_preds=data_dict['box_preds'],
            dir_cls_preds=data_dict['dir_cls_preds']
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False

        return data_dict
