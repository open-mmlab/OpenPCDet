import numpy as np
import torch
import torch.nn as nn
import time

from ..backbones_2d import BaseBEVBackbone
from .anchor_head_template import AnchorHeadTemplate

class SingleHead(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, rpn_head_cfg=None,
                 head_label_indices=None, separate_reg_config=None):
        super().__init__(rpn_head_cfg, input_channels)

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer('head_label_indices', head_label_indices)
        self.class_names = rpn_head_cfg['HEAD_CLS_NAME']

        if self.separate_reg_config is not None:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            num_middle_conv = self.separate_reg_config.NUM_MIDDLE_CONV
            num_middle_filter = self.separate_reg_config.NUM_MIDDLE_FILTER
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_conv):
                conv_cls_list.extend([
                    nn.Conv2d(
                        c_in, num_middle_filter,
                        kernel_size=3, stride=1, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(num_middle_filter),
                    nn.ReLU()
                ])
                c_in = num_middle_filter
            conv_cls_list.append(nn.Conv2d(
                c_in, self.num_anchors_per_location * self.num_class,
                kernel_size=3, stride=1, padding=1
            ))
            self.conv_cls = nn.Sequential(*conv_cls_list)

            for reg_config in self.separate_reg_config.REG_LIST:
                reg_name, reg_channel = reg_config.split(':')
                reg_channel = int(reg_channel)
                cur_conv_list = []
                c_in = input_channels
                for k in range(num_middle_conv):
                    cur_conv_list.extend([
                        nn.Conv2d(
                            c_in, num_middle_filter,
                            kernel_size=3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(num_middle_filter),
                        nn.ReLU()
                    ])
                    c_in = num_middle_filter

                cur_conv_list.append(nn.Conv2d(
                    c_in, self.num_anchors_per_location * int(reg_channel),
                    kernel_size=3, stride=1, padding=1, bias=True
                ))
                code_size_cnt += reg_channel
                self.conv_box[f'conv_{reg_name}'] = nn.Sequential(*cur_conv_list)
                self.conv_box_names.append(f'conv_{reg_name}')

            for m in self.conv_box.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            assert code_size_cnt == code_size, f'Code size does not match: {code_size_cnt}:{code_size}'
        else:
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
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = self.forward_cls_preds(spatial_features_2d)
        ret_dict = self.forward_remaining_preds(ret['sp2d'])
        return ret_dict

    def forward_cls_preds(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']
        ret_dict['sp2d'] = spatial_features_2d

        cls_preds = self.conv_cls(spatial_features_2d)

        if not self.use_multihead:
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = cls_preds.shape[2:]
            batch_size = cls_preds.shape[0]
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                       self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()

            # EXPLANATION 1
            # Commented this part to be compatible with imprecise computation
            # fix views is later used to fix this
            #cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        ret_dict['cls_preds'] = cls_preds

        return ret_dict


    def forward_remaining_preds(self, spatial_features_2d):
        ret_dict = {}

        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)

        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                       self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            # Please check EXPLANATION 1
            #box_preds = box_preds.view(batch_size, -1, self.code_size)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(
                    -1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4,
                                                                                                  2).contiguous()
                # Please check EXPLANATION 1
                #dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        else:
            dir_cls_preds = None

        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds

        return ret_dict

    def fix_views(self, pred_dict):
        if self.use_multihead:
            batch_size = pred_dict['cls_preds'].shape[0]
            pred_dict['cls_preds'] = pred_dict['cls_preds'].view(batch_size, -1, self.num_class)
            pred_dict['box_preds'] = pred_dict['box_preds'].view(batch_size, -1, self.code_size)
            if 'dir_cls_preds' in pred_dict:
                pred_dict['dir_cls_preds'] = \
                        pred_dict['dir_cls_preds'].view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)

        return pred_dict

class AnchorHeadMultiImprecise(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range, predict_boxes_when_training=predict_boxes_when_training
        )
        self.model_cfg = model_cfg
        self.separate_multihead = self.model_cfg.get('SEPARATE_MULTIHEAD', False)

        if self.model_cfg.get('SHARED_CONV_NUM_FILTER', None) is not None:
            self.shared_conv_alternatives = nn.ModuleList()
            shared_conv_num_filter = self.model_cfg.SHARED_CONV_NUM_FILTER
            for i in range(len(input_channels)): # this will come as a list from BaseBEVBackboneImprecise
                self.shared_conv_alternatives.append(nn.Sequential(
                    nn.Conv2d(sum(input_channels[:(i+1)]), shared_conv_num_filter, 3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(shared_conv_num_filter, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))
            c_in = [0] * len(input_channels)
            c_in[0] = self.model_cfg.SHARED_CONV_NUM_FILTER
            shared_conv_num_filter = c_in
        else:
            self.shared_conv_alternatives = None
            shared_conv_num_filter = input_channels
        self.make_multihead(shared_conv_num_filter)

        #TODO, I took these from the anchor sizes in the configuration file
        # of multi head point pillars nuscenes. This process can be automated.
        self.anchor_area_coeffs = torch.Tensor([
            [4.63 * 1.97, 1.],           # car, none
            [6.93 * 2.51, 6.37  * 2.85], # truck, construction_vehicle
            [10.5 * 2.94, 12.29 * 2.90], # bus, trailer
            [0.50 * 2.53, 1.],           # barrier, none
            [2.11 * 0.77, 1.70 * 0.60],  # motocycle, bicycle
            [0.73 * 0.67, 0.41 * 0.41],  # pedestrian, traffic_cone
        ])

    def make_multihead(self, input_channels):
        self.rpn_head_alternatives = nn.ModuleList()
        rpn_head_cfgs = self.model_cfg.RPN_HEAD_CFGS
        class_names = []
        for rpn_head_cfg in rpn_head_cfgs:
            class_names.extend(rpn_head_cfg['HEAD_CLS_NAME'])

        for i in range(len(input_channels)): # this will come as a list from BaseBEVBackboneImprecise
            rpn_heads = []
            for rpn_head_cfg in rpn_head_cfgs:
                num_anchors_per_location = sum([self.num_anchors_per_location[class_names.index(head_cls)]
                                                for head_cls in rpn_head_cfg['HEAD_CLS_NAME']])
                head_label_indices = torch.from_numpy(np.array([
                    self.class_names.index(cur_name) + 1 for cur_name in rpn_head_cfg['HEAD_CLS_NAME']
                ]))

                rpn_head = SingleHead(
                    self.model_cfg, sum(input_channels[:(i+1)]),
                    len(rpn_head_cfg['HEAD_CLS_NAME']) if self.separate_multihead else self.num_class,
                    num_anchors_per_location, self.box_coder.code_size, rpn_head_cfg,
                    head_label_indices=head_label_indices,
                    separate_reg_config=self.model_cfg.get('SEPARATE_REG_CONFIG', None)
                )
                rpn_heads.append(rpn_head)
            # I hope this nested list will work fine
            self.rpn_head_alternatives.append(nn.ModuleList(rpn_heads))
        self.num_heads = len(self.rpn_head_alternatives[0])

        self.heads_to_labels = [rh.head_label_indices.tolist() \
                for rh in self.rpn_head_alternatives[0]]
        print('heads_to_labels:', self.heads_to_labels)
        self.labels_to_heads = []
        for i, rh in enumerate(self.rpn_head_alternatives[0]):
            self.labels_to_heads.extend(len(rh.head_label_indices) * [i])
        print('labels_to_heads:', self.labels_to_heads)

        #self.cuda_streams = [ torch.cuda.Stream() for r in rpn_head_cfgs]

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

        spatial_features_2d = data_dict[f'spatial_features_2d_{cur_stg}']
        if self.shared_conv_alternatives is not None:
            spatial_features_2d = self.shared_conv_alternatives[cur_stg-1](spatial_features_2d)

        ret_dicts = []
        if 'heads_to_run' not in data_dict:
            for rpn_head in self.rpn_head_alternatives[cur_stg-1]:
                #with torch.cuda.stream(cs):
                ret_dicts.append(rpn_head.forward_cls_preds(spatial_features_2d))
        else:
            for h in data_dict['heads_to_run']:
                rpn_head = self.rpn_head_alternatives[cur_stg-1][h]
                ret_dicts.append(rpn_head.forward_cls_preds(spatial_features_2d))

        cls_preds = [ret_dict['cls_preds'] for ret_dict in ret_dicts]
        ret = {
            'cls_preds': cls_preds if self.separate_multihead else torch.cat(cls_preds, dim=1),
        }
        data_dict['cls_preds'] = ret['cls_preds']

        sp2d = [ret_dict['sp2d'] for ret_dict in ret_dicts]
        data_dict[f'sp2d_{cur_stg}'] = sp2d
        self.forward_ret_dict.update(ret)

        return data_dict

    def forward_remaining_preds(self, data_dict):
        cur_stg = data_dict["stages_executed"]

        if 'heads_to_run' not in data_dict:
            data_dict['heads_to_run'] = \
                    torch.arange(0, self.num_heads, dtype=torch.uint8, device='cpu')

        # Each rem forward takes 5.3 ms
        ret_dicts = []
        for i, h in enumerate(data_dict['heads_to_run']):
            rpn_head = self.rpn_head_alternatives[cur_stg-1][h]
            ret_dicts.append(rpn_head.forward_remaining_preds(data_dict[f'sp2d_{cur_stg}'][i]))

        box_preds = [ret_dict['box_preds'] for ret_dict in ret_dicts]
        ret = {
            'box_preds': box_preds if self.separate_multihead else torch.cat(box_preds, dim=1),
        }

        data_dict['box_preds'] = ret['box_preds']

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False):
            dir_cls_preds = [ret_dict['dir_cls_preds'] for ret_dict in ret_dicts]
            ret['dir_cls_preds'] = dir_cls_preds if self.separate_multihead else torch.cat(dir_cls_preds, dim=1)
            data_dict['dir_cls_preds'] = ret['dir_cls_preds']

        self.forward_ret_dict.update(ret)

        if self.training and data_dict["stages_executed"] == 1:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if self.training and cur_stg == 3 and self.predict_boxes_when_training:
            data_dict = self.gen_pred_boxes(data_dict)

        return data_dict

    def fix_views(self, data_dict):
        cur_stg = data_dict["stages_executed"]

        for i, h in enumerate(data_dict['heads_to_run']):
            rpn_head = self.rpn_head_alternatives[cur_stg-1][h]
            pd = {}
            for k in ['cls_preds', 'box_preds', 'dir_cls_preds']:
                if k in data_dict.keys():
                    pd[k] = data_dict[k][i]

            pd = rpn_head.fix_views(pd)
            for k,v in pd.items():
                data_dict[k][i] = v

        return data_dict

    def gen_pred_boxes(self, data_dict):
        if self.separate_multihead:
            data_dict = self.fix_views(data_dict)

        # backup self.anchors, modify them accordingly, and then restore afterward
        anchors_bkp = self.anchors.copy()  # shallow copy, which we want
        # the index 0 below can be 1 or 2 as well
        num_classes = [len(rh.head_label_indices) for rh in self.rpn_head_alternatives[0]]

        anchors_tmp = []
        for i in data_dict['heads_to_run']:
            shift = sum(num_classes[:i])
            numc = len(self.rpn_head_alternatives[0][i].head_label_indices)
            anchors_tmp.extend(self.anchors[shift:(shift+numc)])

        self.anchors = anchors_tmp

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=data_dict['cls_preds'], 
            box_preds=data_dict['box_preds'], 
            dir_cls_preds=data_dict.get('dir_cls_preds', None)
        )

        self.anchors = anchors_bkp  # restore

        if isinstance(batch_cls_preds, list):
            multihead_label_mapping = []
            for idx in data_dict['heads_to_run']:
                multihead_label_mapping.append(self.rpn_head_alternatives[0][idx].head_label_indices)

            data_dict['multihead_label_mapping'] = multihead_label_mapping

        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        if 'cls_preds_normalized' not in data_dict:
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def get_cls_layer_loss(self):
        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        if 'pos_cls_weight' in loss_weights:
            pos_cls_weight = loss_weights['pos_cls_weight']
            neg_cls_weight = loss_weights['neg_cls_weight']
        else:
            pos_cls_weight = neg_cls_weight = 1.0

        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]
        batch_size = int(cls_preds[0].shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0 * neg_cls_weight

        cls_weights = (negative_cls_weights + pos_cls_weight * positives).float()

        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()

        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds[0].dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        start_idx = c_idx = 0
        cls_losses = 0

        for idx, cls_pred in enumerate(cls_preds):
            cur_num_class = self.rpn_head_alternatives[0][idx].num_class
            cls_pred = cls_pred.view(batch_size, -1, cur_num_class)
            if self.separate_multihead:
                one_hot_target = one_hot_targets[:, start_idx:start_idx + cls_pred.shape[1],
                                 c_idx:c_idx + cur_num_class]
                c_idx += cur_num_class
            else:
                one_hot_target = one_hot_targets[:, start_idx:start_idx + cls_pred.shape[1]]
            cls_weight = cls_weights[:, start_idx:start_idx + cls_pred.shape[1]]
            cls_loss_src = self.cls_loss_func(cls_pred, one_hot_target, weights=cls_weight)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss = cls_loss * loss_weights['cls_weight']
            cls_losses += cls_loss
            start_idx += cls_pred.shape[1]
        assert start_idx == one_hot_targets.shape[1]
        tb_dict = {
            'rpn_loss_cls': cls_losses.item()
        }
        return cls_losses, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if not isinstance(box_preds, list):
            box_preds = [box_preds]
        batch_size = int(box_preds[0].shape[0])

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                     for anchor in self.anchors], dim=0
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)

        start_idx = 0
        box_losses = 0
        tb_dict = {}
        for idx, box_pred in enumerate(box_preds):
            box_pred = box_pred.view(
                batch_size, -1,
                box_pred.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_pred.shape[-1]
            )
            box_reg_target = box_reg_targets[:, start_idx:start_idx + box_pred.shape[1]]
            reg_weight = reg_weights[:, start_idx:start_idx + box_pred.shape[1]]
            # sin(a - b) = sinacosb-cosasinb
            if box_dir_cls_preds is not None:
                box_pred_sin, reg_target_sin = self.add_sin_difference(box_pred, box_reg_target)
                loc_loss_src = self.reg_loss_func(box_pred_sin, reg_target_sin, weights=reg_weight)  # [N, M]
            else:
                loc_loss_src = self.reg_loss_func(box_pred, box_reg_target, weights=reg_weight)  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_losses += loc_loss
            tb_dict['rpn_loss_loc'] = tb_dict.get('rpn_loss_loc', 0) + loc_loss.item()

            if box_dir_cls_preds is not None:
                if not isinstance(box_dir_cls_preds, list):
                    box_dir_cls_preds = [box_dir_cls_preds]
                dir_targets = self.get_direction_target(
                    anchors, box_reg_targets,
                    dir_offset=self.model_cfg.DIR_OFFSET,
                    num_bins=self.model_cfg.NUM_DIR_BINS
                )
                box_dir_cls_pred = box_dir_cls_preds[idx]
                dir_logit = box_dir_cls_pred.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
                weights = positives.type_as(dir_logit)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)

                weight = weights[:, start_idx:start_idx + box_pred.shape[1]]
                dir_target = dir_targets[:, start_idx:start_idx + box_pred.shape[1]]
                dir_loss = self.dir_loss_func(dir_logit, dir_target, weights=weight)
                dir_loss = dir_loss.sum() / batch_size
                dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
                box_losses += dir_loss
                tb_dict['rpn_loss_dir'] = tb_dict.get('rpn_loss_dir', 0) + dir_loss.item()
            start_idx += box_pred.shape[1]
        return box_losses, tb_dict
