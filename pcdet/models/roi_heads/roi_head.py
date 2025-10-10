import torch
import torch.nn as nn
from ..backbones_3d.pfe.voxel_set_abstraction import bilinear_interpolate_torch
from ...utils.box_torch_ops import center_to_corner_box2d
from .roi_head_template import RoIHeadTemplate

class RoIHead(RoIHeadTemplate):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_class=1, **kwargs):
        # backbone_channels, point_cloud_range
        super().__init__(num_class=num_class, model_cfg=model_cfg)

        out_stride = model_cfg.FEATURE_MAP_STRIDE
        pc_range_start = point_cloud_range[:2]
        self.add_box_param = model_cfg.ADD_BOX_PARAM
        self.num_point = model_cfg.NUM_POINTS
        self.bev_feature_extractor = BEVFeatureExtractor(pc_range_start, voxel_size, out_stride, self.num_point)

        fc_input_dim = (model_cfg.BEV_FEATURE_DIM * self.num_point) + (self.box_coder.code_size + 1 if self.add_box_param else 0)
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(fc_input_dim if k==0 else self.model_cfg.SHARED_FC[k-1], self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, batch_dict):
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_scores'] = targets_dict['roi_scores']

        B,N,_ = batch_dict['rois'].shape
        batch_centers = self.get_box_center(batch_dict['rois'])
        bev_features = self.bev_feature_extractor(batch_dict['spatial_features_2d'], batch_centers)
        first_stage_pred = batch_dict['rois']

        # target_dict = self.reorder_first_stage_pred_and_feature(batch_dict, first_stage_pred, bev_features)

        # RoI aware pooling
        # if self.add_box_param:
        if self.add_box_param:
            pooled_features = torch.cat([bev_features, batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)
        else:
            pooled_features = bev_features

        pooled_features = pooled_features.reshape(B*N,1,-1).contiguous()
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict    

    
    def get_box_center(self, boxes):
        # box [List]
        centers = []
        for box in boxes:            
            if self.num_point == 1:
                centers.append(box[:, :3])
                
            elif self.num_point == 5:
                center2d = box[:, :2]
                height = box[:, 2:3]
                dim2d = box[:, 3:5]
                rotation_y = box[:, -1]

                corners = center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box[:,:3], front_middle, back_middle, left_middle, right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers




class BEVFeatureExtractor(nn.Module): 
    def __init__(self, pc_start, voxel_size, out_stride, num_point):
        super().__init__()
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride
        self.num_point = num_point

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2

    def forward(self, bev_features, batch_centers):

        batch_size = len(bev_features)
        ret_maps = []

        for batch_idx in range(batch_size):
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            
            # N x C 
            feature_map = bilinear_interpolate_torch(bev_features[batch_idx],
             xs, ys)

            if self.num_point > 1:
                section_size = len(feature_map) // self.num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(self.num_point)], dim=1)

            ret_maps.append(feature_map)

        return torch.stack(ret_maps) 