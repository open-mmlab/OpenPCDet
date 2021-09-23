import numpy as np
import torch
import torch.nn as nn
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_utils
import torch.nn.functional as F
import pdb
# from tools.visual_utils import visualize_utils as V



class AnchorBoundary(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, logger=None, global_cfg=None, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training, logger=logger, global_cfg=global_cfg
        )
            
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_boundary = nn.Conv2d(
            input_channels, self.model_cfg.NUM_BOUNDARY,
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
        self.init_weights()

    def get_line_params(self, corner_points):
        """
        Args:
            corner_points: (N, M, 4, 2)
            x                   y
            ^                   ^
            |                   |
            |                   |
         ---------> y ===> -----------> x
            |                   |
            |                   |
            |                   |
        returns:
            line_param: (N, M, 4, 4)
        """
        # we need to replace x with y according to the lidar coord
        # pdb.set_trace()
        import copy
        temp_x = copy.deepcopy(corner_points[:, :, :, 0])
        # corner_points[:, :, :, 0], corner_points[:, :, :, 1] = corner_points[:, :, :, 1], corner_points[:, :, :, 0]
        corner_points[:, :, :, 0] = corner_points[:, :, :, 1]
        corner_points[:, :, :, 1] = temp_x 
        shift_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        
        param_A = torch.unsqueeze((corner_points[:, :, shift_index[:, 0], 1] - corner_points[:, :, shift_index[:, 1], 1]), dim=-1)
        param_B = torch.unsqueeze((corner_points[:, :, shift_index[:, 1], 0] - corner_points[:, :, shift_index[:, 0], 0]), dim=-1)
        param_C = torch.unsqueeze((corner_points[:, :, shift_index[:, 0], 0] * corner_points[:, :, shift_index[:, 1], 1] - corner_points[:, :, shift_index[:, 1], 0] * corner_points[:, :, shift_index[:, 0], 1]), dim=-1)
        param_coef = torch.sqrt(torch.pow(param_A, 2) + torch.pow(param_B, 2) + 1e-8)
        # pdb.set_trace()
        line_param = torch.cat((param_A, param_B, param_C, param_coef), dim=-1)
        
        return line_param
    
    def compute_boundary_offset(self, feature_coord, inside_mask, line_params, gt_bbox=None, points=None):
        """
        Args:
            feature_coord: (N, H*W, 3)
            inside_mask: (N, H*W)
            line_params: (N, M, 4, 4)
            gt_bbox: (N, M, 7)
        returns:
            boundary_offset_targets: (N, H*W, 4)
        """
        # offset coef N M 4
        N, point_num = feature_coord.shape[0], feature_coord.shape[1]
        boundary_offset_targets = torch.zeros(N, point_num, 4).to(feature_coord.device)
        for i in range(N):
            point_inside = inside_mask[i] != -1
            bbox_index = inside_mask[i][point_inside].long()

            # inside_num, 4, 4
            target_line =  line_params[i][bbox_index].to(feature_coord.device)
            current_feature_coord = feature_coord[i][point_inside].unsqueeze(dim=-1)
            
            # TODO: implement normalized boundary offset prediction
            # bbox_width = gt_bbox[i, :, 3] * 2
            # bbox_length = gt_bbox[i, :, 4] * 2
            # normalize_coef = torch.cat()

            # replace x with y according to the lidar coord and get inside_num * 4
            offset = (current_feature_coord[:, 1] * target_line[:, :, 0] + current_feature_coord[:, 0] * target_line[:, :, 1] + target_line[:, :, 2]) / target_line[:, :, 3]
            # normalize boundary offset
            boundary_offset_targets[i][point_inside] = offset
            # pdb.set_trace()                

        return boundary_offset_targets

    def get_boundary_targets(self, feature_map, boxes, points=None):
        """
        Args:
            lidar coord range: 496 * 432 -> (y, x)
        """
        # this is a hard code here
        voxel_scale = 0.16
        origin_size = self.grid_size[:2]    # (432， 496)   x, y

        B, C, H, W = feature_map.shape      # (4, 384, 248, 216)
        # transform into lidar coord
        coord_x = torch.arange(W, device=feature_map.device) * (origin_size[0] / W) * voxel_scale
        coord_y = (torch.arange(H, device=feature_map.device) - H // 2) * (origin_size[1] / H) * voxel_scale
        grid_x, grid_y = torch.meshgrid(coord_x, coord_y)
        bev_coord = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
        coor_z = torch.zeros(H*W, 1).to(feature_map.device)
        # [B, H*W, 3]
        feature_coord = torch.cat((bev_coord, coor_z), dim=-1).view(1, -1, 3).repeat(B, 1, 1)
        # get inside mask (N, H*W)
        inside_index = roiaware_pool3d_utils.points_in_boxes_gpu(feature_coord, boxes[:, :, :-1])

        # get bev in lidar coord
        boxes_corner_coord = torch.zeros(B, boxes.shape[1], 8, 3)
        for i in range(B):
            boxes_corner_coord[i] = box_utils.boxes_to_corners_3d(boxes[i, :, :-1])
        
        boxes_corner_coord = boxes_corner_coord[:, :, :4, :2]
        line_params = self.get_line_params(boxes_corner_coord)

        # for visualization
        if self.global_cfg.VISUALIZE and self.global_cfg.DEBUG:
            import cv2
            import skimage
            for i in range(B):
                canvas = np.zeros((H, W, 3))
                for index in range(boxes_corner_coord[i].shape[0]):
                    bev_feature_coor = boxes_corner_coord[i, index]
                    for j in range(0, 4):
                        x1, y1 = bev_feature_coor[j, 0] / voxel_scale / 2, bev_feature_coor[j, 1] / voxel_scale / 2
                        x2, y2 = bev_feature_coor[(j + 1) % 4, 0] / voxel_scale / 2 , bev_feature_coor[(j + 1) % 4, 1] / voxel_scale / 2

                        cv2.line(canvas, (y1, x1), (y2, x2), (42,42,128),1)
                save_name = "vis_%s.png" % (i)
                skimage.io.imsave("./"+save_name, (canvas).astype(np.uint8))
        
        self.boundary_offset_targets = self.compute_boundary_offset(feature_coord, inside_index, line_params, boxes, points).view(B, H, W, 4)

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    
    def get_boundary_layer_loss(self):
        boundary_preds = self.forward_ret_dict['boundary_preds']
        boundary_targets = self.forward_ret_dict['boundary_offset_targets']
        loss_weight = torch.ones(boundary_preds.shape).to(boundary_preds.device)
        positive_mask = (boundary_targets.sum(dim=-1) != 0).unsqueeze(dim=-1).float()
        loss_weight = loss_weight * positive_mask

        boundary_loss = F.smooth_l1_loss(loss_weight * boundary_preds, loss_weight * boundary_targets, reduction='sum') / loss_weight.sum()
        tb_dict_boundary = {
            'rpn_loss_boundary': boundary_loss.item()
        }
        # pdb.set_trace()
        
        return boundary_loss, tb_dict_boundary
    
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        boundary_loss, tb_dict_boundary = self.get_boundary_layer_loss()
        
        tb_dict.update(tb_dict_box)
        tb_dict.update(tb_dict_boundary)
        rpn_loss = cls_loss + box_loss + boundary_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        boundary_preds = self.conv_boundary(spatial_features_2d)
        
        # get boundary offset
        offset_targets = self.get_boundary_targets(spatial_features_2d, data_dict['gt_boxes'], data_dict['points'])

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        boundary_preds = boundary_preds.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['boundary_preds'] = boundary_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            # self.forward_ret_dict.update(self.boundary_offset_targets)
            self.forward_ret_dict['boundary_offset_targets'] = self.boundary_offset_targets

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
