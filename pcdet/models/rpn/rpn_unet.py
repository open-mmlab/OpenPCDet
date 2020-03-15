import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
from functools import partial

from ..model_utils.resnet_utils import SparseBasicBlock
from ...config import cfg
from ...utils import common_utils, loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class UNetHead(nn.Module):
    def __init__(self, unet_target_cfg):
        super().__init__()
        self.gt_extend_width = unet_target_cfg.GT_EXTEND_WIDTH
        if 'MEAN_SIZE' in unet_target_cfg:
            self.mean_size = unet_target_cfg.MEAN_SIZE

        self.target_generated_on = unet_target_cfg.GENERATED_ON

        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        self.forward_ret_dict = None

    def assign_targets(self, batch_points, gt_boxes, generate_bbox_reg_labels=False):
        """
        :param points: [(N1, 3), (N2, 3), ...]
        :param gt_boxes: (B, M, 8)
        :param gt_classes: (B, M)
        :param gt_names: (B, M)
        :return:
        """
        batch_size = gt_boxes.shape[0]
        cls_labels_list, part_reg_labels_list, bbox_reg_labels_list = [], [], []
        for k in range(batch_size):
            if True or self.target_generated_on == 'head_cpu':
                cur_cls_labels, cur_part_reg_labels, cur_bbox_reg_labels = self.generate_part_targets_cpu(
                    points=batch_points[k],
                    gt_boxes=gt_boxes[k][:, 0:7],
                    gt_classes=gt_boxes[k][:, 7],
                    generate_bbox_reg_labels=generate_bbox_reg_labels
                )
            else:
                raise NotImplementedError

            cls_labels_list.append(cur_cls_labels)
            part_reg_labels_list.append(cur_part_reg_labels)
            bbox_reg_labels_list.append(cur_bbox_reg_labels)
        cls_labels = torch.cat(cls_labels_list, dim=0).cuda()
        part_reg_labels = torch.cat(part_reg_labels_list, dim=0).cuda()
        bbox_reg_labels = torch.cat(bbox_reg_labels_list, dim=0).cuda() if generate_bbox_reg_labels else None

        targets_dict = {
            'seg_labels': cls_labels,
            'part_labels': part_reg_labels,
            'bbox_reg_labels': bbox_reg_labels
        }

        return targets_dict

    def generate_part_targets_cpu(self, points, gt_boxes, gt_classes, generate_bbox_reg_labels=False):
        """
        :param voxel_centers: (N, 3) [x, y, z]
        :param gt_boxes: (M, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
        """
        k = gt_boxes.__len__() - 1
        while k > 0 and gt_boxes[k].sum() == 0:
            k -= 1
        gt_boxes = gt_boxes[:k + 1]
        gt_classes = gt_classes[:k + 1]

        extend_gt_boxes = common_utils.enlarge_box3d(gt_boxes, extra_width=self.gt_extend_width)
        cls_labels = torch.zeros(points.shape[0]).int()
        part_reg_labels = torch.zeros((points.shape[0], 3)).float()
        bbox_reg_labels = torch.zeros((points.shape[0], 7)).float() if generate_bbox_reg_labels else None

        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points, gt_boxes).long()
        extend_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points, extend_gt_boxes).long()
        for k in range(gt_boxes.shape[0]):
            fg_pt_flag = point_indices[k] > 0
            fg_points = points[fg_pt_flag]
            cls_labels[fg_pt_flag] = gt_classes[k]

            # enlarge the bbox3d, ignore nearby points
            fg_enlarge_flag = extend_point_indices[k] > 0
            ignore_flag = fg_pt_flag ^ fg_enlarge_flag
            cls_labels[ignore_flag] = -1

            # part offset labels
            transformed_points = fg_points - gt_boxes[k, 0:3]
            transformed_points = common_utils.rotate_pc_along_z_torch(
                transformed_points.view(1, -1, 3), -gt_boxes[k, 6]
            )
            part_reg_labels[fg_pt_flag] = (transformed_points / gt_boxes[k, 3:6]) + torch.tensor([0.5, 0.5, 0]).float()

            if generate_bbox_reg_labels:
                # rpn bbox regression target
                center3d = gt_boxes[k, 0:3].clone()
                center3d[2] += gt_boxes[k][5] / 2  # shift to center of 3D boxes
                bbox_reg_labels[fg_pt_flag, 0:3] = center3d - fg_points
                bbox_reg_labels[fg_pt_flag, 6] = gt_boxes[k, 6]  # dy

                cur_mean_size = torch.tensor(self.mean_size[cfg.CLASS_NAMES[gt_classes[k] - 1]])
                bbox_reg_labels[fg_pt_flag, 3:6] = (gt_boxes[k, 3:6] - cur_mean_size) / cur_mean_size

        return cls_labels, part_reg_labels, bbox_reg_labels

    def get_loss(self, forward_ret_dict=None):
        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict

        tb_dict = {}
        u_seg_preds = forward_ret_dict['u_seg_preds'].squeeze(dim=-1)
        u_reg_preds = forward_ret_dict['u_reg_preds']

        # segmentation and part prediction losses
        u_cls_labels, u_reg_labels = forward_ret_dict['seg_labels'], forward_ret_dict['part_labels']
        u_cls_target = (u_cls_labels > 0).float()
        pos_mask = u_cls_labels > 0
        pos = pos_mask.float()
        neg = (u_cls_labels == 0).float()
        u_cls_weights = pos + neg
        pos_normalizer = pos.sum()
        u_cls_weights = u_cls_weights / torch.clamp(pos_normalizer, min=1.0)
        u_loss_cls = self.cls_loss_func(u_seg_preds, u_cls_target, weights=u_cls_weights)
        u_loss_cls_pos = (u_loss_cls * pos).sum()
        u_loss_cls_neg = (u_loss_cls * neg).sum()
        u_loss_cls = u_loss_cls.sum()

        loss_unet = u_loss_cls

        if pos_normalizer > 0:
            u_loss_reg = F.binary_cross_entropy(torch.sigmoid(u_reg_preds[pos_mask]), u_reg_labels[pos_mask])
            loss_unet += u_loss_reg
            tb_dict['rpn_u_loss_reg'] = u_loss_reg.item()

        tb_dict['rpn_loss_u_cls'] = u_loss_cls.item()
        tb_dict['rpn_loss_u_cls_pos'] = u_loss_cls_pos.item()
        tb_dict['rpn_loss_u_cls_neg'] = u_loss_cls_neg.item()
        tb_dict['rpn_loss_unet'] = loss_unet.item()
        tb_dict['rpn_pos_num'] = pos_normalizer.item()

        return loss_unet, tb_dict


class UNetV0(UNetHead):
    def __init__(self, input_channels, **kwargs):
        super().__init__(unet_target_cfg=cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )

        self.seg_cls_layer = nn.Linear(16, 1, bias=True)
        self.seg_reg_layer = nn.Linear(16, 3, bias=True)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        :param x: x.features (N, C1)
        :param out_channels: C2
        :return:
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        ret_dict = {'spatial_features': spatial_features}

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        seg_features = x_up1.features

        seg_cls_preds = self.seg_cls_layer(seg_features)  # (N, 1)
        seg_reg_preds = self.seg_reg_layer(seg_features)  # (N, 3)

        ret_dict.update({'u_seg_preds': seg_cls_preds, 'u_reg_preds': seg_reg_preds, 'seg_features': seg_features})

        if self.training:
            if self.target_generated_on == 'dataset':
                targets_dict = {
                    'seg_labels': kwargs['seg_labels'],
                    'part_labels': kwargs['part_labels'],
                    'bbox_reg_labels': kwargs.get('bbox_reg_labels', None)
                }
            else:
                batch_size = x_up1.batch_size
                bs_idx, coords = x_up1.indices[:, 0].cpu(), x_up1.indices[:, 1:].cpu()
                voxel_size = torch.tensor(cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE)
                pc_range = torch.tensor(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
                voxel_centers = (coords[:, [2, 1, 0]].float() + 0.5) * voxel_size + pc_range[0:3]
                batch_points = [voxel_centers[bs_idx == k] for k in range(batch_size)]
                targets_dict = self.assign_targets(
                    batch_points=batch_points,
                    gt_boxes=kwargs['gt_boxes'].cpu()
                )

            ret_dict['seg_labels'] = targets_dict['seg_labels']
            ret_dict['part_labels'] = targets_dict['part_labels']
            ret_dict['bbox_reg_labels'] = targets_dict.get('bbox_reg_labels', None)

        self.forward_ret_dict = ret_dict
        return ret_dict


class UNetV2(UNetHead):
    def __init__(self, input_channels, **kwargs):
        super().__init__(unet_target_cfg=cfg.MODEL.RPN.BACKBONE.TARGET_CONFIG)

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = self.post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0 if cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE[-1] in [0.1, 0.2] else (1, 0, 0)

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )

        self.seg_cls_layer = nn.Linear(16, 1, bias=True)
        self.seg_reg_layer = nn.Linear(16, 3, bias=True)

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        :param x: x.features (N, C1)
        :param out_channels: C2
        :return:
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0,
                       conv_type='subm', norm_fn=None):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                norm_fn(out_channels),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    def forward(self, input_sp_tensor, **kwargs):
        """
        :param voxel_features:  (N, C)
        :param coors:   (N, 4)  [batch_idx, z_idx, y_idx, x_idx],  sparse_shape: (z_size, y_size, x_size)
        :param batch_size:
        :return:
        """
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        ret_dict = {'spatial_features': spatial_features}

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        seg_features = x_up1.features

        seg_cls_preds = self.seg_cls_layer(seg_features)  # (N, 1)
        seg_reg_preds = self.seg_reg_layer(seg_features)  # (N, 3)

        ret_dict.update({'u_seg_preds': seg_cls_preds, 'u_reg_preds': seg_reg_preds, 'seg_features': seg_features})

        if self.training:
            if self.target_generated_on == 'dataset':
                targets_dict = {
                    'seg_labels': kwargs['seg_labels'],
                    'part_labels': kwargs['part_labels'],
                    'bbox_reg_labels': kwargs.get('bbox_reg_labels', None)
                }
            else:
                batch_size = x_up1.batch_size
                bs_idx, coords = x_up1.indices[:, 0].cpu(), x_up1.indices[:, 1:].cpu()
                voxel_size = torch.tensor(cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE)
                pc_range = torch.tensor(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
                voxel_centers = (coords[:, [2, 1, 0]].float() + 0.5) * voxel_size + pc_range[0:3]
                batch_points = [voxel_centers[bs_idx == k] for k in range(batch_size)]
                targets_dict2 = self.assign_targets(
                    batch_points=batch_points,
                    gt_boxes=kwargs['gt_boxes'].cpu()
                )

            ret_dict['seg_labels'] = targets_dict['seg_labels']
            ret_dict['part_labels'] = targets_dict['part_labels']
            ret_dict['bbox_reg_labels'] = targets_dict.get('bbox_reg_labels', None)

        self.forward_ret_dict = ret_dict
        return ret_dict
