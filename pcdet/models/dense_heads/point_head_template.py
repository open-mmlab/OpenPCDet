import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import loss_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.reg_loss_func = F.smooth_l1_loss if losses_cfg.get('LOSS_REG', None) == 'smooth-l1' else F.l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, 7].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            if ret_box_labels:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(points_single[fg_flag], gt_box_of_fg_points)
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def get_cls_layer_loss(self):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        tb_dict = {
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        }
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        return point_loss_part, {'point_loss_part': point_loss_part.item()}

    def forward(self, **kwargs):
        raise NotImplementedError
