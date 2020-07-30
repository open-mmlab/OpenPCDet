import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import common_utils


class ATSSTargetAssigner(object):
    """
    Reference: https://arxiv.org/abs/1912.02424
    """
    def __init__(self, topk, box_coder, match_height=False):
        self.topk = topk
        self.box_coder = box_coder
        self.match_height = match_height

    def assign_targets(self, anchors_list, gt_boxes_with_classes, use_multihead=False):
        """
        Args:
            anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        if not isinstance(anchors_list, list):
            anchors_list = [anchors_list]
            single_set_of_anchor = True
        else:
            single_set_of_anchor = len(anchors_list) == 1
        cls_labels_list, reg_targets_list, reg_weights_list = [], [], []
        for anchors in anchors_list:
            batch_size = gt_boxes_with_classes.shape[0]
            gt_classes = gt_boxes_with_classes[:, :, -1]
            gt_boxes = gt_boxes_with_classes[:, :, :-1]
            if use_multihead:
                anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
            else:
                anchors = anchors.view(-1, anchors.shape[-1])
            cls_labels, reg_targets, reg_weights = [], [], []
            for k in range(batch_size):
                cur_gt = gt_boxes[k]
                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1
                cur_gt = cur_gt[:cnt + 1]

                cur_gt_classes = gt_classes[k][:cnt + 1]
                cur_cls_labels, cur_reg_targets, cur_reg_weights = self.assign_targets_single(
                    anchors, cur_gt, cur_gt_classes
                )
                cls_labels.append(cur_cls_labels)
                reg_targets.append(cur_reg_targets)
                reg_weights.append(cur_reg_weights)

            cls_labels = torch.stack(cls_labels, dim=0)
            reg_targets = torch.stack(reg_targets, dim=0)
            reg_weights = torch.stack(reg_weights, dim=0)
            cls_labels_list.append(cls_labels)
            reg_targets_list.append(reg_targets)
            reg_weights_list.append(reg_weights)

        if single_set_of_anchor:
            ret_dict = {
                'box_cls_labels': cls_labels_list[0],
                'box_reg_targets': reg_targets_list[0],
                'reg_weights': reg_weights_list[0]
            }
        else:
            ret_dict = {
                'box_cls_labels': torch.cat(cls_labels_list, dim=1),
                'box_reg_targets': torch.cat(reg_targets_list, dim=1),
                'reg_weights': torch.cat(reg_weights_list, dim=1)
            }
        return ret_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes):
        """
        Args:
            anchors: (N, 7) [x, y, z, dx, dy, dz, heading]
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
            gt_classes: (M)
        Returns:

        """
        num_anchor = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        # select topk anchors for each gt_boxes
        if self.match_height:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])  # (N, M)
        else:
            ious = iou3d_nms_utils.boxes_iou_bev(anchors[:, 0:7], gt_boxes[:, 0:7])

        distance = (anchors[:, None, 0:3] - gt_boxes[None, :, 0:3]).norm(dim=-1)  # (N, M)
        _, topk_idxs = distance.topk(self.topk, dim=0, largest=False)  # (K, M)
        candidate_ious = ious[topk_idxs, torch.arange(num_gt)]  # (K, M)
        iou_mean_per_gt = candidate_ious.mean(dim=0)
        iou_std_per_gt = candidate_ious.std(dim=0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt + 1e-6
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]  # (K, M)

        # check whether anchor_center in gt_boxes, only check BEV x-y axes
        candidate_anchors = anchors[topk_idxs.view(-1)]  # (KxM, 7)
        gt_boxes_of_each_anchor = gt_boxes[:, :].repeat(self.topk, 1)  # (KxM, 7)
        xyz_local = candidate_anchors[:, 0:3] - gt_boxes_of_each_anchor[:, 0:3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local[:, None, :], -gt_boxes_of_each_anchor[:, 6]
        ).squeeze(dim=1)
        xy_local = xyz_local[:, 0:2]
        lw = gt_boxes_of_each_anchor[:, 3:5][:, [1, 0]]  # bugfixed: w ==> y, l ==> x in local coords
        is_in_gt = ((xy_local <= lw / 2) & (xy_local >= -lw / 2)).all(dim=-1).view(-1, num_gt)  # (K, M)
        is_pos = is_pos & is_in_gt  # (K, M)

        for ng in range(num_gt):
            topk_idxs[:, ng] += ng * num_anchor

        # select the highest IoU if an anchor box is assigned with multiple gt_boxes
        INF = -0x7FFFFFFF
        ious_inf = torch.full_like(ious, INF).t().contiguous().view(-1)  # (MxN)
        index = topk_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gt, -1).t()  # (N, M)

        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

        # match the gt_boxes to the anchors which have maximum iou with them
        max_iou_of_each_gt, argmax_iou_of_each_gt = ious.max(dim=0)
        anchors_to_gt_indexs[argmax_iou_of_each_gt] = torch.arange(0, num_gt, device=ious.device)
        anchors_to_gt_values[argmax_iou_of_each_gt] = max_iou_of_each_gt

        cls_labels = gt_classes[anchors_to_gt_indexs]
        cls_labels[anchors_to_gt_values == INF] = 0
        matched_gts = gt_boxes[anchors_to_gt_indexs]

        pos_mask = cls_labels > 0
        reg_targets = matched_gts.new_zeros((num_anchor, self.box_coder.code_size))
        reg_weights = matched_gts.new_zeros(num_anchor)
        if pos_mask.sum() > 0:
            reg_targets[pos_mask > 0] = self.box_coder.encode_torch(matched_gts[pos_mask > 0], anchors[pos_mask > 0])
            reg_weights[pos_mask] = 1.0

        return cls_labels, reg_targets, reg_weights
