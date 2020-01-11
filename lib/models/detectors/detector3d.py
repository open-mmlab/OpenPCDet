import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vfe import vfe_modules
from ..rpn import rpn_modules
from ..rcnn import rcnn_modules
from ..bbox_heads import bbox_head_modules
from ...utils import loss_utils, common_utils

from ...config import cfg


class Detector3D(nn.Module):
    def __init__(self, num_class, target_assigner):
        super().__init__()
        # self.output_shape = output_shape
        # self.sparse_shape = output_shape + [1, 0, 0]
        self.num_class = num_class
        self.box_coder = target_assigner.box_coder
        self.num_anchors_per_location = target_assigner.num_anchors_per_location

        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.mode = 'TRAIN' if self.training else 'TEST'

        self.vfe = self.rpn_net = self.rpn_head = self.rcnn_net = self.rcnn_box_coder = None
        self.rpn_cls_loss_func = self.rpn_reg_loss_func = self.rpn_dir_loss_func = self.rcnn_reg_loss_func = None

    def build_networks(self, model_cfg):
        vfe_cfg = model_cfg.VFE
        self.vfe = vfe_modules[vfe_cfg.NAME](**vfe_cfg)
        voxel_feature_num = self.vfe.get_output_feature_dim()

        rpn_cfg = model_cfg.RPN
        self.rpn_net = rpn_modules[rpn_cfg.BACKBONE.NAME](
            input_channels=voxel_feature_num,
            **rpn_cfg.BACKBONE.ARGS
        )

        rpn_head_cfg = model_cfg.RPN.RPN_HEAD
        self.rpn_head = bbox_head_modules[rpn_head_cfg.NAME](
            num_class=self.num_class,
            num_anchor_per_loc=self.num_anchors_per_location,
            box_code_size=self.box_coder.code_size,
            **rpn_head_cfg
        )

        rcnn_cfg = model_cfg.RCNN
        if rcnn_cfg.ENABLED:
            self.rcnn_box_coder = self.box_coder
            self.rcnn_net = rcnn_modules[rcnn_cfg.NAME](
                code_size=self.rcnn_box_coder.code_size,
                num_point_features=cfg.MODEL.RPN.BACKBONE.NUM_POINT_FEATURES,
                rcnn_cfg=rcnn_cfg
            )

    def build_losses(self, losses_cfg):
        # loss function definition
        self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']

        rpn_code_weights = code_weights[3:7] if losses_cfg.RPN_REG_LOSS == 'bin-based' else code_weights
        self.rpn_reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=rpn_code_weights)
        self.rpn_dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()

        rcnn_cfg = cfg.MODEL.RCNN
        if rcnn_cfg.ENABLED:
            self.rcnn_reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=code_weights)

    def update_global_step(self):
        self.global_step += 1

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            pass

    def forward(self, input_dict):
        raise NotImplementedError

    def predict(self, dataset, input_dict, batch_cls_preds, batch_box_preds, batch_gt_boxes=None,
                batch_rois=None, batch_roi_labels=None, batch_roi_raw_scores=None, batch_rcnn_bbox=None, mode='RPN'):
        recall_dict = {'roi_05': 0, 'rcnn_05': 0, 'gt': 0, 'roi_07': 0, 'rcnn_07': 0,
                       'rcnn_rank_acc': 0, 'valid_sample_num': 0, 'rcnn_iou_diff': 0}
        predictions_dicts = []

        batch_size = batch_cls_preds.shape[0]
        batch_index = np.arange(batch_size)

        for index, cls_preds, box_preds in zip(batch_index, batch_cls_preds, batch_box_preds):
            if mode == 'RPN':
                if cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS:
                    # this don't support softmax
                    assert cfg.RPN_STAGE.USE_SIGMOID_SCORE is True
                    rank_scores = torch.sigmoid(cls_preds)
                    rank_raw_scores = cls_preds
                else:
                    # encode background as first element in one-hot vector
                    if cfg.RPN_STAGE.USE_SIGMOID_SCORE:
                        rank_scores = torch.sigmoid(cls_preds)[..., 1:]
                    else:
                        rank_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
                    rank_raw_scores = cls_preds[..., 1:]

                if cfg.TEST.USE_MULTI_CLASSES_NMS:
                    top_scores = rank_scores
                    top_raw_scores = rank_raw_scores
                    top_labels = None
                else:
                    top_scores, top_labels = torch.max(rank_scores, dim=-1)
                    top_labels = top_labels + 1  # shift to [1, num_classes]
                    top_raw_scores, _ = torch.max(rank_raw_scores, dim=-1)
            else:
                if batch_gt_boxes is not None:
                    gt_boxes = batch_gt_boxes[index]
                    gt_iou = self.calculate_recall(cls_preds, box_preds, batch_rois[index], gt_boxes, recall_dict)

                    pred_mat = kitti_utils.score_to_compare_matrix(cls_preds)
                    gt_mat = kitti_utils.score_to_compare_matrix(gt_iou)
                    mask = (torch.abs(gt_mat) > 1e-3)
                    fg = ((pred_mat > 0) == (gt_mat > 0))
                    rank_acc = (fg * mask).float().sum() / torch.clamp_min(mask.float().sum(), min=1.0)
                    recall_dict['rcnn_rank_acc'] += rank_acc.item()
                    recall_dict['valid_sample_num'] += (gt_iou.max() > 0).item()
                    iou_diff = torch.abs((torch.sigmoid(cls_preds) - gt_iou))
                    recall_dict['rcnn_iou_diff'] += iou_diff.mean().item()
                else:
                    gt_iou = cls_preds
                rank_raw_scores, rank_scores = self.get_rank_scores(cls_preds, batch_roi_raw_scores[index], gt_iou)

                top_scores = rank_scores.squeeze(-1)
                top_labels = batch_roi_labels[index]
                top_raw_scores = rank_raw_scores

            thresh = torch.tensor([cfg.TEST.SCORE_THRESH], device=rank_scores.device).type_as(rank_scores)
            if cfg.TEST.USE_MULTI_CLASSES_NMS and mode == 'RPN':
                boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds)

                selected_list = []
                selected_top_scores = []
                selected_top_raw_scores = []
                selected_labels = []
                num_classes = len(cfg.CLASSES)
                for k in range(0, num_classes):
                    class_scores = rank_scores[:, k]
                    class_scores_keep = class_scores >= thresh

                    if class_scores_keep.int().sum() > 0:
                        original_idxs = class_scores_keep.nonzero().view(-1)
                        cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                        cur_top_raw_scores = top_raw_scores[class_scores_keep, k]
                        cur_selected = iou3d_utils.nms_gpu(cur_boxes_for_nms, cur_top_raw_scores, cfg.TEST.NMS_THRESH)

                        if cur_selected.shape[0] > 0:
                            selected_list.append(original_idxs[cur_selected])
                            selected_top_scores.append(class_scores[class_scores_keep][cur_selected])
                            selected_top_raw_scores.append(cur_top_raw_scores[cur_selected])
                            selected_labels.append(
                                torch.full([cur_selected.shape[0]], k + 1,
                                           dtype=torch.int64, device=box_preds.device)
                            )

                if selected_list.__len__() >= 1:
                    selected = torch.cat(selected_list, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_top_scores = torch.cat(selected_top_scores, dim=0)
                    selected_top_raw_scores = torch.cat(selected_top_raw_scores, dim=0)
                else:
                    selected = []
                    selected_top_scores = top_scores[selected]
                    selected_top_raw_scores = top_raw_scores[selected]
                    selected_labels = selected_top_scores
                selected_boxes = box_preds[selected]
            else:
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] > 0:
                    top_raw_scores = top_raw_scores.masked_select(top_scores_keep)
                    box_preds = box_preds[top_scores_keep]

                    # NMS in birdeye view
                    # TODO: use my rotated nms, not checked
                    # boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds)
                    # selected = iou3d_utils.nms_gpu(boxes_for_nms, top_raw_scores, cfg.TEST.NMS_THRESH)
                    # selected = selected[:cfg.MAX_OBJECTS_EACH_SCENE]  # max boxes 500

                    top_raw_scores_nms, indices = torch.topk(top_raw_scores,
                                                             k=min(cfg.TEST.FINAL_NMS_PRE_MAXSIZE,
                                                                   top_raw_scores.shape[0]))
                    box_preds_nms = box_preds[indices]
                    boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds_nms)#

                    if cfg.TEST.NMS_TYPE == 'rotated':
                        keep_idx = iou3d_utils.nms_gpu(boxes_for_nms, top_raw_scores_nms, cfg.TEST.NMS_THRESH)
                        selected = indices[keep_idx[:cfg.TEST.FINAL_NMS_POST_MAXSIZE]]
                    elif cfg.TEST.NMS_TYPE == 'soft_rotated_nms':
                        import utils.nms.nms_utils as nms_utils
                        top_scores_nms = torch.sigmoid(top_raw_scores_nms)
                        keep_idx, soft_scores = nms_utils.soft_nms_cpu(boxes_for_nms, top_scores_nms,
                                                                       score_thresh=cfg.TEST.SOFTNMS_SCORE_THRESH,
                                                                       sigma=cfg.TEST.NMS_SIGMA,
                                                                       Nt=0.3,
                                                                       soft_type=cfg.TEST.NMS_SOFT_TYPE)
                        selected = indices[keep_idx[:cfg.TEST.FINAL_NMS_POST_MAXSIZE]]
                        top_raw_scores[selected] = top_scores[selected] = soft_scores
                    else:
                        raise NotImplementedError
                    selected = selected[:cfg.MAX_OBJECTS_EACH_SCENE]  # max boxes 500
                else:
                    selected = []

                selected_labels = top_labels[top_scores_keep][selected]
                selected_top_scores = top_scores[selected]
                selected_top_raw_scores = top_raw_scores[selected]
                selected_boxes = box_preds[selected]

            record_dict = {
                'boxes': selected_boxes,
                'scores': selected_top_scores,
                'raw_scores': selected_top_raw_scores,
                'labels': selected_labels
            }

            if mode == 'RCNN':
                record_dict['roi_raw_scores'] = batch_roi_raw_scores[index][top_scores_keep][selected]
                record_dict['rois'] = batch_rois[index][top_scores_keep][selected]
                record_dict['gt_iou'] = gt_iou[top_scores_keep][selected]

                if batch_rcnn_bbox is not None:
                    record_dict['predict_bbox'] = batch_rcnn_bbox[index][top_scores_keep][selected]

                # filter invalid RoIs
                mask = (record_dict['rois'][:, 3:6].sum(dim=1) > 0)
                if mask.sum() != record_dict['rois'].shape[0]:
                    kitti_utils.dict_select(record_dict, mask)

            predictions_dict = dataset.generate_prediction_dict(input_dict, index, record_dict)

            predictions_dicts.append(predictions_dict)
        return predictions_dicts, recall_dict

    def get_rank_scores(self, cls_preds, roi_rawscores, gt_iou):
        if cfg.TEST.SCORE_TYPE == 'rcnn':
            rank_scores = torch.sigmoid(cls_preds)
            rank_raw_scores = cls_preds.squeeze(-1)
        elif cfg.TEST.SCORE_TYPE == 'roi':
            rank_scores = torch.sigmoid(roi_rawscores)
            rank_raw_scores = roi_rawscores
        elif cfg.TEST.SCORE_TYPE == 'mul':
            rank_scores = rank_raw_scores = torch.sigmoid(roi_rawscores) * torch.sigmoid(cls_preds)
        elif cfg.TEST.SCORE_TYPE == 'gt':
            rank_scores = rank_raw_scores = raw_scores = gt_iou
        elif cfg.TEST.SCORE_TYPE == 'avg':
            rank_scores = rank_raw_scores = (torch.sigmiod(roi_rawscores) + torch.sigmoid(cls_preds)) / 2
        elif cfg.TEST.SCORE_TYPE == 'combine':
            rank_raw_scores = cls_preds.clone().detach()
            mask = torch.sigmoid(roi_rawscores) < 0.5
            rank_raw_scores[mask] = rank_raw_scores[mask] / 1.5
            rank_scores = torch.sigmoid(rank_raw_scores)
        elif cfg.TEST.SCORE_TYPE == 'mul_raw':
            roi_scores = torch.sigmoid(roi_rawscores)
            rank_raw_scores = roi_scores * cls_preds
            rank_scores = torch.sigmoid(rank_raw_scores)
        else:
            raise NotImplementedError

        return rank_raw_scores, rank_scores

    def calculate_recall(self, cls_preds, box_preds, rois, gt_boxes, recall_dict):
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.sum() == 0:
            gt_iou = cls_preds * 0
        else:
            box_type = 'nuscenes' if cfg.DATASET == 'nuscenes' else 'lidar'
            iou3d_roi = iou3d_utils.boxes_iou3d_gpu(rois, cur_gt, box_type=box_type)
            iou3d_rcnn = iou3d_utils.boxes_iou3d_gpu(box_preds, cur_gt, box_type=box_type)

            roi_recalled = (iou3d_roi.max(dim=0)[0] > 0.7).sum().item()
            rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > 0.7).sum().item()
            recall_dict['roi_07'] += roi_recalled
            recall_dict['rcnn_07'] += rcnn_recalled
            roi_recalled = (iou3d_roi.max(dim=0)[0] > 0.5).sum().item()
            rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > 0.5).sum().item()
            recall_dict['roi_05'] += roi_recalled
            recall_dict['rcnn_05'] += rcnn_recalled
            recall_dict['gt'] += cur_gt.shape[0]

            iou3d_rcnn = iou3d_rcnn.max(dim=1)[0]
            gt_iou = iou3d_rcnn

        return gt_iou

    def get_anchor_box_loss(self, cls_preds, box_preds, box_cls_labels, box_reg_targets,
                            box_dir_cls_preds=None, anchors=None):
        """
        :param cls_preds:
        :param box_preds:
        :param box_cls_labels:
        :param box_reg_targets:
        :param dir_cls_preds:
        :return:
        """
        loss_cfgs = cfg.MODEL.LOSSES

        # rpn head losses
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        batch_size = int(box_preds.shape[0])
        num_class = self.num_class

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
            one_hot_targets = one_hot_targets[..., 1:]
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

        loss_weights_dict = loss_cfgs.LOSS_WEIGHTS
        cls_loss = self.rpn_cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced = cls_loss_reduced * loss_weights_dict['rpn_cls_weight']

        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        if loss_cfgs.RPN_REG_LOSS == 'smooth-l1':
            # sin(a - b) = sinacosb-cosasinb
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss = self.rpn_reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            loc_loss_reduced = loc_loss.sum() / batch_size
        else:
            raise NotImplementedError

        loc_loss_reduced = loc_loss_reduced * loss_weights_dict['rpn_loc_weight']

        rpn_loss = loc_loss_reduced + cls_loss_reduced

        tb_dict = {
            'rpn_loss_loc': loc_loss_reduced.item(),
            'rpn_loss_cls': cls_loss_reduced.item()
        }
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS['dir_offset'],
                num_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS['num_dir_bins']
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, cfg.MODEL.RPN.RPN_HEAD.ARGS['num_dir_bins'])
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.rpn_dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['rpn_dir_weight']
            rpn_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_rcnn_loss(self, rcnn_ret_dict, input_dict):
        code_size = self.rcnn_box_coder.code_size
        rcnn_cls = rcnn_ret_dict['rcnn_cls']
        rcnn_cls_labels = rcnn_ret_dict['rcnn_cls_labels'].float().view(-1)
        reg_valid_mask = rcnn_ret_dict['reg_valid_mask']
        gt_boxes3d_ct = rcnn_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = rcnn_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = rcnn_ret_dict['rcnn_reg']
        roi_boxes3d = rcnn_ret_dict['rois']
        rcnn_batch_size = rcnn_cls_labels.shape[0]

        rcnn_loss = 0
        loss_cfgs = cfg.MODEL.LOSSES
        LOSS_WEIGHTS = loss_cfgs.LOSS_WEIGHTS

        # rcnn classification loss
        if loss_cfgs.RCNN_CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels, reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = rcnn_loss_cls * LOSS_WEIGHTS['rcnn_cls_weight']
        else:
            raise NotImplementedError

        rcnn_loss += rcnn_loss_cls
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum == 0:
            # To be consistent with DistributedDataParallel
            # Faked a rcnn_loss to make gradient of regression branch be zero
            temp_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            faked_reg_target = temp_rcnn_reg.detach()
            rcnn_loss_reg = self.rcnn_reg_loss_func(temp_rcnn_reg, faked_reg_target)  # [N, M]
            rcnn_loss_reg = rcnn_loss_reg.sum() / 1.0
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        else:
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            if loss_cfgs.RCNN_REG_LOSS == 'smooth-l1':
                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                reg_targets = self.box_coder.encode_torch(
                    gt_boxes3d_ct.view(rcnn_batch_size, code_size)[fg_mask], rois_anchor[fg_mask]
                )
                rcnn_loss_reg = self.rcnn_reg_loss_func(
                    rcnn_reg.view(rcnn_batch_size, -1)[fg_mask].unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0)
                )  # [N, M]
                rcnn_loss_reg = rcnn_loss_reg.sum() / max(fg_sum, 0)
                rcnn_loss_reg = rcnn_loss_reg * LOSS_WEIGHTS['rcnn_reg_weight']
                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

                if loss_cfgs.CORNER_LOSS_REGULARIZATION:
                    # TODO: NEED to BE CHECK
                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.rcnn_box_coder.decode_torch(
                        fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                    ).view(-1, code_size)

                    rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(
                        rcnn_boxes3d.unsqueeze(dim=1), (roi_ry + np.pi / 2)
                    ).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz

                    loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * LOSS_WEIGHTS['rcnn_corner_weight']

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError

        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss
        return rcnn_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim+1]) * torch.cos(boxes2[..., dim:dim+1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim+1]) * torch.sin(boxes2[..., dim:dim+1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim+1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim+1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period_torch(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

