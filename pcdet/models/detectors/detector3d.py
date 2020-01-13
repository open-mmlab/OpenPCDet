import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vfe import vfe_modules
from ..rpn import rpn_modules
from ..rcnn import rcnn_modules
from ..bbox_heads import bbox_head_modules
from ...utils import loss_utils, common_utils, box_utils
from ...utils.iou3d_nms import iou3d_nms_utils

from ...config import cfg


class Detector3D(nn.Module):
    def __init__(self, num_class, dataset):
        super().__init__()
        # self.output_shape = output_shape
        # self.sparse_shape = output_shape + [1, 0, 0]
        self.num_class = num_class
        self.dataset = dataset
        target_assigner = getattr(dataset, 'target_assigner', None)
        self.box_coder = getattr(target_assigner, 'box_coder', None)
        self.num_anchors_per_location = getattr(target_assigner, 'num_anchors_per_location', None)
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

    def predict_boxes(self, rpn_ret_dict, rcnn_ret_dict, input_dict):
        batch_size = input_dict['batch_size']

        if rcnn_ret_dict is None:
            batch_anchors = input_dict['anchors'].view(batch_size, -1, input_dict['anchors'].shape[-1])
            num_anchors = batch_anchors.shape[1]
            batch_cls_preds = rpn_ret_dict['rpn_cls_preds'].view(batch_size, num_anchors, -1).float()

            batch_box_preds = self.box_coder.decode_with_head_direction_torch(
                box_preds=rpn_ret_dict['rpn_box_preds'].view(batch_size, num_anchors, -1),
                anchors=batch_anchors,
                dir_cls_preds=rpn_ret_dict.get('rpn_dir_cls_preds', None),
                num_dir_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('num_direction_bins', None),
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_offset', None),
                dir_limit_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_limit_offset', None)
            )

        else:
            batch_rois = rcnn_ret_dict['rois']  # (B, N, 7)
            code_size = self.rcnn_box_coder.code_size

            batch_cls_preds = rcnn_ret_dict['rcnn_cls'].view(batch_size, -1)  # (B * N, 1)
            if cfg.MODEL.LOSSES.RCNN_CLS_LOSS == 'smooth-l1':
                roi_ry = batch_rois[:, :, 6].view(-1)
                roi_xyz = batch_rois[:, :, 0:3].view(-1, 3)
                local_rois = batch_rois.clone().detach()
                local_rois[:, :, 0:3] = 0
                rcnn_boxes3d = self.rcnn_box_coder.decode_torch(
                    rcnn_ret_dict['rcnn_reg'].view(local_rois.shape[0], -1, code_size), local_rois
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(
                    rcnn_boxes3d.unsqueeze(dim=1), (roi_ry + np.pi / 2)
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz
                batch_box_preds = rcnn_boxes3d.view(batch_size, -1, code_size)
            else:
                raise NotImplementedError

        pred_dicts, recall_dicts = self.post_processing(batch_cls_preds, batch_box_preds, rcnn_ret_dict, input_dict)
        return pred_dicts, recall_dicts

    def post_processing(self, batch_cls_preds, batch_box_preds, rcnn_ret_dict, input_dict):
        recall_dict = {'gt': 0}
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            recall_dict['roi_%s' % (str(cur_thresh))] = 0
            recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        pred_dicts = []

        batch_size = batch_cls_preds.shape[0]
        batch_index = np.arange(batch_size)
        batch_gt_boxes = input_dict.get('gt_boxes', None)

        for index, cls_preds, box_preds in zip(batch_index, batch_cls_preds, batch_box_preds):
            if cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros']:
                cls_preds = cls_preds[..., 1:]
            normalized_scores = torch.sigmoid(cls_preds)

            if rcnn_ret_dict is not None and batch_gt_boxes is not None:
                self.generate_recall_record(
                    box_preds,
                    rcnn_ret_dict['rois'][index],
                    batch_gt_boxes[index],
                    recall_dict,
                    thresh_list=cfg.MODEL.TEST.RECALL_THRESH_LIST
                )

            if cfg.MODEL.TEST.MULTI_CLASSES_NMS:
                selected, final_labels = self.multi_classes_nms(
                    rank_scores=cls_preds,
                    normalized_scores=normalized_scores,
                    box_preds=box_preds,
                    score_thresh=cfg.MODEL.TEST.SCORE_THRESH,
                    nms_thresh=cfg.MODEL.TEST.NMS_THRESH,
                    nms_type=cfg.MODEL.TEST.NMS_TYPE
                )
                final_boxes = box_preds[selected]
                final_scores = cls_preds[selected] if cfg.MODEL.TEST.USE_RAW_SCORE else normalized_scores[selected]
            else:
                if len(cls_preds.shape) > 1 and cls_preds.shape[1] > 1:
                    rank_scores, class_labels = torch.max(cls_preds, dim=-1)
                    normalized_scores = torch.sigmoid(rank_scores)
                    class_labels = class_labels + 1  # shift to [1, num_classes]
                else:
                    class_labels = rcnn_ret_dict['roi_labels'][index]

                selected = self.class_agnostic_nms(
                    rank_scores=cls_preds,
                    normalized_scores=normalized_scores,
                    box_preds=box_preds,
                    score_thresh=cfg.MODEL.TEST.SCORE_THRESH,
                    nms_thresh=cfg.MODEL.TEST.NMS_THRESH,
                    nms_type=cfg.MODEL.TEST.NMS_TYPE
                )

                final_labels = class_labels[selected]
                final_scores = cls_preds[selected] if cfg.MODEL.TEST.USE_RAW_SCORE else normalized_scores[selected]
                final_boxes = box_preds[selected]

            record_dict = {
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            }

            if rcnn_ret_dict is not None:
                record_dict['roi_raw_scores'] = rcnn_ret_dict['roi_raw_scores'][index][selected]
                record_dict['rois'] = rcnn_ret_dict['rois'][index][selected]

                # filter invalid RoIs
                mask = (record_dict['rois'][:, 3:6].sum(dim=1) > 0)
                if mask.sum() != record_dict['rois'].shape[0]:
                    common_utils.dict_select(record_dict, mask)

            cur_pred_dict = self.dataset.generate_prediction_dict(input_dict, index, record_dict)

            pred_dicts.append(cur_pred_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def multi_classes_nms(rank_scores, normalized_scores, box_preds, score_thresh, nms_thresh, nms_type='nms_gpu'):
        """
        :param rank_scores: (N, num_classes)
        :param box_preds: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :param score_thresh: (N) or float
        :param nms_thresh: (N) or float
        :param nms_type:
        :return:
        """
        assert rank_scores.shape[1] == len(cfg.CLASS_NAMES), 'Rank_score shape: %s' % (str(rank_scores.shape))
        selected_list = []
        selected_labels = []
        num_classes = rank_scores.shape[1]
        boxes_for_nms = box_utils.boxes3d_to_bevboxes_lidar_torch(box_preds)

        score_thresh = score_thresh if isinstance(score_thresh, list) else [score_thresh for x in range(num_classes)]
        nms_thresh = nms_thresh if isinstance(nms_thresh, list) else [nms_thresh for x in range(num_classes)]
        for k in range(0, num_classes):
            class_scores_keep = normalized_scores[:, k] >= score_thresh[k]

            if class_scores_keep.int().sum() > 0:
                original_idxs = class_scores_keep.nonzero().view(-1)
                cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                cur_rank_scores = rank_scores[class_scores_keep, k]

                cur_selected = getattr(iou3d_nms_utils, nms_type)(
                    cur_boxes_for_nms, cur_rank_scores, nms_thresh[k]
                )

                if cur_selected.shape[0] == 0:
                    continue
                selected_list.append(original_idxs[cur_selected])
                selected_labels.append(
                    torch.full([cur_selected.shape[0]], k + 1, dtype=torch.int64, device=box_preds.device)
                )

        selected = torch.cat(selected_list, dim=0) if selected_list.__len__() > 0 else []
        return selected, selected_labels

    @staticmethod
    def class_agnostic_nms(rank_scores, normalized_scores, box_preds, score_thresh, nms_thresh, nms_type='nms_gpu'):
        scores_mask = (normalized_scores >= score_thresh)
        rank_scores_masked = rank_scores[scores_mask]
        cur_selected = []
        if rank_scores_masked.shape[0] > 0:
            box_preds = box_preds[scores_mask]

            rank_scores_nms, indices = torch.topk(
                rank_scores_masked, k=min(cfg.MODEL.TEST.NMS_PRE_MAXSIZE_LAST, rank_scores_masked.shape[0])
            )
            box_preds_nms = box_preds[indices]
            boxes_for_nms = box_utils.boxes3d_to_bevboxes_lidar_torch(box_preds_nms)  #

            keep_idx = getattr(iou3d_nms_utils, nms_type)(
                boxes_for_nms, rank_scores_nms, nms_thresh
            )
            cur_selected = indices[keep_idx[:cfg.MODEL.TEST.NMS_POST_MAXSIZE_LAST]]

        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[cur_selected]
        return selected

    def generate_recall_record(self, box_preds, rois, gt_boxes, recall_dict, thresh_list=(0.5, 0.7)):
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.sum() > 0:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois, cur_gt)
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds, cur_gt)

            for cur_thresh in thresh_list:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            recall_dict['gt'] += cur_gt.shape[0]

            iou3d_rcnn = iou3d_rcnn.max(dim=1)[0]
            gt_iou = iou3d_rcnn
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
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

    def get_rcnn_loss(self, rcnn_ret_dict):
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

                    loss_corner = loss_utils.get_corner_loss_lidar(
                        rcnn_boxes3d[:, 0:7],
                        gt_of_rois_src[fg_mask][:, 0:7]
                    )
                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * LOSS_WEIGHTS['rcnn_corner_weight']

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError

        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss.item()
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

    def load_params_from_file(self, filename, to_cpu=False, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info("==> Loading parameters from checkpoint %s to %s" % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info("==> Done (loaded %d/%d)" % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info("==> Loading parameters from checkpoint %s to %s" % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            logger.info("==> Loading optimizer parameters from checkpoint %s to %s"
                        % (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")

        return it, epoch
