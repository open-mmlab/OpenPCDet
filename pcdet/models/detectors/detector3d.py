import os
import numpy as np
import torch
import torch.nn as nn

from ..vfe import vfe_modules
from ..rpn import rpn_modules
from ..rcnn import rcnn_modules
from ..bbox_heads import bbox_head_modules
from ...utils import common_utils, box_utils
from ...ops.iou3d_nms import iou3d_nms_utils

from ...config import cfg


class Detector3D(nn.Module):
    def __init__(self, num_class, dataset):
        super().__init__()
        self.num_class = num_class
        self.dataset = dataset
        self.grid_size = dataset.voxel_generator.grid_size
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.vfe = self.rpn_net = self.rpn_head = self.rcnn_net = None

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build_networks(self, model_cfg):
        vfe_cfg = model_cfg.VFE
        self.vfe = vfe_modules[vfe_cfg.NAME](
            num_input_features=cfg.DATA_CONFIG.NUM_POINT_FEATURES['use'],
            voxel_size=cfg.DATA_CONFIG.VOXEL_GENERATOR.VOXEL_SIZE,
            pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            **vfe_cfg.ARGS
        )
        voxel_feature_num = self.vfe.get_output_feature_dim()

        rpn_cfg = model_cfg.RPN
        self.rpn_net = rpn_modules[rpn_cfg.BACKBONE.NAME](
            input_channels=voxel_feature_num,
            **rpn_cfg.BACKBONE.ARGS
        )

        rpn_head_cfg = model_cfg.RPN.RPN_HEAD
        self.rpn_head = bbox_head_modules[rpn_head_cfg.NAME](
            num_class=self.num_class,
            args=rpn_head_cfg.ARGS,
            grid_size=self.grid_size,
            anchor_target_cfg=rpn_head_cfg.TARGET_CONFIG
        )

        rcnn_cfg = model_cfg.RCNN
        if rcnn_cfg.ENABLED:
            self.rcnn_net = rcnn_modules[rcnn_cfg.NAME](
                num_point_features=cfg.MODEL.RCNN.NUM_POINT_FEATURES,
                rcnn_cfg=rcnn_cfg
            )

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
            batch_anchors = rpn_ret_dict['anchors'].view(1, -1, rpn_ret_dict['anchors'].shape[-1]).repeat(batch_size, 1, 1)
            num_anchors = batch_anchors.shape[1]
            batch_cls_preds = rpn_ret_dict['rpn_cls_preds'].view(batch_size, num_anchors, -1).float()

            batch_box_preds = self.rpn_head.box_coder.decode_with_head_direction_torch(
                box_preds=rpn_ret_dict['rpn_box_preds'].view(batch_size, num_anchors, -1),
                anchors=batch_anchors,
                dir_cls_preds=rpn_ret_dict.get('rpn_dir_cls_preds', None),
                num_dir_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('num_direction_bins', None),
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_offset', None),
                dir_limit_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_limit_offset', None),
                use_binary_dir_classifier=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('use_binary_dir_classifier', False)
            )

        else:
            batch_rois = rcnn_ret_dict['rois']  # (B, N, 7)
            code_size = self.rcnn_net.box_coder.code_size

            batch_cls_preds = rcnn_ret_dict['rcnn_cls'].view(batch_size, -1)  # (B * N, 1)
            if cfg.MODEL.LOSSES.RCNN_REG_LOSS == 'smooth-l1':
                roi_ry = batch_rois[:, :, 6].view(-1)
                roi_xyz = batch_rois[:, :, 0:3].view(-1, 3)
                local_rois = batch_rois.clone().detach()
                local_rois[:, :, 0:3] = 0
                rcnn_boxes3d = self.rcnn_net.box_coder.decode_torch(
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
            if not cfg.MODEL.RPN.RPN_HEAD.ARGS['encode_background_as_zeros'] and rcnn_ret_dict is None:
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
                    if rcnn_ret_dict is not None:
                        class_labels = rcnn_ret_dict['roi_labels'][index]
                    else:
                        class_labels = cls_preds.new_ones(cls_preds.shape[0])
                    rank_scores = cls_preds.view(-1)
                    normalized_scores = normalized_scores.view(-1)

                selected = self.class_agnostic_nms(
                    rank_scores=rank_scores,
                    normalized_scores=normalized_scores,
                    box_preds=box_preds,
                    score_thresh=cfg.MODEL.TEST.SCORE_THRESH,
                    nms_thresh=cfg.MODEL.TEST.NMS_THRESH,
                    nms_type=cfg.MODEL.TEST.NMS_TYPE
                )

                final_labels = class_labels[selected]
                final_scores = rank_scores[selected] if cfg.MODEL.TEST.USE_RAW_SCORE else normalized_scores[selected]
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

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
