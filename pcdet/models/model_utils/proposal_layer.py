import torch
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils
from ...config import cfg


def proposal_layer(batch_size, batch_cls_preds, batch_box_preds, code_size=7, batch_idx=None, mode='TRAIN'):
    """
    :param batch_cls_preds: (B, N, num_class)
    :param batch_box_preds: (B, N, 7 or C)
    :param mode:
    :return:
    """
    rois = torch.zeros((batch_size, cfg.MODEL[mode].NMS_POST_MAXSIZE, code_size),
                       device=batch_box_preds.device, dtype=torch.float)
    roi_scores = torch.zeros((batch_size, cfg.MODEL[mode].NMS_POST_MAXSIZE),
                             device=batch_box_preds.device, dtype=torch.float)
    roi_rawscores = torch.zeros((batch_size, cfg.MODEL[mode].NMS_POST_MAXSIZE),
                                device=batch_box_preds.device, dtype=torch.float)
    roi_rawscores.fill_(-100000)  # default scores
    roi_labels = torch.ones((batch_size, cfg.MODEL[mode].NMS_POST_MAXSIZE),
                            device=batch_box_preds.device, dtype=torch.long)

    for bs_cnt in range(batch_size):
        if batch_idx is None:
            box_preds = batch_box_preds[bs_cnt]
            cls_preds = batch_cls_preds[bs_cnt]
        else:
            bs_mask = (batch_idx == bs_cnt)
            box_preds = batch_box_preds[bs_mask]
            cls_preds = batch_cls_preds[bs_mask]

        raw_top_scores, top_labels = torch.max(cls_preds, dim=-1)
        top_labels += 1  # shift to [1, num_class]
        top_scores = torch.sigmoid(raw_top_scores)

        if top_scores.shape[0] != 0:
            top_scores, indices = torch.topk(top_scores, k=min(cfg.MODEL[mode].NMS_PRE_MAXSIZE, top_scores.shape[0]))
            box_preds = box_preds[indices]
            raw_top_scores = raw_top_scores[indices]
            top_labels = top_labels[indices]

            boxes_for_nms = box_utils.boxes3d_to_bevboxes_lidar_torch(box_preds)

            keep_idx = getattr(iou3d_nms_utils, cfg.MODEL[mode].RPN_NMS_TYPE)(
                boxes_for_nms, top_scores, cfg.MODEL[mode].RPN_NMS_THRESH
            )

            selected = keep_idx[:cfg.MODEL[mode].NMS_POST_MAXSIZE]
        else:
            selected = []

        selected_boxes = box_preds[selected]
        selected_scores = top_scores[selected]
        selected_rawscores = raw_top_scores[selected]
        selected_labels = top_labels[selected]
        rois[bs_cnt, :selected.shape[0], :] = selected_boxes
        roi_scores[bs_cnt, :selected.shape[0]] = selected_scores
        roi_rawscores[bs_cnt, :selected.shape[0]] = selected_rawscores
        roi_labels[bs_cnt, :selected.shape[0]] = selected_labels

    ret_dict = {
        'rois': rois,
        'roi_raw_scores': roi_rawscores,
        'roi_labels': roi_labels
    }
    return ret_dict


