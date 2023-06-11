import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes


def class_specific_nms(box_scores, box_preds, box_labels, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N,)
        box_preds: (N, 7 + C)
        box_labels: (N,)
        nms_config:

    Returns:

    """
    selected = []
    for k in range(len(nms_config.NMS_THRESH)):
        curr_mask = box_labels == k
        if score_thresh is not None and isinstance(score_thresh, float):
            curr_mask *= (box_scores > score_thresh)
        elif score_thresh is not None and isinstance(score_thresh, list):
            curr_mask *= (box_scores > score_thresh[k])
        curr_idx = torch.nonzero(curr_mask)[:, 0]
        curr_box_scores = box_scores[curr_mask]
        cur_box_preds = box_preds[curr_mask]

        if curr_box_scores.shape[0] > 0:
            curr_box_scores_nms = curr_box_scores
            curr_boxes_for_nms = cur_box_preds

            keep_idx, _ = getattr(iou3d_nms_utils, 'nms_gpu')(
                curr_boxes_for_nms, curr_box_scores_nms,
                thresh=nms_config.NMS_THRESH[k],
                pre_maxsize=nms_config.NMS_PRE_MAXSIZE[k],
                post_max_size=nms_config.NMS_POST_MAXSIZE[k]
            )
            curr_selected = curr_idx[keep_idx]
            selected.append(curr_selected)
    if len(selected) != 0:
        selected = torch.cat(selected)
        

    return selected, box_scores[selected]
