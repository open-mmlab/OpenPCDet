import torch
from .semi_utils import reverse_transform, load_data_to_gpu, construct_pseudo_label
from pcdet.models.model_utils.model_nms_utils import class_agnostic_nms

@torch.no_grad()
def iou_match_3d_filter(batch_dict, cfgs):
    batch_size = batch_dict['batch_size']
    pred_dicts = []
    for index in range(batch_size):
        box_preds = batch_dict['rois'][index]
        iou_preds = batch_dict['roi_ious'][index]
        cls_preds = batch_dict['roi_scores'][index]
        label_preds = batch_dict['roi_labels'][index]

        if not batch_dict['cls_preds_normalized']:
            iou_preds = torch.sigmoid(iou_preds)
            cls_preds = torch.sigmoid(cls_preds)

        iou_preds = iou_preds.squeeze(-1)
        # filtered by iou
        iou_threshold_per_class = cfgs.IOU_SCORE_THRESH
        num_classes = len(iou_threshold_per_class)
        iou_th = iou_preds.new_zeros(iou_preds.shape)
        for cls_idx in range(num_classes):
            class_mask = label_preds == (cls_idx + 1)
            iou_th[class_mask] = iou_threshold_per_class[cls_idx]
        iou_mask = iou_preds >= iou_th
        iou_preds = iou_preds[iou_mask]
        box_preds = box_preds[iou_mask]
        cls_preds = cls_preds[iou_mask]
        label_preds = label_preds[iou_mask]

        nms_scores = cls_preds # iou_preds
        selected, selected_scores = class_agnostic_nms(
            box_scores=nms_scores, box_preds=box_preds,
            nms_config=cfgs.NMS_CONFIG,
            score_thresh=cfgs.CLS_SCORE_THRESH
        )

        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = box_preds[selected]

        # added filtering boxes with size 0
        zero_mask = (final_boxes[:, 3:6] != 0).all(1)
        final_boxes = final_boxes[zero_mask]
        final_labels = final_labels[zero_mask]
        final_scores = final_scores[zero_mask]

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels,
        }
        pred_dicts.append(record_dict)

    return pred_dicts

def iou_match_3d(teacher_model, student_model,
                  ld_teacher_batch_dict, ld_student_batch_dict,
                  ud_teacher_batch_dict, ud_student_batch_dict,
                  cfgs, epoch_id, dist
                 ):
    assert ld_teacher_batch_dict is None # Only generate labels for unlabeled data

    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)

    if not dist:
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
    else:
        _, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)

    teacher_boxes = iou_match_3d_filter(ud_teacher_batch_dict, cfgs.TEACHER)
    teacher_boxes = reverse_transform(teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
    gt_boxes = construct_pseudo_label(teacher_boxes)
    ud_student_batch_dict['gt_boxes'] = gt_boxes

    if not dist:
        _, ld_ret_dict, _, _ = student_model(ld_student_batch_dict)
        _, ud_ret_dict, tb_dict, disp_dict = student_model(ud_student_batch_dict)
    else:
        (_, ld_ret_dict, _, _), (_, ud_ret_dict, tb_dict, disp_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    loss = ld_ret_dict['loss'].mean() + ud_ret_dict['loss'].mean()

    return loss, tb_dict, disp_dict