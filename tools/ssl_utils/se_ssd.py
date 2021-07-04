import torch
import torch.nn.functional as F
import numpy as np
from .semi_utils import reverse_transform, load_data_to_gpu, filter_boxes
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def get_iou_consistency_loss(teacher_boxes, student_boxes):
    box_losses, cls_losses = [], []
    batch_normalizer = 0
    for teacher_box, student_box in zip(teacher_boxes, student_boxes):
        teacher_cls_preds = teacher_box['pred_cls_preds'].detach_()
        teacher_box_preds = teacher_box['pred_boxes'].detach_()
        student_cls_preds = student_box['pred_cls_preds']
        student_box_preds = student_box['pred_boxes']
        num_teacher_boxes = teacher_box_preds.shape[0]
        num_student_boxes = student_box_preds.shape[0]
        if num_teacher_boxes == 0 or num_student_boxes == 0:
            batch_normalizer += 1
            continue

        with torch.no_grad():
            teacher_class = torch.max(teacher_cls_preds, dim=-1, keepdim=True)[1] # [Nt, 1]
            student_class = torch.max(student_cls_preds, dim=-1, keepdim=True)[1] # [Ns, 1]
            not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]

            iou_3d = boxes_iou3d_gpu(teacher_box_preds, student_box_preds) # [Nt, Ns]
            iou_3d -= not_same_class # iou < 0 if not from the same class
            matched_iou_of_stduent, matched_teacher_index_of_student = iou_3d.max(0) # [Ns]
            MATCHED_IOU_TH = 0.7
            matched_teacher_mask = (matched_iou_of_stduent >= MATCHED_IOU_TH).float().unsqueeze(-1)
            num_matched_boxes = matched_teacher_mask.sum()
            if num_matched_boxes == 0: num_matched_boxes = 1

        matched_teacher_preds = teacher_box_preds[matched_teacher_index_of_student]
        matched_teacher_cls = teacher_cls_preds[matched_teacher_index_of_student]

        student_box_reg, student_box_rot = student_box_preds[:, :6], student_box_preds[:, [6]]
        matched_teacher_reg, matched_teacher_rot = matched_teacher_preds[:, :6], matched_teacher_preds[:, [6]]

        box_loss_reg = F.smooth_l1_loss(student_box_reg, matched_teacher_reg, reduction='none')
        box_loss_reg = (box_loss_reg * matched_teacher_mask).sum() / num_matched_boxes
        box_loss_rot = F.smooth_l1_loss(torch.sin(student_box_rot - matched_teacher_rot), torch.zeros_like(student_box_rot), reduction='none')
        box_loss_rot = (box_loss_rot * matched_teacher_mask).sum() / num_matched_boxes
        consistency_box_loss = box_loss_reg + box_loss_rot
        consistency_cls_loss = F.smooth_l1_loss(student_cls_preds, matched_teacher_cls, reduction='none')
        consistency_cls_loss = (consistency_cls_loss * matched_teacher_mask).sum() / num_matched_boxes

        box_losses.append(consistency_box_loss)
        cls_losses.append(consistency_cls_loss)
        batch_normalizer += 1

    return sum(box_losses)/batch_normalizer, sum(cls_losses)/batch_normalizer

def sigmoid_rampup(current, rampup_start, rampup_end):
    assert rampup_start <= rampup_end
    if current < rampup_start:
        return 0
    elif (current >= rampup_start) and (current < rampup_end):
        rampup_length = max(rampup_end, 0) - max(rampup_start, 0)
        if rampup_length == 0: # no rampup
            return 1
        else:
            phase = 1.0 - (current - max(rampup_start, 0)) / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    elif current >= rampup_end:
        return 1
    else:
        raise Exception('Impossible condition for sigmoid rampup')

def se_ssd(teacher_model, student_model,
         ld_teacher_batch_dict, ld_student_batch_dict,
         ud_teacher_batch_dict, ud_student_batch_dict,
         cfgs, epoch_id, dist
        ):
    load_data_to_gpu(ld_teacher_batch_dict)
    load_data_to_gpu(ld_student_batch_dict)
    load_data_to_gpu(ud_teacher_batch_dict)
    load_data_to_gpu(ud_student_batch_dict)

    # get loss for labeled data
    if not dist:
        ld_teacher_batch_dict = teacher_model(ld_teacher_batch_dict)
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict)
        ld_student_batch_dict, ret_dict, tb_dict, disp_dict = student_model(ld_student_batch_dict)
        ud_student_batch_dict = student_model(ud_student_batch_dict)
    else:
        ld_teacher_batch_dict, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        (ld_student_batch_dict, ret_dict, tb_dict, disp_dict), (ud_student_batch_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    sup_loss = ret_dict['loss'].mean()

    ld_teacher_boxes = filter_boxes(ld_teacher_batch_dict, cfgs)
    ud_teacher_boxes = filter_boxes(ud_teacher_batch_dict, cfgs)
    ld_student_boxes = filter_boxes(ld_student_batch_dict, cfgs)
    ud_student_boxes = filter_boxes(ud_student_batch_dict, cfgs)

    ld_teacher_boxes = reverse_transform(ld_teacher_boxes, ld_teacher_batch_dict, ld_student_batch_dict)
    ud_teacher_boxes = reverse_transform(ud_teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)

    ld_box_loss, ld_cls_loss = get_iou_consistency_loss(ld_teacher_boxes, ld_student_boxes)
    ud_box_loss, ud_cls_loss = get_iou_consistency_loss(ud_teacher_boxes, ud_student_boxes)

    consistency_loss = (ld_box_loss + ud_box_loss) * cfgs.CONSIST_BOX_WEIGHT \
                       + (ld_cls_loss + ud_cls_loss) * cfgs.CONSIST_CLS_WEIGHT
    consistency_weight = cfgs.CONSISTENCY_WEIGHT * sigmoid_rampup(epoch_id, cfgs.TEACHER.EMA_EPOCH[0], cfgs.TEACHER.EMA_EPOCH[1])

    loss = sup_loss + consistency_weight * consistency_loss
    return loss, tb_dict, disp_dict