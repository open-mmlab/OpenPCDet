import torch
import numpy as np

from pcdet.models.model_utils import model_nms_utils

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()

"""
Reverse augmentation transform
"""

def random_world_flip(box_preds, params, reverse = False):
    if reverse:
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
    else:
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
    return box_preds

def random_world_rotation(box_preds, params, reverse = False):
    if reverse:
        noise_rotation = -params
    else:
        noise_rotation = params

    angle = torch.tensor([noise_rotation]).to(box_preds.device)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(1)
    ones = angle.new_ones(1)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(3, 3).float()
    box_preds[:, :3] = torch.matmul(box_preds[:, :3], rot_matrix)
    box_preds[:, 6] += noise_rotation
    return box_preds

def random_world_scaling(box_preds, params, reverse = False):
    if reverse:
        noise_scale = 1.0/params
    else:
        noise_scale = params

    box_preds[:, :6] *= noise_scale
    return box_preds

@torch.no_grad()
def reverse_transform(teacher_boxes, teacher_dict, student_dict):
    augmentation_functions = {
        'random_world_flip': random_world_flip,
        'random_world_rotation': random_world_rotation,
        'random_world_scaling': random_world_scaling
    }
    for bs_idx, teacher_box in enumerate(teacher_boxes):
        teacher_aug_list = teacher_dict['augmentation_list'][bs_idx]
        student_aug_list = student_dict['augmentation_list'][bs_idx]
        teacher_aug_param = teacher_dict['augmentation_params'][bs_idx]
        student_aug_param = student_dict['augmentation_params'][bs_idx]
        box_preds = teacher_box['pred_boxes']
        # inverse teacher augmentation
        teacher_aug_list = teacher_aug_list[::-1]
        for key in teacher_aug_list:
            aug_params = teacher_aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse = True)
        # student_augmentation
        for key in student_aug_list:
            aug_params = student_aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse = False)
        teacher_box['pred_boxes'] = box_preds
    return teacher_boxes

"""
Filter predicted boxes with conditions
"""

def filter_boxes(batch_dict, cfgs):
    batch_size = batch_dict['batch_size']
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_box_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_box_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        cls_preds = batch_dict['batch_cls_preds'][batch_mask]

        if not batch_dict['cls_preds_normalized']:
            cls_preds = torch.sigmoid(cls_preds)

        max_cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        if batch_dict.get('has_class_labels', False):
            label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
            label_preds = batch_dict[label_key][index]
        else:
            label_preds = label_preds + 1

        final_boxes = box_preds
        final_labels = label_preds
        final_cls_preds = cls_preds

        if cfgs.get('FILTER_BY_NMS', False):
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=max_cls_preds, box_preds=final_boxes,
                nms_config=cfgs.NMS.NMS_CONFIG,
                score_thresh=cfgs.NMS.SCORE_THRESH
            )

            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        if cfgs.get('FILTER_BY_SCORE_THRESHOLD', False):
            selected = max_cls_preds > cfgs.SCORE_THRESHOLD
            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        if cfgs.get('FILTER_BY_TOPK', False):
            topk = min(max_cls_preds.shape[0], cfgs.TOPK)
            selected = torch.topk(max_cls_preds, topk)[1]
            final_labels = final_labels[selected]
            final_boxes = final_boxes[selected]
            final_cls_preds = final_cls_preds[selected]
            max_cls_preds = max_cls_preds[selected]

        # added filtering boxes with size 0
        zero_mask = (final_boxes[:, 3:6] != 0).all(1)
        final_boxes = final_boxes[zero_mask]
        final_labels = final_labels[zero_mask]
        final_cls_preds = final_cls_preds[zero_mask]

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_cls_preds': final_cls_preds,
            'pred_labels': final_labels
        }
        pred_dicts.append(record_dict)

    return pred_dicts

"""
Generate gt_boxes in data_dict with prediction
"""

@torch.no_grad()
def construct_pseudo_label(boxes):
    box_list = []
    num_gt_list = []
    for bs_idx, box in enumerate(boxes):
        box_preds = box['pred_boxes']
        label_preds = box['pred_labels'].float().unsqueeze(-1)
        num_gt_list.append(box_preds.shape[0])
        box_list.append(torch.cat([box_preds, label_preds], dim=1))
    batch_size = len(boxes)
    num_max_gt = max(num_gt_list)
    gt_boxes = box_list[0].new_zeros((batch_size, num_max_gt, 8))
    for bs_idx in range(batch_size):
        num_gt = num_gt_list[bs_idx]
        gt_boxes[bs_idx, :num_gt, :] = box_list[bs_idx]
    return gt_boxes