import numpy as np
import torch
import utils.iou3d.iou3d_utils as iou3d_utils
from ...config import cfg


def proposal_target_layer(input_dict, sequence_type=False):
    rois = input_dict['rois']
    roi_raw_scores = input_dict['roi_raw_scores']
    roi_labels = input_dict['roi_labels']
    gt_boxes = input_dict['gt_boxes']  # (B, N, 7 + ? + 1)

    if sequence_type:
        gt_boxes = torch.cat([gt_boxes, gt_boxes])

    batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels = \
        sample_rois_for_rcnn(rois, gt_boxes, roi_raw_scores, roi_labels)

    # regression valid mask
    reg_valid_mask = (batch_roi_iou > cfg.RCNN_STAGE.REG_FG_THRESH).long()

    # classification label
    if cfg.RCNN_STAGE.CLS_SCORE_TYPE == 'cls':
        batch_cls_label = (batch_roi_iou > cfg.RCNN_STAGE.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > cfg.RCNN_STAGE.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN_STAGE.CLS_FG_THRESH)
        batch_cls_label[invalid_mask > 0] = -1
    elif cfg.RCNN_STAGE.CLS_SCORE_TYPE == 'roi_iou':
        fg_mask = batch_roi_iou > cfg.RCNN_STAGE.CLS_FG_THRESH
        bg_mask = batch_roi_iou < cfg.RCNN_STAGE.CLS_BG_THRESH
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        batch_cls_label = (fg_mask > 0).float()
        batch_cls_label[interval_mask] = batch_roi_iou[interval_mask] * 2 - 0.5
    elif cfg.RCNN_STAGE.CLS_SCORE_TYPE == 'rcnn_iou':
        batch_cls_label = batch_roi_iou
    else:
        raise NotImplementedError

    if sequence_type:
        batch_size = rois.shape[0]//2
        batch_cls_label[batch_size:, ...] = -1
        reg_valid_mask[batch_size:, ...] = 0

    output_dict = {'rcnn_cls_labels': batch_cls_label.view(-1),
                   'reg_valid_mask': reg_valid_mask.view(-1),
                   'gt_of_rois': batch_gt_of_rois,
                   'gt_iou': batch_roi_iou,
                   'rois': batch_rois,
                   'roi_raw_scores': batch_roi_raw_scores,
                   'roi_labels': batch_roi_labels}
    return output_dict


def sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d, roi_raw_scores, roi_labels):
    """
    :param roi_boxes3d: (B, M, 7 + ?) [x, y, z, w, l, h, ry] in LiDAR coords
    :param gt_boxes3d: (B, N, 7 + ? + 1) [x, y, z, w, l, h, ry, class]
    :param roi_raw_scores: (B, N)
    :param roi_labels: (B, N)
    :return
        batch_rois: (B, N, 7)
        batch_gt_of_rois: (B, N, 7 + 1)
        batch_roi_iou: (B, N)
    """
    batch_size = roi_boxes3d.size(0)

    fg_rois_per_image = int(np.round(cfg.RCNN_STAGE.FG_RATIO * cfg.RCNN_STAGE.ROI_PER_IMAGE))

    code_size = roi_boxes3d.shape[-1]
    # batch_labels = gt_boxes3d.new(batch_size, rois_per_image).zero_()
    batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN_STAGE.ROI_PER_IMAGE, code_size).zero_()
    batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN_STAGE.ROI_PER_IMAGE, code_size + 1).zero_()
    batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN_STAGE.ROI_PER_IMAGE).zero_()
    batch_roi_raw_scores = gt_boxes3d.new(batch_size, cfg.RCNN_STAGE.ROI_PER_IMAGE).zero_()
    batch_roi_labels = gt_boxes3d.new(batch_size, cfg.RCNN_STAGE.ROI_PER_IMAGE).zero_().long()

    for idx in range(batch_size):
        cur_roi, cur_gt, cur_roi_raw_scores, cur_roi_labels = \
            roi_boxes3d[idx], gt_boxes3d[idx], roi_raw_scores[idx], roi_labels[idx]

        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if len(cfg.CLASSES) == 1:
            box_type = 'nuscenes' if cfg.DATASET == 'nuscenes' else 'lidar'
            try:
                iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7], box_type=box_type)  # (M, N)
            except:
                print(cur_roi.shape, cur_gt.shape)
                import pdb
                pdb.set_trace()
            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
        else:
            cur_gt_labels = cur_gt[:, -1].long()
            max_overlaps, gt_assignment = get_maxiou3d_with_same_class(cur_roi, cur_roi_labels,
                                                                       cur_gt[:, 0:7], cur_gt_labels)

        # sample fg, easy_bg, hard_bg
        fg_thresh = min(cfg.RCNN_STAGE.REG_FG_THRESH, cfg.RCNN_STAGE.CLS_FG_THRESH)
        fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)

        # TODO: this will mix the fg and bg when CLS_BG_THRESH_LO < iou < CLS_BG_THRESH
        # fg_inds = torch.cat((fg_inds, roi_assignment), dim=0)  # consider the roi which has max_iou3d with gt as fg

        easy_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN_STAGE.CLS_BG_THRESH_LO)).view(-1)
        hard_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN_STAGE.REG_FG_THRESH) &
                                     (max_overlaps >= cfg.RCNN_STAGE.CLS_BG_THRESH_LO)).view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = cfg.RCNN_STAGE.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(cfg.RCNN_STAGE.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_image = cfg.RCNN_STAGE.ROI_PER_IMAGE
            bg_rois_per_this_image = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = cfg.RCNN_STAGE.ROI_PER_IMAGE
            bg_inds = sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

            fg_rois_per_this_image = 0
        else:
            print('Iou3D.shape=', iou3d.shape)
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            import pdb
            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list, roi_score_list, roi_labels_list = [], [], [], [], []
        if fg_rois_per_this_image > 0:
            fg_rois_src = cur_roi[fg_inds]
            gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]

            iou3d_src = max_overlaps[fg_inds]
            if cfg.RCNN_STAGE.ROI_FG_AUG_TIMES > 0:
                fg_rois, fg_iou3d = aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois[:, 0:7], iou3d_src,
                                                           aug_times=cfg.RCNN_STAGE.ROI_FG_AUG_TIMES)
            else:
                fg_rois, fg_iou3d = fg_rois_src, iou3d_src

            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)
            roi_score_list.append(cur_roi_raw_scores[fg_inds])
            roi_labels_list.append(cur_roi_labels[fg_inds])

        if bg_rois_per_this_image > 0:
            bg_rois_src = cur_roi[bg_inds]
            gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
            iou3d_src = max_overlaps[bg_inds]

            aug_times = 1 if cfg.RCNN_STAGE.ROI_FG_AUG_TIMES > 0 else 0
            if aug_times > 0:
                bg_rois, bg_iou3d = aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois[:, 0:7], iou3d_src, aug_times=aug_times)
            else:
                bg_rois, bg_iou3d = bg_rois_src, iou3d_src

            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)
            roi_score_list.append(cur_roi_raw_scores[bg_inds])
            roi_labels_list.append(cur_roi_labels[bg_inds])

        rois = torch.cat(roi_list, dim=0)
        iou_of_rois = torch.cat(roi_iou_list, dim=0)
        gt_of_rois = torch.cat(roi_gt_list, dim=0)
        cur_roi_raw_scores = torch.cat(roi_score_list, dim=0)
        cur_roi_labels = torch.cat(roi_labels_list, dim=0)

        batch_rois[idx] = rois
        batch_gt_of_rois[idx] = gt_of_rois
        batch_roi_iou[idx] = iou_of_rois
        batch_roi_raw_scores[idx] = cur_roi_raw_scores
        batch_roi_labels[idx] = cur_roi_labels

    return batch_rois, batch_gt_of_rois, batch_roi_iou, batch_roi_raw_scores, batch_roi_labels


def get_maxiou3d_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
    """
    :param rois: (N, 7)
    :param roi_labels: (N)
    :param gt_boxes: (N, 8)
    :return:
    """
    max_overlaps = rois.new_zeros(rois.shape[0])
    gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

    for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
        roi_mask = (roi_labels == k)
        gt_mask = (gt_labels == k)
        if roi_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_roi = rois[roi_mask]
            cur_gt = gt_boxes[gt_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)

            box_type = 'nuscenes' if cfg.DATASET == 'nuscenes' else 'lidar'
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt, box_type=box_type)  # (M, N)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[roi_mask] = cur_max_overlaps
            gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

    return max_overlaps, gt_assignment


def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
    if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
        hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN_STAGE.HARD_BG_RATIO)
        easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        hard_bg_inds = hard_bg_inds[rand_idx]

        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        easy_bg_inds = easy_bg_inds[rand_idx]

        bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
    elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
        hard_bg_rois_num = bg_rois_per_this_image
        # sampling hard bg
        rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
        bg_inds = hard_bg_inds[rand_idx]
    elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
        easy_bg_rois_num = bg_rois_per_this_image
        # sampling easy bg
        rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
        bg_inds = easy_bg_inds[rand_idx]
    else:
        raise NotImplementedError

    return bg_inds
