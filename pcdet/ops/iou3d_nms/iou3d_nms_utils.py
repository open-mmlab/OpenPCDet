"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019.
"""
import torch
from ...utils import box_utils
from . import iou3d_nms_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, w, l, h, ry]  in LiDAR
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = box_utils.boxes3d_to_bevboxes_lidar_torch(boxes_a)
    boxes_b_bev = box_utils.boxes3d_to_bevboxes_lidar_torch(boxes_b)
    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).view(-1, 1)
    boxes_a_height_min = boxes_a[:, 2].view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).view(1, -1)
    boxes_b_height_min = boxes_b[:, 2].view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


if __name__ == '__main__':
    pass

