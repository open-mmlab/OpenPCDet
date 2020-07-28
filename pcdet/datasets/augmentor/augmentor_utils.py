import numpy as np
from ...utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        if gt_boxes.size > 0:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        if gt_boxes.size > 0:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    if gt_boxes.size > 0:
        gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
        gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    if gt_boxes.size > 0:
        gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points
