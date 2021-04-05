import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

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
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def random_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    points[:, 0] += offset
    gt_boxes[:, 0] += offset

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7] += offset

    return gt_boxes, points


def random_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    points[:, 1] += offset
    gt_boxes[:, 1] += offset

    if gt_boxes.shape[1] > 8:
        gt_boxes[:, 8] += offset

    return gt_boxes, points


def random_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    points[:, 2] += offset
    gt_boxes[:, 2] += offset

    return gt_boxes, points


def local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    for box in gt_boxes:
        filter_arr = filter_points_in_box(points, box)
        points_in_box = points[filter_arr]
        for point in points:
            if np.isin(point, points_in_box).all():
                point[0] += offset

    gt_boxes[:, 0] += offset

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7] += offset

    return gt_boxes, points


def local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    for box in gt_boxes:
        filter_arr = filter_points_in_box(points, box)
        points_in_box = points[filter_arr]
        for point in points:
            if np.isin(point, points_in_box).all():
                point[1] += offset

    gt_boxes[:, 1] += offset

    if gt_boxes.shape[1] > 8:
        gt_boxes[:, 8] += offset

    return gt_boxes, points


def local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    offset = np.random.uniform(offset_range[0], offset_range[1])

    for box in gt_boxes:
        filter_arr = filter_points_in_box(points, box)
        points_in_box = points[filter_arr]
        for point in points:
            if np.isin(point, points_in_box).all():
                point[2] += offset

    gt_boxes[:, 2] += offset

    return gt_boxes, points


def local_scaling(gt_boxes, points, scale_range):
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
    for box in gt_boxes:
        filter_arr = filter_points_in_box(points, box)
        points_in_box = points[filter_arr]

        for point in points:
            if np.isin(point, points_in_box).all():
                # tranlation to axis center
                point[0] -= box[0]
                point[1] -= box[1]
                point[2] -= box[2]

                # apply scaling
                point[:3] *= noise_scale

                # tranlation back to original position
                point[0] += box[0]
                point[1] += box[1]
                point[2] += box[2]

    gt_boxes[:, 3:6] *= noise_scale
    return gt_boxes, points


def local_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])

    for box in gt_boxes:
        filter_arr = filter_points_in_box(points, box)
        points_in_box = points[filter_arr]
        for point in points:
            if np.isin(point, points_in_box).all():
                centroid_x = box[0]
                centroid_y = box[1]
                centroid_z = box[2]

                #print("BEFORE:\n" + str(point))
                # tranlation to axis center
                point[0] -= centroid_x
                point[1] -= centroid_y
                point[2] -= centroid_z
                box[0] -= centroid_x
                box[1] -= centroid_y
                box[2] -= centroid_z

                # apply rotation
                point[:] = common_utils.rotate_points_along_z(point[np.newaxis, np.newaxis, :], np.array([noise_rotation]))[0][0]
                box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][0]

                # tranlation back to original position
                point[0] += centroid_x
                point[1] += centroid_y
                point[2] += centroid_z
                box[0] += centroid_x
                box[1] += centroid_y
                box[2] += centroid_z
                #print("AFTER:\n" + str(point))

    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 8:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def filter_points_in_box(points, gt_box):
    filter_arr = []
    for point in points:
        x, y, z, dx, dy, dz = gt_box[0], gt_box[1], gt_box[2], gt_box[3], gt_box[4], gt_box[5]
        if point[0] <= x + dx/2 and point[0] >= x - dx/2 \
            and point[1] <= y + dy/2 and point[1] >= y - dy/2 \
                and point[2] <= z + dz/2 and point[2] >= z - dz/2:
                    filter_arr.append(True)
        else:
            filter_arr.append(False)

    # points_in_box = points[filter_arr]
    return filter_arr
