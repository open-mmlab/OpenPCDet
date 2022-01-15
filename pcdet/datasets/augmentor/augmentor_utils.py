import numpy as np
import math
import copy
from ...utils import common_utils
from ...utils import box_utils


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


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)
        
        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes


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
    
    # if gt_boxes.shape[1] > 7:
    #     gt_boxes[:, 7] += offset
    
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
    
    # if gt_boxes.shape[1] > 8:
    #     gt_boxes[:, 8] += offset
    
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


def random_local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 0] += offset
        
        gt_boxes[idx, 0] += offset
    
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[idx, 7] += offset
    
    return gt_boxes, points


def random_local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 1] += offset
        
        gt_boxes[idx, 1] += offset
    
        # if gt_boxes.shape[1] > 8:
        #     gt_boxes[idx, 8] += offset
    
    return gt_boxes, points


def random_local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 2] += offset
        
        gt_boxes[idx, 2] += offset
    
    return gt_boxes, points


def global_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    # threshold = max - length * uniform(0 ~ 0.2)
    threshold = np.max(points[:, 2]) - intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    
    points = points[points[:, 2] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] < threshold]
    return gt_boxes, points


def global_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 2]) + intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    points = points[points[:, 2] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] > threshold]
    
    return gt_boxes, points


def global_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.max(points[:, 1]) - intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] < threshold]
    
    return gt_boxes, points


def global_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 1]) + intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] > threshold]
    
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
    
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # augs[f'object_{idx}'] = noise_scale
        points_in_box, mask = get_points_in_box(points, box)
        
        # tranlation to axis center
        points[mask, 0] -= box[0]
        points[mask, 1] -= box[1]
        points[mask, 2] -= box[2]
        
        # apply scaling
        points[mask, :3] *= noise_scale
        
        # tranlation back to original position
        points[mask, 0] += box[0]
        points[mask, 1] += box[1]
        points[mask, 2] += box[2]
        
        gt_boxes[idx, 3:6] *= noise_scale
    return gt_boxes, points


def local_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # augs[f'object_{idx}'] = noise_rotation
        points_in_box, mask = get_points_in_box(points, box)
        
        centroid_x = box[0]
        centroid_y = box[1]
        centroid_z = box[2]
        
        # tranlation to axis center
        points[mask, 0] -= centroid_x
        points[mask, 1] -= centroid_y
        points[mask, 2] -= centroid_z
        box[0] -= centroid_x
        box[1] -= centroid_y
        box[2] -= centroid_z
        
        # apply rotation
        points[mask, :] = common_utils.rotate_points_along_z(points[np.newaxis, mask, :], np.array([noise_rotation]))[0]
        box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][0]
        
        # tranlation back to original position
        points[mask, 0] += centroid_x
        points[mask, 1] += centroid_y
        points[mask, 2] += centroid_z
        box[0] += centroid_x
        box[1] += centroid_y
        box[2] += centroid_z
        
        gt_boxes[idx, 6] += noise_rotation
        if gt_boxes.shape[1] > 8:
            gt_boxes[idx, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[idx, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]
    
    return gt_boxes, points


def local_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z + dz / 2) - intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z - dz / 2) + intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] <= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y + dy / 2) - intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y - dy / 2) + intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] <= threshold))]
    
    return gt_boxes, points


def get_points_in_box(points, gt_box):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz
    
    MARGIN = 1e-1
    cosa, sina = math.cos(-rz), math.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    
    mask = np.logical_and(abs(shift_z) <= dz / 2.0, 
                          np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
                                         abs(local_y) <= dy / 2.0 + MARGIN))
    
    points = points[mask]
    
    return points, mask


def get_pyramids(boxes):
    pyramid_orders = np.array([
        [0, 1, 5, 4],
        [4, 5, 6, 7],
        [7, 6, 2, 3],
        [3, 2, 1, 0],
        [1, 2, 6, 5],
        [0, 4, 7, 3]
    ])
    boxes_corners = box_utils.boxes_to_corners_3d(boxes).reshape(-1, 24)
    
    pyramid_list = []
    for order in pyramid_orders:
        # frustum polygon: 5 corners, 5 surfaces
        pyramid = np.concatenate((
            boxes[:, 0:3],
            boxes_corners[:, 3 * order[0]: 3 * order[0] + 3],
            boxes_corners[:, 3 * order[1]: 3 * order[1] + 3],
            boxes_corners[:, 3 * order[2]: 3 * order[2] + 3],
            boxes_corners[:, 3 * order[3]: 3 * order[3] + 3]), axis=1)
        pyramid_list.append(pyramid[:, None, :])
    pyramids = np.concatenate(pyramid_list, axis=1)  # [N, 6, 15], 15=5*3
    return pyramids


def one_hot(x, num_class=1):
    if num_class is None:
        num_class = 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


def points_in_pyramids_mask(points, pyramids):
    pyramids = pyramids.reshape(-1, 5, 3)
    flags = np.zeros((points.shape[0], pyramids.shape[0]), dtype=np.bool)
    for i, pyramid in enumerate(pyramids):
        flags[:, i] = np.logical_or(flags[:, i], box_utils.in_hull(points[:, 0:3], pyramid))
    return flags


def local_pyramid_dropout(gt_boxes, points, dropout_prob, pyramids=None):
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    drop_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
    drop_pyramid_one_hot = one_hot(drop_pyramid_indices, num_class=6)
    drop_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= dropout_prob
    if np.sum(drop_box_mask) != 0:
        drop_pyramid_mask = (np.tile(drop_box_mask[:, None], [1, 6]) * drop_pyramid_one_hot) > 0
        drop_pyramids = pyramids[drop_pyramid_mask]
        point_masks = points_in_pyramids_mask(points, drop_pyramids)
        points = points[np.logical_not(point_masks.any(-1))]
    # print(drop_box_mask)
    pyramids = pyramids[np.logical_not(drop_box_mask)]
    return gt_boxes, points, pyramids


def local_pyramid_sparsify(gt_boxes, points, prob, max_num_pts, pyramids=None):
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    if pyramids.shape[0] > 0:
        sparsity_prob, sparsity_num = prob, max_num_pts
        sparsify_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
        sparsify_pyramid_one_hot = one_hot(sparsify_pyramid_indices, num_class=6)
        sparsify_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= sparsity_prob
        sparsify_pyramid_mask = (np.tile(sparsify_box_mask[:, None], [1, 6]) * sparsify_pyramid_one_hot) > 0
        # print(sparsify_box_mask)
        
        pyramid_sampled = pyramids[sparsify_pyramid_mask]  # (-1,6,5,3)[(num_sample,6)]
        # print(pyramid_sampled.shape)
        pyramid_sampled_point_masks = points_in_pyramids_mask(points, pyramid_sampled)
        pyramid_sampled_points_num = pyramid_sampled_point_masks.sum(0)  # the number of points in each surface pyramid
        valid_pyramid_sampled_mask = pyramid_sampled_points_num > sparsity_num  # only much than sparsity_num should be sparse
        
        sparsify_pyramids = pyramid_sampled[valid_pyramid_sampled_mask]
        if sparsify_pyramids.shape[0] > 0:
            point_masks = pyramid_sampled_point_masks[:, valid_pyramid_sampled_mask]
            remain_points = points[
                np.logical_not(point_masks.any(-1))]  # points which outside the down sampling pyramid
            to_sparsify_points = [points[point_masks[:, i]] for i in range(point_masks.shape[1])]
            
            sparsified_points = []
            for sample in to_sparsify_points:
                sampled_indices = np.random.choice(sample.shape[0], size=sparsity_num, replace=False)
                sparsified_points.append(sample[sampled_indices])
            sparsified_points = np.concatenate(sparsified_points, axis=0)
            points = np.concatenate([remain_points, sparsified_points], axis=0)
        pyramids = pyramids[np.logical_not(sparsify_box_mask)]
    return gt_boxes, points, pyramids


def local_pyramid_swap(gt_boxes, points, prob, max_num_pts, pyramids=None):
    def get_points_ratio(points, pyramid):
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        alphas = ((points[:, 0:3] - pyramid[3:6]) * vector_0).sum(-1) / np.power(vector_0, 2).sum()
        betas = ((points[:, 0:3] - pyramid[3:6]) * vector_1).sum(-1) / np.power(vector_1, 2).sum()
        gammas = ((points[:, 0:3] - surface_center) * vector_2).sum(-1) / np.power(vector_2, 2).sum()
        return [alphas, betas, gammas]
    
    def recover_points_by_ratio(points_ratio, pyramid):
        alphas, betas, gammas = points_ratio
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        points = (alphas[:, None] * vector_0 + betas[:, None] * vector_1) + pyramid[3:6] + gammas[:, None] * vector_2
        return points
    
    def recover_points_intensity_by_ratio(points_intensity_ratio, max_intensity, min_intensity):
        return points_intensity_ratio * (max_intensity - min_intensity) + min_intensity
    
    # swap partition
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    swap_prob, num_thres = prob, max_num_pts
    swap_pyramid_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= swap_prob
    
    if swap_pyramid_mask.sum() > 0:
        point_masks = points_in_pyramids_mask(points, pyramids)
        point_nums = point_masks.sum(0).reshape(pyramids.shape[0], -1)  # [N, 6]
        non_zero_pyramids_mask = point_nums > num_thres  # ingore dropout pyramids or highly occluded pyramids
        selected_pyramids = non_zero_pyramids_mask * swap_pyramid_mask[:,
                                                     None]  # selected boxes and all their valid pyramids
        # print(selected_pyramids)
        if selected_pyramids.sum() > 0:
            # get to_swap pyramids
            index_i, index_j = np.nonzero(selected_pyramids)
            selected_pyramid_indices = [np.random.choice(index_j[index_i == i]) \
                                            if e and (index_i == i).any() else 0 for i, e in
                                        enumerate(swap_pyramid_mask)]
            selected_pyramids_mask = selected_pyramids * one_hot(selected_pyramid_indices, num_class=6) == 1
            to_swap_pyramids = pyramids[selected_pyramids_mask]
            
            # get swapped pyramids
            index_i, index_j = np.nonzero(selected_pyramids_mask)
            non_zero_pyramids_mask[selected_pyramids_mask] = False
            swapped_index_i = np.array([np.random.choice(np.where(non_zero_pyramids_mask[:, j])[0]) if \
                                            np.where(non_zero_pyramids_mask[:, j])[0].shape[0] > 0 else
                                        index_i[i] for i, j in enumerate(index_j.tolist())])
            swapped_indicies = np.concatenate([swapped_index_i[:, None], index_j[:, None]], axis=1)
            swapped_pyramids = pyramids[
                swapped_indicies[:, 0].astype(np.int32), swapped_indicies[:, 1].astype(np.int32)]
            
            # concat to_swap&swapped pyramids
            swap_pyramids = np.concatenate([to_swap_pyramids, swapped_pyramids], axis=0)
            swap_point_masks = points_in_pyramids_mask(points, swap_pyramids)
            remain_points = points[np.logical_not(swap_point_masks.any(-1))]
            
            # swap pyramids
            points_res = []
            num_swapped_pyramids = swapped_pyramids.shape[0]
            for i in range(num_swapped_pyramids):
                to_swap_pyramid = to_swap_pyramids[i]
                swapped_pyramid = swapped_pyramids[i]
                
                to_swap_points = points[swap_point_masks[:, i]]
                swapped_points = points[swap_point_masks[:, i + num_swapped_pyramids]]
                # for intensity transform
                to_swap_points_intensity_ratio = (to_swap_points[:, -1:] - to_swap_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (to_swap_points[:, -1:].max() - to_swap_points[:, -1:].min()),
                                                     1e-6, 1)
                swapped_points_intensity_ratio = (swapped_points[:, -1:] - swapped_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (swapped_points[:, -1:].max() - swapped_points[:, -1:].min()),
                                                     1e-6, 1)
                
                to_swap_points_ratio = get_points_ratio(to_swap_points, to_swap_pyramid.reshape(15))
                swapped_points_ratio = get_points_ratio(swapped_points, swapped_pyramid.reshape(15))
                new_to_swap_points = recover_points_by_ratio(swapped_points_ratio, to_swap_pyramid.reshape(15))
                new_swapped_points = recover_points_by_ratio(to_swap_points_ratio, swapped_pyramid.reshape(15))
                # for intensity transform
                new_to_swap_points_intensity = recover_points_intensity_by_ratio(
                    swapped_points_intensity_ratio, to_swap_points[:, -1:].max(),
                    to_swap_points[:, -1:].min())
                new_swapped_points_intensity = recover_points_intensity_by_ratio(
                    to_swap_points_intensity_ratio, swapped_points[:, -1:].max(),
                    swapped_points[:, -1:].min())
                
                # new_to_swap_points = np.concatenate([new_to_swap_points, swapped_points[:, -1:]], axis=1)
                # new_swapped_points = np.concatenate([new_swapped_points, to_swap_points[:, -1:]], axis=1)
                
                new_to_swap_points = np.concatenate([new_to_swap_points, new_to_swap_points_intensity], axis=1)
                new_swapped_points = np.concatenate([new_swapped_points, new_swapped_points_intensity], axis=1)
                
                points_res.append(new_to_swap_points)
                points_res.append(new_swapped_points)
            
            points_res = np.concatenate(points_res, axis=0)
            points = np.concatenate([remain_points, points_res], axis=0)
    return gt_boxes, points
