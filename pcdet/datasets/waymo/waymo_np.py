"""
Transaltion between range image and point cloud with Numpy 
Written by Jihan Yang
All Rights Reserved 2020-2021.
"""

import numpy as np

def group_max(groups, data):
    # this is only needed if groups is unsorted
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return groups[index], data[index]


def scatter_nd_with_pool_np(index, value, shape, pool_method=group_max):
    """Similar as tf.scatter_nd but allows custom pool method.
    tf.scatter_nd accumulates (sums) values if there are duplicate indices.
    Args:
    index: [N, 2] np.array. Inner dims are coordinates along height (row) and then
        width (col).
    value: [N, ...] np.array. Values to be scattered.
    shape: (height,width) list that specifies the shape of the output tensor.
    pool_method: pool method when there are multiple points scattered to one
        location.
    Returns:
    image: tensor of shape with value scattered. Missing pixels are set to 0.
    """
    if len(shape) != 2:
        raise ValueError('shape must be of size 2')
    width = shape[1]
    # idx: [N]
    idx_1d = index[:, 0] * width + index[:, 1]
    index_encoded, value_pooled = pool_method(idx_1d, value)

    index_unique = np.stack(
        [index_encoded // width, np.mod(index_encoded, width)], axis=-1
    )

    image = np.zeros(shape, dtype=np.float32)
    image[index_unique[:, 0], index_unique[:, 1]] = value_pooled

    return image


def build_range_image_from_point_cloud_np(points_frame,
                                          num_points,
                                          inclination,
                                          range_image_size,
                                          dtype=np.float32):
    """Build virtual range image from point cloud assuming uniform azimuth.
    Args:
    points_frame: np array with shape [N, 3] in the vehicle frame.
    num_points: int32 saclar indicating the number of points for each frame.
    inclination: np array of shape [H] that is the inclination angle per
        row. sorted from highest value to lowest.
    range_image_size: a size 2 [height, width] list that configures the size of
        the range image.
    dtype: the data type to use.
    Returns:
    range_images : [H, W, 3] or [B, H, W] tensor. Range images built from the
        given points. Data type is the same as that of points_frame. 0.0
        is populated when a pixel is missing.
    ri_indices: np int32 array [N, 2]. It represents the range image index
        for each point.
    ri_ranges: [N] tensor. It represents the distance between a point and
        sensor frame origin of each point.
    """
    points_frame_dtype = points_frame.dtype

    points_frame = points_frame.astype(dtype)
    inclination = inclination.astype(dtype)

    height, width = range_image_size

    # Points in sensor frame
    # [N, 3]
    points = points_frame
    # [N]
    xy_norm = np.linalg.norm(points[..., 0:2], axis=-1)
    # [N]
    point_inclination = np.arctan2(points[..., 2], xy_norm)
    # [N, H]
    point_inclination_diff = np.abs(
        np.expand_dims(point_inclination, axis=-1) -
        np.expand_dims(inclination, axis=0))
    # [N]
    point_ri_row_indices = np.argmin(point_inclination_diff, axis=-1)

    # [N], within [-pi, pi]
    point_azimuth = np.arctan2(points[..., 1].astype(np.float64), points[..., 0].astype(np.float64)).astype(np.float64) - 1e-9

    # solve the problem of np.float32 accuracy problem
    point_azimuth_gt_pi_mask = point_azimuth > np.pi
    point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    point_azimuth = point_azimuth - point_azimuth_gt_pi_mask.astype(np.float64) * 2 * np.pi
    point_azimuth = point_azimuth + point_azimuth_lt_minus_pi_mask.astype(np.float64) * 2 * np.pi

    # [N].
    point_ri_col_indices = width - 1.0 + 0.5 - (point_azimuth +
                                                np.pi) / (2.0 * np.pi) * width
    point_ri_col_indices = np.round(point_ri_col_indices).astype(np.int32)

    assert (point_ri_col_indices >= 0).all()
    assert (point_ri_col_indices < width).all()

    # [N, 2]
    ri_indices = np.stack([point_ri_row_indices, point_ri_col_indices], -1)
    # [N]
    ri_ranges = np.linalg.norm(points, axis=-1).astype(points_frame_dtype)

    def fn(args):
        """Builds a range image for each frame.
        Args:
            args: a tuple containing:
            - ri_index: [N, 2] int tensor.
            - ri_value: [N] float tensor.
            - num_point: scalar tensor
        Returns:
            range_image: [H, W]
        """
        ri_index, ri_value, num_point = args

        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point, ...]
        range_image = scatter_nd_with_pool_np(
            ri_index, ri_value, [height, width], group_max
        )
        return range_image

    elems = [ri_indices, ri_ranges, num_points]

    range_images = fn(elems)

    return range_images, ri_indices, ri_ranges

def build_points_from_range_image_np(kitti_range_image,
                                     kitti_inclinations,
                                     kitti_azimuths):
    """build point clouds under lidar frame from range image.
    Args:
        kitti_range_image: [H, W] np.array, the range image range value.
        kitti_inclinations: [H] np.array, the defined beam inclinations.
        kitti_azimuths: [W] np.array, the defined beam azimuth.
    Returns:
        kitti_points: [N, 3] lidar points.
    """
    inclinations = np.repeat(kitti_inclinations[:, np.newaxis],
                             kitti_azimuths.shape[0], axis=1)
    azimuths = np.repeat(kitti_azimuths[np.newaxis, :],
                         kitti_inclinations.shape[0], axis=0)

    # [H, W]
    range_image_mask = kitti_range_image > 0

    cos_azimuth = np.cos(azimuths)
    sin_azimuth = np.sin(azimuths)
    cos_incl = np.cos(inclinations)
    sin_incl = np.sin(inclinations)

    # [H, W].
    x = cos_azimuth * cos_incl * kitti_range_image
    y = sin_azimuth * cos_incl * kitti_range_image
    z = sin_incl * kitti_range_image

    # [H, W, 3]
    range_image_cartesian = np.stack([x, y, z], -1)

    kitti_points = range_image_cartesian[range_image_mask, :].reshape(-1, 3)

    return kitti_points
