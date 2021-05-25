"""
Translation between range image and point cloud with Numpy
Modified from the original version written by Jihan Yang that only support KITTI dataset
This version realize almost all function of official waymo version.
Written by Boyu Meng
All Rights Reserved.
"""

import numpy as np
import math

# A magic number that provides a good resolution we need for lidar range after
# quantization from float to uint16.

_RANGE_TO_METERS = 0.00585532144


def _encode_range(r):
    """Encodes lidar range from float to uint16.

    Args:
    r: np.array A float tensor represents lidar range.

    Returns:
    Encoded range with type as uint16.
    """
    encoded_r = r / _RANGE_TO_METERS
    assert (encoded_r >= 0).all()
    assert (encoded_r <= math.pow(2, 16) - 1.).all()
    return encoded_r.astype(np.uint16)


def _decode_range(r):
    """Decodes lidar range from integers to float32.

    Args:
    r: A integer tensor.

    Returns:
    Decoded range.
    """
    return r.astype(np.float32) * _RANGE_TO_METERS


def _encode_intensity(intensity):
    """Encodes lidar intensity from float to uint16.

    The integer value stored here is the upper 16 bits of a float. This
    preserves the exponent and truncates the mantissa to 7bits, which gives
    plenty of dynamic range and preserves about 3 decimal places of
    precision.

    Args:
    intensity: A float tensor represents lidar intensity.

    Returns:
    Encoded intensity with type as uint32.
    """
    if intensity.dtype != np.float32:
        raise TypeError('intensity must be of type float32')

    intensity.dtype = np.uint32
    intensity_uint32_shifted = np.right_shift(intensity, 16)
    return intensity_uint32_shifted.astype(np.uint16)


def _decode_intensity(intensity):
    """Decodes lidar intensity from uint16 to float32.

    The given intensity is encoded with _encode_intensity.

    Args:
    intensity: A uint16 tensor represents lidar intensity.

    Returns:
    Decoded intensity with type as float32.
    """
    if intensity.dtype != np.uint16:
        raise TypeError('intensity must be of type uint16')

    intensity_uint32 = intensity.astype(np.uint32)
    intensity_uint32_shifted = np.left_shift(intensity_uint32, 16)
    intensity_uint32_shifted.dtype = np.float32
    return intensity_uint32_shifted


def _encode_elongation(elongation):
    """Encodes lidar elongation from float to uint8.

    Args:
    elongation: A float tensor represents lidar elongation.

    Returns:
    Encoded lidar elongation.
    """
    encoded_elongation = elongation / _RANGE_TO_METERS
    assert (encoded_elongation >= 0).all()
    assert (encoded_elongation <= math.pow(2, 8) - 1.).all()
    return encoded_elongation.astype(np.uint8)


def _decode_elongation(elongation):
    """Decodes lidar elongation from uint8 to float.

    Args:
    elongation: A uint8 tensor represents lidar elongation.

    Returns:
    Decoded lidar elongation.
    """
    return elongation.astype(np.float32) * _RANGE_TO_METERS


def encode_lidar_features(lidar_point_feature):
    """Encodes lidar features (range, intensity, elongation).

    This function encodes lidar point features such that all features have the
    same ordering as lidar range.

    Args:
    lidar_point_feature: [N, 3] float32 tensor.

    Returns:
    [N, 3] int64 tensors that encodes lidar_point_feature.
    """
    if lidar_point_feature.dtype != np.float32:
        raise TypeError('lidar_point_feature must be of type float32.')
    feature_tuple = list(map(np.squeeze, np.split(lidar_point_feature, lidar_point_feature.shape[-1], axis=-1)))
    r = feature_tuple[0]
    intensity = feature_tuple[1]

    encoded_r = _encode_range(r).astype(np.uint32)
    encoded_intensity = _encode_intensity(intensity).astype(np.uint32)

    encoded_r_shifted = np.left_shift(encoded_r, 16)
    encoded_intensity = np.bitwise_or(encoded_r_shifted, encoded_intensity).astype(np.int64)
    encoded_r = encoded_r.astype(np.int64)

    if len(feature_tuple) > 2:
        elongation = feature_tuple[2]
        encoded_elongation = _encode_elongation(elongation).astype(np.uint32)
        encoded_elongation = np.bitwise_or(encoded_r_shifted, encoded_elongation).astype(np.int64)

        return np.stack([encoded_r, encoded_intensity, encoded_elongation], axis=-1)

    return np.stack([encoded_r, encoded_intensity], axis=-1)


def decode_lidar_features(lidar_point_feature):
    """Decodes lidar features (range, intensity, elongation).

    This function decodes lidar point features encoded by 'encode_lidar_features'.

    Args:
      lidar_point_feature: [N, 3] int64 tensor.

    Returns:
      [N, 3] float tensors that encodes lidar_point_feature.
    """

    feature_tuple = list(map(np.squeeze, np.split(lidar_point_feature, lidar_point_feature.shape[-1], axis=-1)))
    r = feature_tuple[0]
    intensity = feature_tuple[1]
    # r, intensity, elongation = list(map(np.squeeze, np.split(lidar_point_feature, 3, axis=-1)))

    decoded_r = _decode_range(r)
    intensity = np.bitwise_and(intensity, int(0xFFFF))
    decoded_intensity = _decode_intensity(intensity.astype(np.uint16))
    if len(feature_tuple) > 2:
        elongation = feature_tuple[2]
        elongation = np.bitwise_and(elongation, int(0xFF))
        decoded_elongation = _decode_elongation(elongation.astype(np.uint8))

        return np.stack([decoded_r, decoded_intensity, decoded_elongation], axis=-1)

    return np.stack([decoded_r, decoded_intensity], axis=-1)


def group_max(groups, data, keep='closest'):
    # this is only needed if groups is unsorted
    if len(data.shape) > 1:
        order = np.lexsort((data[:, 0], groups))
    else:
        order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    # farthest
    if keep=='farthest':
        index[-1] = True
        index[:-1] = groups[1:] != groups[:-1]
    elif keep=='closest':
        # closest
        index[0] = True
        index[1:] = groups[1:] != groups[:-1]
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
    if len(value_pooled.shape) == 1:
        image = np.zeros(shape, dtype=np.int64)
    else:
        image = np.zeros([*shape, *value_pooled.shape[1:]], dtype=np.int64)
    image[index_unique[:, 0], index_unique[:, 1]] = value_pooled

    return image


def build_range_image_from_point_cloud_np(points_frame,
                                          num_points,
                                          inclination,
                                          range_image_size,
                                          extrinsic=None,
                                          point_features=None,
                                          dtype=np.float64):
    """Build virtual range image from point cloud assuming uniform azimuth.
    Args:
    points_frame: np array with shape [N, 3] in the vehicle frame.
    num_points: int32 saclar indicating the number of points for each frame.
    extrinsic: np array with shape [4, 4].
    inclination: np array of shape [H] that is the inclination angle per
        row. sorted from highest value to lowest.
    point_features: If not None, it is a np array with shape [N, 2] that
        represents lidar 'intensity' and 'elongation'.
    range_image_size: a size 2 [height, width] list that configures the size of
        the range image.
    dtype: the data type to use.
    Returns:
    range_images : [H, W, 3] or [H, W] tensor. Range images built from the
        given points. Data type is the same as that of points_frame. 0.0
        is populated when a pixel is missing.
    ri_indices: np int32 array [N, 2]. It represents the range image index
        for each point.
    ri_ranges: [N] tensor. It represents the distance between a point and
        sensor frame origin of each point.
    """

    def veh2laser(points_frame, extrinsic, dtype):

        points_frame = points_frame.astype(dtype)

        extrinsic = extrinsic.astype(dtype)
        # [4, 4]
        vehicle_to_laser = np.linalg.inv(extrinsic)
        # [3, 3]
        rotation = vehicle_to_laser[0:3, 0:3]
        # [1, 3]
        translation = np.expand_dims(vehicle_to_laser[0:3, 3], 0)

        # Points in sensor frame
        # [N, 3]
        # trans(R^-1)
        points = np.einsum('ij,kj->ik', points_frame, rotation) + translation

        # [1,], within [-pi, pi]
        az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
        return points, az_correction

    if extrinsic is not None:
        points, az_correction = veh2laser(points_frame, extrinsic, dtype)
    else:
        points = points_frame
        az_correction = 0

    points_frame_dtype = points_frame.dtype
    inclination = inclination.astype(dtype)
    height, width = range_image_size

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
    # point_azimuth = np.arctan2(points[..., 1].astype(np.float64), points[..., 0].astype(np.float64)).astype(
    #     np.float64) + az_correction - 1e-9
    point_azimuth = np.arctan2(points[..., 1], points[..., 0]) + az_correction - 1e-6

    # solve the problem of np.float32 accuracy problem
    point_azimuth_gt_pi_mask = point_azimuth > np.pi
    point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    point_azimuth = point_azimuth - point_azimuth_gt_pi_mask.astype(dtype) * 2 * np.pi
    point_azimuth = point_azimuth + point_azimuth_lt_minus_pi_mask.astype(dtype) * 2 * np.pi

    # [N].
    point_ri_col_indices = width - 1.0 + 0.5 - (point_azimuth +
                                                np.pi) / (2.0 * np.pi) * width
    point_ri_col_indices = np.round(point_ri_col_indices).astype(np.int32)

    try:
        assert (point_ri_col_indices >= 0).all()
        assert (point_ri_col_indices < width).all()
    except:
        import pudb
        pudb.set_trace()

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
        if len(args) == 3:
            ri_index, ri_value, num_point = args
        else:
            ri_index, ri_value, num_point, point_feature = args
            ri_value = np.concatenate([ri_value[..., np.newaxis], point_feature], axis=-1)
            ri_value = encode_lidar_features(ri_value)

        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point, ...]
        range_image = scatter_nd_with_pool_np(
            ri_index, ri_value, [height, width], group_max
        )
        if len(args) != 3:
            range_image = decode_lidar_features(range_image)
        return range_image

    elems = [ri_indices, ri_ranges, num_points]

    if point_features is not None:
        elems.append(point_features)
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
