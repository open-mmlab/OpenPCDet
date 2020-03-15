# This file is modified from https://github.com/traveller59/second.pytorch

import numpy as np
import numpy.random as npr
import numba
from ...utils import common_utils


def unmap(data, count, inds, fill=0):
    '''Unmap a subset of item (data) back to the original set of items (of
    size count)'''
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=((1.6, 3.9, 1.56),),
                            rotations=(0, np.pi / 2),
                            dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """

    anchor_range = np.array(anchor_range, dtype)
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(
        dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(common_utils.limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


class AnchorGeneratorRange(object):
    def __init__(self, anchor_ranges, sizes=((1.6, 3.9, 1.56),), rotations=(0, np.pi / 2), class_name=None,
                 match_threshold=-1, unmatch_threshold=-1, custom_values=None, dtype=np.float32, feature_map_size=None):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values
        self._feature_map_size = feature_map_size

    @property
    def class_name(self):
        return self._class_name

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def custom_values(self):
        return self.custom_values

    @property
    def feature_map_size(self):
        return self._feature_map_size

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        anchors = create_anchors_3d_range(feature_map_size, self._anchor_ranges, self._sizes,
                                                     self._rotations, self._dtype)
        if self._custom_values is not None:
            custom_values = np.zeros((*anchors.shape[:-1], len(self._custom_values)), dtype=self._dtype)
            for k in range(len(self._custom_values)):
                custom_values[..., k] = self._custom_values[k]
            anchors = np.concatenate((anchors, custom_values), axis=-1)
        return anchors


class TargetAssigner(object):
    def __init__(self, anchor_generators, pos_fraction, sample_size, region_similarity_fn_name, box_coder, logger=None):
        super().__init__()
        self.anchor_generators = anchor_generators
        self.pos_fraction = pos_fraction if pos_fraction >= 0 else None
        self.sample_size = sample_size
        self.region_similarity_calculator = getattr(self, region_similarity_fn_name)
        self.box_coder = box_coder
        self.logger = logger

    def generate_anchors(self, feature_map_size=None, use_multi_head=False):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self.anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self.anchor_generators]
        match_list, unmatch_list = [], []
        for anchor_generator, match_thresh, unmatch_thresh in zip(self.anchor_generators,
                                                                  matched_thresholds, unmatched_thresholds):
            if use_multi_head:
                anchors = anchor_generator.generate(anchor_generator.feature_map_size)  # (1, H, W, 2#, code_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
                ndim = len(anchor_generator.feature_map_size)
                anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)  # (2#, 1, H, W, code_size)
                anchors = anchors.reshape(-1, anchors.shape[-1])
            else:
                anchors = anchor_generator.generate(feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])

            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
        anchors = np.concatenate(anchors_list, axis=-2)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            'anchors': anchors,
            'matched_thresholds': matched_thresholds,
            'unmatched_thresholds': unmatched_thresholds
        }

    def generate_anchors_dict(self, feature_map_size, use_multi_head=False):
        anchors_list = []
        matched_thresholds = [a.match_threshold for a in self.anchor_generators]
        unmatched_thresholds = [a.unmatch_threshold for a in self.anchor_generators]
        match_list, unmatch_list = [], []
        anchors_dict = {a.class_name: {} for a in self.anchor_generators}
        for anchor_generator, match_thresh, unmatch_thresh in zip(self.anchor_generators,
                                                                  matched_thresholds, unmatched_thresholds):
            if use_multi_head:
                anchors = anchor_generator.generate(anchor_generator.feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])
                ndim = len(feature_map_size)
                anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            else:
                anchors = anchor_generator.generate(feature_map_size)
                anchors = anchors.reshape([*anchors.shape[:3], -1, anchors.shape[-1]])

            anchors_list.append(anchors)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]['anchors'] = anchors
            anchors_dict[class_name]['matched_thresholds'] = match_list[-1]
            anchors_dict[class_name]['unmatched_thresholds'] = unmatch_list[-1]
        return anchors_dict

    @staticmethod
    def nearest_iou_similarity(boxes1, boxes2):
        boxes1_bv = rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = rbbox2d_to_near_bbox(boxes2)
        ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret

    def assign_v2(self, anchors_dict, gt_boxes, anchors_mask=None, gt_classes=None, gt_names=None):
        prune_anchor_fn = None if anchors_mask is None else lambda _: np.where(anchors_mask)[0]

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self.region_similarity_calculator(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self.box_coder.encode_np(boxes, anchors)

        targets_list = []
        for class_name, anchor_dict in anchors_dict.items():
            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)
            targets = self.create_target_np(
                # anchor_dict['anchors'].reshape(-1, self.box_coder.code_size),
                anchor_dict['anchors'].reshape(-1, anchor_dict['anchors'].shape[-1]),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict['matched_thresholds'],
                unmatched_threshold=anchor_dict['unmatched_thresholds'],
                positive_fraction=self.pos_fraction,
                rpn_batch_size=self.sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size
            )
            targets_list.append(targets)
            feature_map_size = anchor_dict['anchors'].shape[:3]
        targets_dict = {
            'labels': [t['labels'] for t in targets_list],
            'bbox_targets': [t['bbox_targets'] for t in targets_list],
            'bbox_src_targets': [t['bbox_src_targets'] for t in targets_list],
            'bbox_outside_weights': [t['bbox_outside_weights'] for t in targets_list],
        }
        # bbox_targets: (H, W, num_anchors_per_loc, code_size)
        targets_dict['bbox_targets'] = np.concatenate([v.reshape(*feature_map_size, -1, self.box_coder.code_size)
                                                       for v in targets_dict['bbox_targets']], axis=-2)
        targets_dict['bbox_src_targets'] = np.concatenate([v.reshape(*feature_map_size, -1, self.box_coder.code_size)
                                                           for v in targets_dict['bbox_src_targets']], axis=-2)
        targets_dict['labels'] = np.concatenate([v.reshape(*feature_map_size, -1)
                                                 for v in targets_dict['labels']], axis=-1)
        targets_dict['bbox_outside_weights'] = np.concatenate([v.reshape(*feature_map_size, -1)
                                                               for v in targets_dict['bbox_outside_weights']], axis=-1)

        targets_dict['bbox_targets'] = targets_dict['bbox_targets'].reshape(-1, self.box_coder.code_size)
        targets_dict['bbox_src_targets'] = targets_dict['bbox_src_targets'].reshape(-1, self.box_coder.code_size)
        targets_dict['labels'] = targets_dict['labels'].reshape(-1)
        targets_dict['bbox_outside_weights'] = targets_dict['bbox_outside_weights'].reshape(-1)

        return targets_dict

    def assign_multihead(self, anchors_dict, gt_boxes, anchors_mask=None, gt_classes=None, gt_names=None):
        prune_anchor_fn = None if anchors_mask is None else lambda _: np.where(anchors_mask)[0]

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self.region_similarity_calculator(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self.box_coder.encode_np(boxes, anchors)

        targets_list = []
        for class_name, anchor_dict in anchors_dict.items():
            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)
            targets = self.create_target_np(
                # anchor_dict['anchors'].reshape(-1, self.box_coder.code_size),
                anchor_dict['anchors'].reshape(-1, anchor_dict['anchors'].shape[-1]),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict['matched_thresholds'],
                unmatched_threshold=anchor_dict['unmatched_thresholds'],
                positive_fraction=self.pos_fraction,
                rpn_batch_size=self.sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size
            )
            targets_list.append(targets)
        targets_dict = {
            'labels': [t['labels'] for t in targets_list],
            'bbox_targets': [t['bbox_targets'] for t in targets_list],
            'bbox_outside_weights': [t['bbox_outside_weights'] for t in targets_list],
        }
        # # bbox_targets: (H, W, num_anchors_per_loc, code_size)
        targets_dict['bbox_targets'] = np.concatenate([v.reshape(-1, self.box_coder.code_size)
                                                       for v in targets_dict['bbox_targets']], axis=0)
        targets_dict['labels'] = np.concatenate([v.reshape(-1) for v in targets_dict['labels']], axis=0)
        targets_dict['bbox_outside_weights'] = np.concatenate([v.reshape(-1)
                                                               for v in targets_dict['bbox_outside_weights']], axis=0)
        return targets_dict

    def create_target_np(self, all_anchors,
                         gt_boxes,
                         similarity_fn,
                         box_encoding_fn,
                         prune_anchor_fn=None,
                         gt_classes=None,
                         matched_threshold=0.6,
                         unmatched_threshold=0.45,
                         bbox_inside_weight=None,
                         positive_fraction=None,
                         rpn_batch_size=300,
                         norm_by_num_examples=False,
                         box_code_size=7):
        '''Modified from FAIR detectron.
        Args:
            all_anchors: [num_of_anchors, box_ndim] float tensor.
            gt_boxes: [num_gt_boxes, box_ndim] float tensor.
            similarity_fn: a function, accept anchors and gt_boxes, return
                similarity matrix(such as IoU).
            box_encoding_fn: a function, accept gt_boxes and anchors, return
                box encodings(offsets).
            prune_anchor_fn: a function, accept anchors, return indices that
                indicate valid anchors.
            gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
                start with 1.
            matched_threshold: float, iou greater than matched_threshold will
                be treated as positives.
            unmatched_threshold: float, iou smaller than unmatched_threshold will
                be treated as negatives.
            bbox_inside_weight: unused
            positive_fraction: [0-1] float or None. if not None, we will try to
                keep ratio of pos/neg equal to positive_fraction when sample.
                if there is not enough positives, it fills the rest with negatives
            rpn_batch_size: int. sample size
            norm_by_num_examples: bool. norm box_weight by number of examples, but
                I recommend to do this outside.
        Returns:
            labels, bbox_targets, bbox_outside_weights
        '''
        total_anchors = all_anchors.shape[0]
        if prune_anchor_fn is not None:
            inds_inside = prune_anchor_fn(all_anchors)
            anchors = all_anchors[inds_inside, :]
            if not isinstance(matched_threshold, float):
                matched_threshold = matched_threshold[inds_inside]
            if not isinstance(unmatched_threshold, float):
                unmatched_threshold = unmatched_threshold[inds_inside]
        else:
            anchors = all_anchors
            inds_inside = None
        num_inside = len(inds_inside) if inds_inside is not None else total_anchors
        box_ndim = all_anchors.shape[1]
        if self.logger is not None:
            self.logger.info('total_anchors: {}'.format(total_anchors))
            self.logger.info('inds_inside: {}'.format(num_inside))
            self.logger.info('anchors.shape: {}'.format(anchors.shape))
        if gt_classes is None:
            gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
        # Compute anchor labels:
        # label=1 is positive, 0 is negative, -1 is don't care (ignore)
        labels = np.empty((num_inside,), dtype=np.int32)
        gt_ids = np.empty((num_inside,), dtype=np.int32)
        labels.fill(-1)
        gt_ids.fill(-1)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # Compute overlaps between the anchors and the gt boxes overlaps
            anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
            # Map from anchor to gt box that has highest overlap
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
            # For each anchor, amount of overlap with most overlapping gt box
            anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                    anchor_to_gt_argmax]  #
            # Map from gt box to an anchor that has highest overlap
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
            # For each gt box, amount of overlap with most overlapping anchor
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax,
                np.arange(anchor_by_gt_overlap.shape[1])]
            # must remove gt which doesn't match any anchor.
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            anchors_with_max_overlap = np.where(
                anchor_by_gt_overlap == gt_to_anchor_max)[0]
            # Fg label: for each gt use anchors with highest overlap
            # (including ties)
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force
            # Fg label: above threshold IOU
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds]
            gt_ids[pos_inds] = gt_inds
            bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
        else:
            # labels[:] = 0
            bg_inds = np.arange(num_inside)
        fg_inds = np.where(labels > 0)[0]
        fg_max_overlap = None
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_max_overlap = anchor_to_gt_max[fg_inds]
        gt_pos_ids = gt_ids[fg_inds]
        # bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
        # bg_inds = np.where(labels == 0)[0]
        # subsample positive labels if we have too many
        if positive_fraction is not None:
            num_fg = int(positive_fraction * rpn_batch_size)
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1
                fg_inds = np.where(labels > 0)[0]

            # subsample negative labels if we have too many
            # (samples with replacement, but since the set of bg inds is large most
            # samples will not have repeats)
            num_bg = rpn_batch_size - np.sum(labels > 0)
            # print(num_fg, num_bg, len(bg_inds) )
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
                labels[enable_inds] = 0
            bg_inds = np.where(labels == 0)[0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                labels[bg_inds] = 0
                # re-enable anchors_with_max_overlap
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        bbox_targets = np.zeros(
            (num_inside, box_code_size), dtype=all_anchors.dtype)
        bbox_src_targets = np.zeros(
            (num_inside, box_code_size), dtype=all_anchors.dtype)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # print(anchors[fg_inds, :].shape, gt_boxes[anchor_to_gt_argmax[fg_inds], :].shape)
            # bbox_targets[fg_inds, :] = box_encoding_fn(
            #     anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = box_encoding_fn(fg_gt_boxes, fg_anchors)
            temp_src_gt_boxes = fg_gt_boxes.copy()
            temp_src_gt_boxes[:, 0:3] = fg_gt_boxes[:, 0:3] - fg_anchors[:, 0:3]
            bbox_src_targets[fg_inds, :] = temp_src_gt_boxes

        # Bbox regression loss has the form:
        #   loss(x) = weight_outside * L(weight_inside * x)
        # Inside weights allow us to set zero loss on an element-wise basis
        # Bbox regression is only trained on positive examples so we set their
        # weights to 1.0 (or otherwise if config is different) and 0 otherwise
        # NOTE: we don't need bbox_inside_weights, remove it.
        # bbox_inside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
        # bbox_inside_weights[labels == 1, :] = [1.0] * box_ndim

        # The bbox regression loss only averages by the number of images in the
        # mini-batch, whereas we need to average by the total number of example
        # anchors selected
        # Outside weights are used to scale each element-wise loss so the final
        # average over the mini-batch is correct
        # bbox_outside_weights = np.zeros((num_inside, box_ndim), dtype=np.float32)
        bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)
        # uniform weighting of examples (given non-uniform sampling)
        if norm_by_num_examples:
            num_examples = np.sum(labels >= 0)  # neg + pos
            num_examples = np.maximum(1.0, num_examples)
            bbox_outside_weights[labels > 0] = 1.0 / num_examples
        else:
            bbox_outside_weights[labels > 0] = 1.0
        # bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

        # Map up to original set of anchors
        if inds_inside is not None:
            labels = unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_src_targets = unmap(bbox_src_targets, total_anchors, inds_inside, fill=0)
            # bbox_inside_weights = unmap(
            #     bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = unmap(
                bbox_outside_weights, total_anchors, inds_inside, fill=0)
        # return labels, bbox_targets, bbox_outside_weights
        ret = {
            'labels': labels,
            'bbox_targets': bbox_targets,
            'bbox_outside_weights': bbox_outside_weights,
            'assigned_anchors_overlap': fg_max_overlap,
            'positive_gt_id': gt_pos_ids,
            'bbox_src_targets': bbox_src_targets,
        }
        if inds_inside is not None:
            ret['assigned_anchors_inds'] = inds_inside[fg_inds]
        else:
            ret['assigned_anchors_inds'] = fg_inds
        return ret

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self.anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    def num_anchors_per_location_class(self, class_name):
        if isinstance(class_name, int):
            class_name = self.classes[class_name]
        assert class_name in self.classes
        class_idx = self.classes.index(class_name)
        return self.anchor_generators[class_idx].num_anchors_per_localization

    @property
    def classes(self):
        return [a.class_name for a in self.anchor_generators]
