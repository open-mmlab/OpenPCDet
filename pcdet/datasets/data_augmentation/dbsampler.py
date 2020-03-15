# This file is modified from https://github.com/traveller59/second.pytorch

import numpy as np
import copy
import os
from ...utils import common_utils, box_utils
from . import augmentation_utils


class BatchSampler:
    def __init__(self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class DataBaseSampler(object):
    def __init__(self, db_infos, sampler_cfg, class_names, logger=None):
        super().__init__()

        if logger is not None:
            for k, v in db_infos.items():
                logger.info('Database before filter %s: %d' % (k, len(v)))
        for prep_func, val in sampler_cfg.PREPARE.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        if logger is not None:
            for k, v in db_infos.items():
                logger.info('Database after filter %s: %d' % (k, len(v)))

        self.db_infos = db_infos
        self.rate = sampler_cfg.RATE
        self.sample_groups = []
        for x in sampler_cfg.SAMPLE_GROUPS:
            name, num = x.split(':')
            if name not in class_names:
                continue
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    @staticmethod
    def filter_by_frontview(db_infos, front_dist_list):
        for name_num in front_dist_list:
            name, front_dist = name_num.split(':')
            filtered_infos = []
            for info in db_infos[name]:
                if info['box3d_lidar'][0] >= 0:
                    filtered_infos.append(info)
            db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, root_path, gt_boxes, gt_names, num_point_features=4,
                   road_planes=None, calib=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes, self.sample_max_nums):
            sampled_num = int(max_sample_num - np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num, avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack([s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_boxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            center = sampled_gt_boxes[:, 0:3]

            # road_planes = None
            if road_planes is not None:
                # image plane
                a, b, c, d = road_planes
                center_cam = calib.lidar_to_rect(center)
                cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
                center_cam[:, 1] = cur_height_cam
                lidar_tmp_point = calib.rect_to_lidar(center_cam)
                cur_lidar_height = lidar_tmp_point[:, 2]
                mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                sampled_gt_boxes[:, 2] -= mv_height  # lidar view

            num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(root_path, info['path'])
                s_points = np.fromfile(file_path, dtype=np.float32).reshape([-1, num_point_features])

                if 'rot_transform' in info:
                    rot = info['rot_transform']
                    s_points = common_utils.rotate_pc_along_z(s_points, rot)
                s_points[:, :3] += info['box3d_lidar'][:3]

                if road_planes is not None:
                    # mv height
                    s_points[:, 2] -= mv_height[count]
                count += 1

                s_points_list.append(s_points)

            ret = {'gt_names': np.array([s['name'] for s in sampled]),
                   'difficulty': np.array([s['difficulty'] for s in sampled]), 'gt_boxes': sampled_gt_boxes,
                   'points': np.concatenate(s_points_list, axis=0), 'gt_masks': np.ones((num_sampled,), dtype=np.bool_),
                   'group_ids': np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled))}

        return ret

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = box_utils.boxes3d_to_corners3d_lidar(gt_boxes)[:, 0:4, 0:2]  # (N, 4, 2)

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = box_utils.boxes3d_to_corners3d_lidar(sp_boxes_new)[:, 0:4, 0:2]  # (N, 4, 2)

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)
        coll_mat = augmentation_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
