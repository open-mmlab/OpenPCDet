from functools import partial
import numpy as np
import copy

from ...utils import common_utils
from .ssl_database_sampler import SSLDataBaseSampler

class SSLDataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.aug_list = []
        self.augmentor_queue = []
        aug_config_list = augmentor_configs.AUG_CONFIG_LIST
        for cur_cfg in aug_config_list:
            if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.augmentor_queue.append(cur_augmentor)
            self.aug_list.append(cur_cfg.NAME)

    def gt_sampling(self, config=None):
        db_sampler = SSLDataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        params = []

        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None
        for cur_axis in config['ALONG_AXIS_LIST']:
            if cur_axis == 'x':
                enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                if enable:
                    points[:, 1] = -points[:, 1]
                    if 'gt_boxes' in data_dict:
                        gt_boxes[:, 1] = -gt_boxes[:, 1]
                        gt_boxes[:, 6] = -gt_boxes[:, 6]
                        if gt_boxes.shape[1] > 7:
                            gt_boxes[:, 8] = -gt_boxes[:, 8]
                    params.append('x')

            elif cur_axis == 'y':
                enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                if enable:
                    points[:, 0] = -points[:, 0]
                    if 'gt_boxes' in data_dict:
                        gt_boxes[:, 0] = -gt_boxes[:, 0]
                        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
                        if gt_boxes.shape[1] > 7:
                            gt_boxes[:, 7] = -gt_boxes[:, 7]
                    params.append('y')
            else:
                raise NotImplementedError

        data_dict['augmentation_params']['random_world_flip'] = params

        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        if 'gt_boxes' in data_dict:
            gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
            gt_boxes[:, 6] += noise_rotation
            if gt_boxes.shape[1] > 7:
                gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
                    np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                    np.array([noise_rotation])
                )[0][:, 0:2]

        data_dict['augmentation_params']['random_world_rotation'] = noise_rotation

        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        scale_range = config['WORLD_SCALE_RANGE']
        points = data_dict['points']
        gt_boxes = data_dict['gt_boxes'] if 'gt_boxes' in data_dict else None
        if scale_range[1] - scale_range[0] < 1e-3:
            noise_scale = 1
        else:
            noise_scale = np.random.uniform(scale_range[0], scale_range[1])
            points[:, :3] *= noise_scale
            if 'gt_boxes' in data_dict:
                gt_boxes[:, :6] *= noise_scale

        data_dict['augmentation_params']['random_world_scaling'] = noise_scale

        data_dict['points'] = points
        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict['augmentation_list'] = copy.deepcopy(self.aug_list)
        data_dict['augmentation_params'] = {}

        for cur_augmentor in self.augmentor_queue:
            data_dict = cur_augmentor(data_dict)

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')

        return data_dict