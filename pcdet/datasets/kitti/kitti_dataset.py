import os
import sys
import pickle
import copy
import numpy as np
from skimage import io
from pathlib import Path
import torch
import spconv

from pcdet.utils import box_utils, object3d_utils, calibration, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg
from pcdet.datasets.data_augmentation.dbsampler import DataBaseSampler
from pcdet.datasets import DatasetTemplate


class BaseKittiDataset(DatasetTemplate):
    def __init__(self, root_path, split='train'):
        super().__init__()
        self.root_path = root_path
        self.root_split_path = os.path.join(self.root_path, 'training' if split != 'test' else 'testing')
        self.split = split

        if split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.root_path, 'ImageSets', split + '.txt')

        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

    def set_split(self, split):
        self.__init__(self.root_path, split)

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_split_path, 'velodyne', '%s.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = os.path.join(self.root_split_path, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = os.path.join(self.root_split_path, 'label_2', '%s.txt' % idx)
        assert os.path.exists(label_file)
        return object3d_utils.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path, 'calib', '%s.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.root_split_path, 'planes', '%s.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        '''
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param img_shape:
        :return:
        '''
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc_lidar, w, l, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes3d_to_corners3d_lidar(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        # temp = process_single_scene(self.sample_id_list[0])
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dict(input_dict, index, record_dict):
        # finally generate predictions.
        sample_idx = input_dict['sample_idx'][index] if 'sample_idx' in input_dict else -1
        boxes3d_lidar_preds = record_dict['boxes'].cpu().numpy()

        if boxes3d_lidar_preds.shape[0] == 0:
            return {'sample_idx': sample_idx}

        calib = input_dict['calib'][index]
        image_shape = input_dict['image_shape'][index]

        boxes3d_camera_preds = box_utils.boxes3d_lidar_to_camera(boxes3d_lidar_preds, calib)
        boxes2d_image_preds = box_utils.boxes3d_camera_to_imageboxes(boxes3d_camera_preds, calib,
                                                                     image_shape=image_shape)
        # predictions
        predictions_dict = {
            'bbox': boxes2d_image_preds,
            'box3d_camera': boxes3d_camera_preds,
            'box3d_lidar': boxes3d_lidar_preds,
            'scores': record_dict['scores'].cpu().numpy(),
            'label_preds': record_dict['labels'].cpu().numpy(),
            'sample_idx': sample_idx,
        }
        return predictions_dict

    @staticmethod
    def generate_annotations(input_dict, pred_dicts, class_names, save_to_file=False, output_dir=None):
        def get_empty_prediction():
            ret_dict = {
                'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
                'boxes_lidar': np.zeros([0, 7])
            }
            return ret_dict

        def generate_single_anno(idx, box_dict):
            num_example = 0
            if 'bbox' not in box_dict:
                return get_empty_prediction(), num_example

            area_limit = image_shape = None
            if cfg.MODEL.TEST.BOX_FILTER['USE_IMAGE_AREA_FILTER']:
                image_shape = input_dict['image_shape'][idx]
                area_limit = image_shape[0] * image_shape[1] * 0.8

            sample_idx = box_dict['sample_idx']
            box_preds_image = box_dict['bbox']
            box_preds_camera = box_dict['box3d_camera']
            box_preds_lidar = box_dict['box3d_lidar']
            scores = box_dict['scores']
            label_preds = box_dict['label_preds']

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': [], 'boxes_lidar': []}

            for box_camera, box_lidar, bbox, score, label in zip(box_preds_camera, box_preds_lidar, box_preds_image,
                                                                 scores, label_preds):
                if area_limit is not None:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0] or bbox[2] < 0 or bbox[3] < 0:
                        continue
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > area_limit:
                        continue

                if 'LIMIT_RANGE' in cfg.MODEL.TEST.BOX_FILTER:
                    limit_range = np.array(cfg.MODEL.TEST.BOX_FILTER['LIMIT_RANGE'])
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(box_lidar[:3] > limit_range[3:]):
                        continue

                if not (np.all(box_lidar[3:6] > -0.1)):
                    print('Invalid size(sample %s): ' % str(sample_idx), box_lidar)
                    continue

                anno['name'].append(class_names[int(label - 1)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box_camera[6])
                anno['bbox'].append(bbox)
                anno['dimensions'].append(box_camera[3:6])
                anno['location'].append(box_camera[:3])
                anno['rotation_y'].append(box_camera[6])
                anno['score'].append(score)
                anno['boxes_lidar'].append(box_lidar)

                num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = get_empty_prediction()

            return anno, num_example

        annos = []
        for i, box_dict in enumerate(pred_dicts):
            sample_idx = box_dict['sample_idx']
            single_anno, num_example = generate_single_anno(i, box_dict)
            single_anno['num_example'] = num_example
            single_anno['sample_idx'] = np.array([sample_idx] * num_example, dtype=np.int64)
            annos.append(single_anno)
            if save_to_file:
                cur_det_file = os.path.join(output_dir, '%s.txt' % sample_idx)
                with open(cur_det_file, 'w') as f:
                    bbox = single_anno['bbox']
                    loc = single_anno['location']
                    dims = single_anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_anno['name'][idx], single_anno['alpha'][idx], bbox[idx][0], bbox[idx][1],
                                 bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_anno['rotation_y'][idx], single_anno['score'][idx]),
                              file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        assert 'annos' in self.kitti_infos[0].keys()
        import pcdet.datasets.kitti.kitti_object_eval_python.eval as kitti_eval

        if 'annos' not in self.kitti_infos[0]:
            return 'None', {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict


class KittiDataset(BaseKittiDataset):
    def __init__(self, root_path, class_names, split, training, logger=None):
        """
        :param root_path: KITTI data path
        :param split:
        """
        super().__init__(root_path=root_path, split=split)

        self.class_names = class_names
        self.training = training
        self.logger = logger

        self.mode = 'TRAIN' if self.training else 'TEST'

        self.kitti_infos = []
        self.include_kitti_data(self.mode, logger)
        # self.kitti_infos = self.kitti_infos[:100]
        self.dataset_init(class_names, logger)

    def include_kitti_data(self, mode, logger):
        if cfg.LOCAL_RANK == 0 and logger is not None:
            logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in cfg.DATA_CONFIG[mode].INFO_PATH:
            info_path = cfg.ROOT_DIR / info_path
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if cfg.LOCAL_RANK == 0 and logger is not None:
            logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def dataset_init(self, class_names, logger):
        self.db_sampler = None
        db_sampler_cfg = cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER
        if self.training and db_sampler_cfg.ENABLED:
            db_infos = []
            for db_info_path in db_sampler_cfg.DB_INFO_PATH:
                db_info_path = cfg.ROOT_DIR / db_info_path
                with open(str(db_info_path), 'rb') as f:
                    infos = pickle.load(f)
                    if db_infos.__len__() == 0:
                        db_infos = infos
                    else:
                        [db_infos[cls].extend(infos[cls]) for cls in db_infos.keys()]

            self.db_sampler = DataBaseSampler(
                db_infos=db_infos, sampler_cfg=db_sampler_cfg, class_names=class_names, logger=logger
            )

        voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR

        # Support spconv 1.0 and 1.1
        points = np.zeros((1, 3))
        try:
            self.voxel_generator = spconv.utils.VoxelGenerator(
                voxel_size=voxel_generator_cfg.VOXEL_SIZE,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
                max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
            )
            voxels, coordinates, num_points = self.voxel_generator.generate(points)
        except:
            self.voxel_generator = spconv.utils.VoxelGeneratorV2(
                voxel_size=voxel_generator_cfg.VOXEL_SIZE,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
                max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
            )
            voxel_grid = self.voxel_generator.generate(points)


    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            bbox = annos['bbox']
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            else:
                gt_boxes_lidar = box_utils.boxes3d_camera_to_lidar(gt_boxes, calib)

            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'gt_box2d': bbox,
                'gt_boxes_lidar': gt_boxes_lidar
            })

        example = self.prepare_data(input_dict=input_dict, has_label='annos' in info)

        example['sample_idx'] = sample_idx
        example['image_shape'] = img_shape

        return example


def create_kitti_infos(data_path, save_path, workers=4):
    dataset = BaseKittiDataset(root_path=data_path)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        create_kitti_infos(
            data_path=cfg.ROOT_DIR / 'data' / 'kitti',
            save_path=cfg.ROOT_DIR / 'data' / 'kitti'
        )
    else:
        A = KittiDataset(root_path='data/kitti', class_names=cfg.CLASS_NAMES, split='train', training=True)
        import pdb
        pdb.set_trace()
        ans = A[1]


