import copy
import pickle
import os
import json
from pypcd4 import PointCloud 
import torch

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from .point_processor_nuscenes import PointProcessorNuscenes


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.mode_with_labels = self.dataset_cfg.get('MODE_WITH_LABELS', ["train", "val", "test"])

        self.auxiliary_radar = self.dataset_cfg.get('AUXILIARY_RADAR', ["right", "left"])
        self.COMP_POINTS_MOTIONS = self.dataset_cfg.get('COMP_POINTS_MOTIONS', False)

        print(f"Front radar will be used for motion compensation: {self.COMP_POINTS_MOTIONS}")
        print(f"Auxiliary radars: {self.auxiliary_radar}")
        #self.auxiliary_radar = self.dataset_cfg.get('AUXILIARY_RADAR', [])
        self.auxiliary_radar_calib = {}
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.radar_infos = {}
        self.indexes_str = []

        self.include_data(self.mode)
        self.map_class_to_kitti = {"car": "Car", "bicycle": "Cyclist", "pedestrian": "Pedestrian", "Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", "Truck": "Truck", "Bus": "Bus", "Motorcycle": "Motorcycle", "Bicycle": "Bicycle", "Traffic_cone": "DontCare", "Barrier": "DontCare"}

        self.point_processor = PointProcessorNuscenes(
            radar_offset_tx=self.front_radar_calib['shift_from_odom_x'],
            radar_offset_ty=self.front_radar_calib['shift_from_odom_y'],
            radar_offset_yaw=self.front_radar_calib['shift_from_odom_yaw'],
            n_frames=self.dataset_cfg.MAX_SWEEPS,
            COMP_POINTS_MOTION=self.COMP_POINTS_MOTIONS
        )

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        with open(self.root_path / mode / 'metadata/calibration.json', 'r') as f:
            calib_data = json.load(f)
            self.front_radar_calib = calib_data['front']

            for radar in self.auxiliary_radar: 
                self.auxiliary_radar_calib[radar] = calib_data[radar]

        with open(self.root_path / mode / 'metadata/radar_front_linked_interpolated.json', 'r') as f:
            radar_front_data = json.load(f)
            self.radar_infos['front'] = radar_front_data
            for item in radar_front_data:
                key = list(item.keys())[0]
                self.indexes_str.append(key)

        for radar in self.auxiliary_radar:
            with open(self.root_path / mode / f'metadata/radar_{radar}_interpolated.json', 'r') as f:
                radar_data = json.load(f)
                self.radar_infos[radar] = radar_data

        if self.mode in self.mode_with_labels:
            self.labels_idx = []
            files = os.listdir(self.root_path / self.mode / 'label')
            for file in files:
                if file.endswith('.json'):
                    idx = file.split('.')[0]
                    self.labels_idx.append(idx)


        print(f"Front radar calibration: {self.front_radar_calib}")
        print(f"Auxiliary radar calibration: {self.auxiliary_radar_calib}")
        print(f"Front radar infos length: {len(self.radar_infos['front'])}")
        for radar in self.auxiliary_radar:
            print(f"{radar.capitalize()} radar infos length: {len(self.radar_infos[radar])}")

    def get_label(self, idx):
        label_file = self.root_path / self.mode / 'label' / ('%s.json' % str(idx).zfill(6))
        gt_boxes = []
        gt_names = []
        with open(label_file, 'r') as f:
            label_data = json.load(f)
            for item in label_data:
                gt_boxes.append([item['psr']['position']['x'], item['psr']['position']['y'], item['psr']['position']['z'], item['psr']['scale']['x'], item['psr']['scale']['y'], item['psr']['scale']['z'], item['psr']['rotation']['z']])
                gt_names.append(item['obj_type'])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_radar(self, idx, radar_name):
        radar_file = self.root_path / self.mode / 'radar' / radar_name / f'{str(idx).zfill(6)}.pcd'
        if not radar_file.exists():
            return None
        pcd = PointCloud.from_path(str(radar_file))
        point_features = pcd.numpy()
        #print(f"Loaded radar data from {radar_file}, shape: {point_features.shape}")
        #print(f"Loaded radar data from {radar_file}, shape: {point_features.shape}")
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / self.mode / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self.mode in self.mode_with_labels:
            return len(self.labels_idx)
        return len(self.indexes_str)

    def get_velocity(self, radar_name, idx_):
        key = list(self.radar_infos[radar_name][idx_].keys())[0]
        vx_front = self.radar_infos[radar_name][idx_][key]["v_linear_x"]
        vy_front = self.radar_infos[radar_name][idx_][key]["v_linear_y"]
        v_yaw_front = self.radar_infos[radar_name][idx_][key]["v_angular_z"]
        return vx_front, vy_front, v_yaw_front

    def get_aux_calib(self, radar_name):
        shift_from_front_x = self.auxiliary_radar_calib[radar_name]['shift_from_front_x']
        shift_from_front_y = self.auxiliary_radar_calib[radar_name]['shift_from_front_y']
        shift_from_front_z = self.auxiliary_radar_calib[radar_name]['shift_from_front_z']
        shift_from_front_roll = self.auxiliary_radar_calib[radar_name]['shift_from_front_roll']
        shift_from_front_pitch = self.auxiliary_radar_calib[radar_name]['shift_from_front_pitch']
        shift_from_front_yaw = self.auxiliary_radar_calib[radar_name]['shift_from_front_yaw']
        return shift_from_front_x, shift_from_front_y, shift_from_front_z, shift_from_front_roll, shift_from_front_pitch, shift_from_front_yaw


    def get_merged_sweep_radar(self, index):
        # print(f"Processing index: {index}")
        for i in range(self.dataset_cfg.MAX_SWEEPS, 0, -1):
            idx_ = index - (i - 1) 
            if 0 <= idx_ < len(self.indexes_str):
                points = self.get_radar(idx_, 'front')
                if points is None:
                    continue
                key = list(self.radar_infos['front'][idx_].keys())[0]
                timestamp = self.radar_infos['front'][idx_][key]["timestamp"]
                vx_front, vy_front, v_yaw_front = self.get_velocity('front', idx_)
                
                self.point_processor.add_timestamp(timestamp)

                self.point_processor.processPoints(points, vx_front, vy_front, v_yaw_front)

                for radar in self.auxiliary_radar:
                    # print(f"Processing auxiliary radar: {radar} for index: {idx_}")
                    idx_aux = self.radar_infos['front'][idx_][key][f"radar_{radar}"]
                    idx_aux = int(idx_aux)
                    points_aux = self.get_radar(idx_aux, radar)
                    if points_aux is None:
                        continue
                    # print(f"Loaded auxiliary radar data from {radar} radar, shape: {points_aux.shape}")
                    calib = self.auxiliary_radar_calib[radar]


                    key_aux = list(self.radar_infos[radar][idx_aux].keys())[0]
                    timestamp_aux = self.radar_infos[radar][idx_aux][key_aux]["timestamp"]
                    vx_aux, vy_aux, v_yaw_aux = self.get_velocity(radar, idx_aux)
                    # print(f"Auxiliary radar velocity for {radar} radar: vx={vx_aux:.2f} m/s, vy={vy_aux:.2f} m/s, v_yaw={v_yaw_aux:.2f} rad/s")
                    offset_x, offset_y, _, _, _, offset_yaw = self.get_aux_calib(radar)
                    self.point_processor.add_auxiliar_cloud(points_aux, timestamp_aux, offset_x, offset_y, offset_yaw, vx_aux, vy_aux, v_yaw_aux)
                
        return self.point_processor.multiframe_points

    def filter_gt_boxes(self, gt_boxes, gt_names, points):
        boxes_tensor = torch.from_numpy(gt_boxes[:, :7]).float().reshape(-1, 7)
        points_tensor = torch.from_numpy(points[:, :3]).float().reshape(-1, 3)
        
        try:
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(boxes_tensor, points_tensor).numpy()
        except AssertionError:
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor, boxes_tensor).numpy()
        
        if point_indices.ndim == 2:
            point_counts = point_indices.sum(axis=1)
        else:
            point_counts = np.zeros(gt_boxes.shape[0], dtype=np.int32)
            valid_indices = point_indices[point_indices != -1]
            unique_boxes, counts = np.unique(valid_indices, return_counts=True)
            point_counts[unique_boxes] = counts
        
        mask = point_counts >= self.dataset_cfg.FILTER_MIN_POINTS_IN_GT
        return gt_boxes[mask], gt_names[mask]

    def __getitem__(self, index):
        if self.mode in self.mode_with_labels:
            index = int(self.labels_idx[index])

        points = self.get_merged_sweep_radar(index)


        input_dict = {
            'frame_id': str(index).zfill(6),
            'points': points
        }

        if self.mode in self.mode_with_labels:
            gt_boxes, gt_names = self.get_label(index)
            # print(f"Loaded GT boxes for index {index}: {gt_boxes}")

            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                # print(f"Filtering {gt_boxes.shape[0]} GT boxes for index {index} with minimum points in GT: {self.dataset_cfg.FILTER_MIN_POINTS_IN_GT}")
                gt_boxes, gt_names = self.filter_gt_boxes(gt_boxes, gt_names, points)
                # print(f"After filtering, {gt_boxes.shape[0]} GT boxes remain for index {index}.")

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        
        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = []
        for i in range(self.__len__()):
            gt_boxes, gt_names = self.get_label(self.labels_idx[i])
            points = self.get_merged_sweep_radar(i)
            gt_boxes, gt_names = self.filter_gt_boxes(gt_boxes, gt_names, points)

            data_dict = {
                'frame_id': str(self.labels_idx[i]).zfill(6),
                'gt_names': gt_names,
                'gt_boxes_lidar': gt_boxes
            }
            eval_gt_annos.append(data_dict)

        

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    # NOT USED
    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            # print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    # NOT USED
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_radar(sample_idx, 'front')
            annos = info['annos']
            names = annos['name']
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
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)

# NOT USED
def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )
