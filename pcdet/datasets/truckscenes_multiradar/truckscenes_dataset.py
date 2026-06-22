import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image
from truckscenes.utils.data_classes import RadarPointCloud, reduce


class TruckScenesMultiRadarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.include_truckscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_truckscenes_data(self, mode):
        self.logger.info('Loading TruckScenes dataset')
        truckscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                truckscenes_infos.extend(infos)

        self.infos.extend(truckscenes_infos)
        self.logger.info('Total samples for TruckScenes dataset: %d' % (len(truckscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T
    
    def get_merged_sweeps(self, index, max_sweeps=1):
        merged_points = []
        info = self.infos[index]

        # Get the neutral anchor matrices
        ref_from_car = info['ref_from_car']
        car_from_global = info['car_from_global']
        #print(info['radars'].items())

        for chan_name, chan_info in info['radars'].items():
            
            
            # Load the raw points for this radar
            points = self.get_lidar_with_sweeps(index, chan_name, max_sweeps) 

            lidar_from_radar = reduce(np.dot, [
                #ref_from_car, 
                car_from_global, 
                chan_info['global_from_radar_car'], 
                chan_info['car_from_radar']
            ])

            xyz_hom = np.hstack((points[:, 0:3], np.ones((points.shape[0], 1))))
            pts_lidar = xyz_hom @ lidar_from_radar.T
            points[:, 0:3] = pts_lidar[:, 0:3]

            rot_matrix = lidar_from_radar[0:3, 0:3]
            
            # Temporarily pad 2D velocity into 3D so the rotation matrix works
            v_3d = np.hstack((points[:, 4:6], np.zeros((points.shape[0], 1))))
            
            # Rotate the velocity vectors
            vel_lidar = v_3d @ rot_matrix.T
            
            # Put the newly rotated vx and vy back into their correct columns
            points[:, 4:6] = vel_lidar[:, 0:2]
            merged_points.append(points)

       
        final_point_cloud = np.concatenate(merged_points, axis=0)
        
        return final_point_cloud

    def get_lidar_with_sweeps(self, index, chan, max_sweeps=1):
        info = self.infos[index]['radars'][chan]
        lidar_path = self.root_path / info['lidar_path']

        pc = RadarPointCloud.from_file(str(lidar_path))
        points = pc.points.T

        v_ego_local = np.zeros(3)
        if len(info['sweeps']) > 0:
            v_x, v_y, v_z = self.compensate_ego_motion(points, info['sweeps'][0])

            points[:, 3] = v_x
            points[:, 4] = v_y
            points[:, 5] = v_z

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        if max_sweeps > 1 and len(info['sweeps']) > 0:
            #print(f"Sample {index} has {len(info['sweeps'])} sweeps, sampling up to {max_sweeps - 1} sweeps for augmentation.")
            num_sweeps = len(info['sweeps'])
            choices = np.random.choice(num_sweeps, max_sweeps - 1, replace=False)
            
            for k in choices:
                sweep = info['sweeps'][k]
                sweep_path = self.root_path / sweep['lidar_path']
                sweep_pc = RadarPointCloud.from_file(str(sweep_path))
                sweep_points = sweep_pc.points.T

                v_x, v_y, v_z = self.compensate_ego_motion(sweep_points, sweep)

                sweep_points[:, 3] = v_x
                sweep_points[:, 4] = v_y
                sweep_points[:, 5] = v_z

                if sweep['transform_matrix'] is not None:
                    sweep_points[:, :3] = sweep_points[:, :3] @ sweep['transform_matrix'][:3, :3].T
                    sweep_points[:, :3] += sweep['transform_matrix'][:3, 3]
                    sweep_points[:, 3:6] = sweep_points[:, 3:6] @ sweep['transform_matrix'][:3, :3].T

                sweep_points_list.append(sweep_points)
                sweep_times_list.append(sweep['time_lag'] * np.ones((sweep_points.shape[0], 1)))

        all_points = np.concatenate(sweep_points_list, axis=0)
        all_times = np.concatenate(sweep_times_list, axis=0)
        
        combined = np.concatenate((all_points, all_times), axis=1)
        final_points = combined[:, [0, 1, 2, 6, 3, 4, 7]]

        return final_points.astype(np.float32)
    
    def compensate_ego_motion(self, points, info_sweep):
        if info_sweep['transform_matrix'] is None or info_sweep['time_lag'] <= 0:
            print(f"Warning: No valid transform matrix or non-positive time lag for ego motion compensation. Returning original velocities. {info_sweep}")
            return points[:, 3], points[:, 4], points[:, 5]

        translation = info_sweep['transform_matrix'][:3, 3]
        v_ego_local = -translation / info_sweep['time_lag']
        #print(f"Ego velocity (local): {v_ego_local}, time lag: {info_sweep['time_lag']}")
        #print(f"Mean speed before compensation: {np.mean(points[:, 3])} {np.mean(points[:, 4])} {np.mean(points[:, 5])}")

        d = np.linalg.norm(points[:, :3], axis=1)
        ux = points[:, 0] / (d[:] + 1e-8)
        uy = points[:, 1] / (d[:] + 1e-8)
        uz = points[:, 2] / (d[:] + 1e-8)

        v_ego_local_points = ux * v_ego_local[0] + uy * v_ego_local[1] + uz * v_ego_local[2]
        v_points = ux * points[:, 3] + uy * points[:, 4] + uz * points[:, 5]
        v_comp = v_points + v_ego_local_points
        #print(f"Mean speed after compensation: {np.mean(ux * v_comp)} {np.mean(uy * v_comp)} {np.mean(uz * v_comp)}")

        return ux * v_comp, uy * v_comp, uz * v_comp

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_merged_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)


        #print(f"DEBUG - PATH : {info['lidar_path']}, POINTS SHAPE: {points.shape}")

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                import torch
                from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
                
                gt_boxes = info['gt_boxes']

                gt_vel = gt_boxes[:, 6:9] 

                if np.isnan(gt_vel).any() or np.isinf(gt_vel).any():
                    #print(f"Warning: Found NaN or Inf values in gt_boxes for sample {index}. Replacing with zeros.")
                    gt_boxes[:, 6:9]  = np.nan_to_num(gt_vel, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Filter out empty gt_boxes
                if len(gt_boxes) == 0:
                    mask = np.zeros(0, dtype=bool)
                else:
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
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from truckscenes.truckscenes import TruckScenes
        from . import truckscenes_utils
        nusc = TruckScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = truckscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': False,
            'use_radar': True,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of TruckScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.1-test':
            return 'No ground-truth annotations for evaluation', {}

        from truckscenes.eval.detection.config import config_factory
        from truckscenes.eval.detection.evaluate import TruckScenesEval

        eval_set_map = {
            'v1.1-mini': 'mini_val',
            'v1.1-trainval': 'val',
            'v1.1-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2024'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2024'
            eval_config = config_factory(eval_version)

        nusc_eval = TruckScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = truckscenes_utils.format_truckscenes_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'truckscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_merged_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_truckscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
    from truckscenes.truckscenes import TruckScenes
    from truckscenes.utils import splits
    from . import truckscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.1-trainval', 'v1.1-test', 'v1.1-mini']
    if version == 'v1.1-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.1-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.1-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = TruckScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = truckscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = truckscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
    )

    if version == 'v1.1-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'truckscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'truckscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'truckscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_truckscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.1-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()

    if args.func == 'create_truckscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_truckscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'truckscenes',
            save_path=ROOT_DIR / 'data' / 'truckscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            with_cam=args.with_cam
        )

        truckscenes_dataset = TruckScenesMultiRadarDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'truckscenes',
            logger=common_utils.create_logger(), training=True
        )
        truckscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
