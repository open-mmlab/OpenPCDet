import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils
from ..dataset import DatasetTemplate


class LyftDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=self.root_path, logger=logger
        )
        self.infos = []
        self.include_lyft_data(self.mode)

    def include_lyft_data(self, mode):
        self.logger.info('Loading lyft dataset')
        lyft_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                lyft_infos.extend(infos)

        self.infos.extend(lyft_infos)
        self.logger.info('Total samples for lyft dataset: %d' % (len(lyft_infos)))

    @staticmethod
    def remove_ego_points(points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius*1.5) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]

    def get_sweep(self, sweep_info):
        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        if points_sweep.shape[0] % 5 != 0:
            points_sweep = points_sweep[: points_sweep.shape[0] - (points_sweep.shape[0] % 5)]
        points_sweep = points_sweep.reshape([-1, 5])[:, :4]

        points_sweep = self.remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1)
        if points.shape[0] % 5 != 0:
            points = points[: points.shape[0] - (points.shape[0] % 5)]
        points = points.reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            input_dict.update({
                'gt_boxes': info['gt_boxes'],
                'gt_names': info['gt_names']
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        from ..kitti import kitti_utils

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
            'bicycle': 'Cyclist',
            'motorcycle': 'Cyclist'
        }

        kitti_utils.transform_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        )

        kitti_class_names = [map_name_to_kitti[x] for x in class_names]

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.infos)
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'lyft':
            return self.lyft_eval(det_annos, class_names, 
                                  iou_thresholds=self.dataset_cfg.EVAL_LYFT_IOU_LIST)
        else:
            raise NotImplementedError
    
    def lyft_eval(self, det_annos, class_names, iou_thresholds=[0.5]):
        from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
        from . import lyft_utils
        # from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions
        from .lyft_mAP_eval.lyft_eval import get_average_precisions

        lyft = Lyft(json_path=self.root_path / 'data', data_path=self.root_path, verbose=True)

        det_lyft_boxes, sample_tokens = lyft_utils.convert_det_to_lyft_format(lyft, det_annos)
        gt_lyft_boxes = lyft_utils.load_lyft_gt_by_tokens(lyft, sample_tokens)

        average_precisions = get_average_precisions(gt_lyft_boxes, det_lyft_boxes, class_names, iou_thresholds)

        ap_result_str, ap_dict = lyft_utils.format_lyft_results(average_precisions, class_names, iou_thresholds, version=self.dataset_cfg.VERSION)

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database'
        db_info_save_path = self.root_path / f'lyft_dbinfos_{max_sweeps}sweeps.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
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


def create_lyft_info(version, data_path, save_path, split, max_sweeps=10):
    from lyft_dataset_sdk.lyftdataset import LyftDataset
    from . import lyft_utils
    data_path = data_path / version
    save_path = save_path / version
    split_path = data_path.parent / 'ImageSets'

    if split is not None:
        save_path = save_path / split
        split_path = split_path / split

    save_path.mkdir(exist_ok=True)

    assert version in ['trainval', 'one_scene', 'test']

    if version == 'trainval':
        train_split_path = split_path / 'train.txt'
        val_split_path = split_path / 'val.txt'
    elif version == 'test':
        train_split_path = split_path / 'test.txt'
        val_split_path = None
    elif version == 'one_scene':
        train_split_path = split_path / 'one_scene.txt'
        val_split_path = split_path / 'one_scene.txt'
    else:
        raise NotImplementedError

    train_scenes = [x.strip() for x in open(train_split_path).readlines()] if train_split_path.exists() else []
    val_scenes = [x.strip() for x in open(val_split_path).readlines()] if val_split_path is not None and val_split_path.exists() else []

    lyft = LyftDataset(json_path=data_path / 'data', data_path=data_path, verbose=True)

    available_scenes = lyft_utils.get_available_scenes(lyft)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_lyft_infos, val_lyft_infos = lyft_utils.fill_trainval_infos(
        data_path=data_path, lyft=lyft, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'test':
        print('test sample: %d' % len(train_lyft_infos))
        with open(save_path / f'lyft_infos_test.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_lyft_infos), len(val_lyft_infos)))
        with open(save_path / f'lyft_infos_train.pkl', 'wb') as f:
            pickle.dump(train_lyft_infos, f)
        with open(save_path / f'lyft_infos_val.pkl', 'wb') as f:
            pickle.dump(val_lyft_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_lyft_infos', help='')
    parser.add_argument('--version', type=str, default='trainval', help='')
    parser.add_argument('--split', type=str, default=None, help='')
    parser.add_argument('--max_sweeps', type=int, default=10, help='')
    args = parser.parse_args()

    if args.func == 'create_lyft_infos':
        try:
            yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(open(args.cfg_file))
        dataset_cfg = EasyDict(yaml_config)
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        dataset_cfg.MAX_SWEEPS = args.max_sweeps 
        create_lyft_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'lyft',
            save_path=ROOT_DIR / 'data' / 'lyft',
            split=args.split,
            max_sweeps=dataset_cfg.MAX_SWEEPS
        )

        lyft_dataset = LyftDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'lyft',
            logger=common_utils.create_logger(), training=True
        )

        if args.version != 'test':
            lyft_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
