import copy
import pickle
import torch
import numpy as np

from ..dataset import DatasetTemplate
from .argo2_utils.so3 import yaw_to_quat
from .argo2_utils.constants import LABEL_ATTR
from os import path as osp
from pathlib import Path


class Argo2Dataset(DatasetTemplate):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Argoverse2 dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Argo2 dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
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
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes_img = pred_boxes
            pred_boxes_camera = pred_boxes

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin')
        calib = None
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_bboxes_3d
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def format_results(self,
                       outputs,
                       class_names,
                       pklfile_prefix=None,
                       submission_prefix=None,
                       ):
        """Format the results to .feather file with argo2 format.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        import pandas as pd

        assert len(self.kitti_infos) == len(outputs)
        num_samples = len(outputs)
        print('\nGot {} samples'.format(num_samples))
        
        serialized_dts_list = []
        
        print('\nConvert predictions to Argoverse 2 format')
        for i in range(num_samples):
            out_i = outputs[i]
            log_id, ts = self.kitti_infos[i]['uuid'].split('/')
            track_uuid = None
            #cat_id = out_i['labels_3d'].numpy().tolist()
            #category = [class_names[i].upper() for i in cat_id]
            category = [class_name.upper() for class_name in out_i['name']]
            serialized_dts = pd.DataFrame(
                self.lidar_box_to_argo2(out_i['bbox']).numpy(), columns=list(LABEL_ATTR)
            )
            serialized_dts["score"] = out_i['score']
            serialized_dts["log_id"] = log_id
            serialized_dts["timestamp_ns"] = int(ts)
            serialized_dts["category"] = category
            serialized_dts_list.append(serialized_dts)
        
        dts = (
            pd.concat(serialized_dts_list)
            .set_index(["log_id", "timestamp_ns"])
            .sort_index()
        )

        dts = dts.sort_values("score", ascending=False).reset_index()

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.feather')):
                pklfile_prefix = f'{pklfile_prefix}.feather'
            dts.to_feather(pklfile_prefix)
            print(f'Result is saved to {pklfile_prefix}.')

        dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()

        return dts 
    
    def lidar_box_to_argo2(self, boxes):
        boxes = torch.Tensor(boxes)
        cnt_xyz = boxes[:, :3]
        #cnt_xyz[:, 2] += boxes[:, 5] * 0.5
        lwh = boxes[:, [4, 3, 5]]
        #yaw = -boxes[:, 6] - np.pi/2
        yaw = boxes[:, 6] #- np.pi/2

        yaw = -yaw - 0.5 * np.pi
        while (yaw < -np.pi).any():
            yaw[yaw < -np.pi] += 2 * np.pi
        while (yaw > np.pi).any():
            yaw[yaw > np.pi] -= 2 * np.pi

        quat = yaw_to_quat(yaw)
        argo_cuboid = torch.cat([cnt_xyz, lwh, quat], dim=1)
        return argo_cuboid

    def evaluation(self,
                 results,
                 class_names,
                 eval_metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 output_path=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        from av2.evaluation.detection.constants import CompetitionCategories
        from av2.evaluation.detection.utils import DetectionCfg
        from av2.evaluation.detection.eval import evaluate
        from av2.utils.io import read_feather

        dts = self.format_results(results, class_names, pklfile_prefix, submission_prefix)
        argo2_root = "../data/argo2/"
        val_anno_path = osp.join(argo2_root, 'val_anno.feather')
        gts = read_feather(val_anno_path)
        gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

        valid_uuids_gts = gts.index.tolist()
        valid_uuids_dts = dts.index.tolist()
        valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
        gts = gts.loc[list(valid_uuids)].sort_index()

        categories = set(x.value for x in CompetitionCategories)
        categories &= set(gts["category"].unique().tolist())

        split = 'val'
        dataset_dir = Path(argo2_root) / 'sensor' / split
        cfg = DetectionCfg(
            dataset_dir=dataset_dir,
            categories=tuple(sorted(categories)),
            #split=split,
            max_range_m=200.0,
            eval_only_roi_instances=True,
        )

        # Evaluate using Argoverse detection API.
        eval_dts, eval_gts, metrics = evaluate(
            dts.reset_index(), gts.reset_index(), cfg
        )

        valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
        ap_dict = {}
        for index, row in metrics.iterrows():
            ap_dict[index] = row.to_json()
        return metrics.loc[valid_categories], ap_dict
