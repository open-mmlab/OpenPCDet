import copy
import pickle
import argparse
import os
from os import path as osp
import torch
from av2.utils.io import read_feather
import numpy as np
import multiprocessing as mp
import pickle as pkl
from pathlib import Path
import pandas as pd

from ..dataset import DatasetTemplate
from .argo2_utils.so3 import yaw_to_quat, quat_to_yaw
from .argo2_utils.constants import LABEL_ATTR


def process_single_segment(segment_path, split, info_list, ts2idx, output_dir, save_bin):
    test_mode = 'test' in split
    if not test_mode:
        segment_anno = read_feather(Path(osp.join(segment_path, 'annotations.feather')))
    segname = segment_path.split('/')[-1]

    frame_path_list = os.listdir(osp.join(segment_path, 'sensors/lidar/'))

    for frame_name in frame_path_list:
        ts = int(osp.basename(frame_name).split('.')[0])

        if not test_mode:
            frame_anno = segment_anno[segment_anno['timestamp_ns'] == ts]
        else:
            frame_anno = None

        frame_path = osp.join(segment_path, 'sensors/lidar/', frame_name)
        frame_info = process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin)
        info_list.append(frame_info)


def process_and_save_frame(frame_path, frame_anno, ts2idx, segname, output_dir, save_bin):
    frame_info = {}
    frame_info['uuid'] = segname + '/' + frame_path.split('/')[-1].split('.')[0]
    frame_info['sample_idx'] = ts2idx[frame_info['uuid']]
    frame_info['image'] = dict()
    frame_info['point_cloud'] = dict(
        num_features=4,
        velodyne_path=None,
    )
    frame_info['calib'] = dict()  # not need for lidar-only
    frame_info['pose'] = dict()  # not need for single frame
    frame_info['annos'] = dict(
        name=None,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox=None,  # not need for lidar-only
        dimensions=None,
        location=None,
        rotation_y=None,
        index=None,
        group_ids=None,
        camera_id=None,
        difficulty=None,
        num_points_in_gt=None,
    )
    frame_info['sweeps'] = []  # not need for single frame
    if frame_anno is not None:
        frame_anno = frame_anno[frame_anno['num_interior_pts'] > 0]
        cuboid_params = frame_anno.loc[:, list(LABEL_ATTR)].to_numpy()
        cuboid_params = torch.from_numpy(cuboid_params)
        yaw = quat_to_yaw(cuboid_params[:, -4:])
        xyz = cuboid_params[:, :3]
        lwh = cuboid_params[:, [3, 4, 5]]

        cat = frame_anno['category'].to_numpy().tolist()
        cat = [c.lower().capitalize() for c in cat]
        cat = np.array(cat)

        num_obj = len(cat)

        annos = frame_info['annos']
        annos['name'] = cat
        annos['truncated'] = np.zeros(num_obj, dtype=np.float64)
        annos['occluded'] = np.zeros(num_obj, dtype=np.int64)
        annos['alpha'] = -10 * np.ones(num_obj, dtype=np.float64)
        annos['dimensions'] = lwh.numpy().astype(np.float64)
        annos['location'] = xyz.numpy().astype(np.float64)
        annos['rotation_y'] = yaw.numpy().astype(np.float64)
        annos['index'] = np.arange(num_obj, dtype=np.int32)
        annos['num_points_in_gt'] = frame_anno['num_interior_pts'].to_numpy().astype(np.int32)
    # frame_info['group_ids'] = np.arange(num_obj, dtype=np.int32)
    prefix2split = {'0': 'training', '1': 'training', '2': 'testing'}
    sample_idx = frame_info['sample_idx']
    split = prefix2split[sample_idx[0]]
    abs_save_path = osp.join(output_dir, split, 'velodyne', f'{sample_idx}.bin')
    rel_save_path = osp.join(split, 'velodyne', f'{sample_idx}.bin')
    frame_info['point_cloud']['velodyne_path'] = rel_save_path
    if save_bin:
        save_point_cloud(frame_path, abs_save_path)
    return frame_info


def save_point_cloud(frame_path, save_path):
    lidar = read_feather(Path(frame_path))
    lidar = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    lidar.tofile(save_path)


def prepare(root):
    ts2idx = {}
    ts_list = []
    bin_idx_list = []
    seg_path_list = []
    seg_split_list = []
    assert root.split('/')[-1] == 'sensor'
    # include test if you need it
    splits = ['train', 'val']  # , 'test']
    num_train_samples = 0
    num_val_samples = 0
    num_test_samples = 0

    # 0 for training, 1 for validation and 2 for testing.
    prefixes = [0, 1, ]  # 2]

    for i in range(len(splits)):
        split = splits[i]
        prefix = prefixes[i]
        split_root = osp.join(root, split)
        seg_file_list = os.listdir(split_root)
        print(f'num of {split} segments:', len(seg_file_list))
        for seg_idx, seg_name in enumerate(seg_file_list):
            seg_path = osp.join(split_root, seg_name)
            seg_path_list.append(seg_path)
            seg_split_list.append(split)
            assert seg_idx < 1000
            frame_path_list = os.listdir(osp.join(seg_path, 'sensors/lidar/'))
            for frame_idx, frame_path in enumerate(frame_path_list):
                assert frame_idx < 1000
                bin_idx = str(prefix) + str(seg_idx).zfill(3) + str(frame_idx).zfill(3)
                ts = frame_path.split('/')[-1].split('.')[0]
                ts = seg_name + '/' + ts  # ts is not unique, so add seg_name
                ts2idx[ts] = bin_idx
                ts_list.append(ts)
                bin_idx_list.append(bin_idx)
        if split == 'train':
            num_train_samples = len(ts_list)
        elif split == 'val':
            num_val_samples = len(ts_list) - num_train_samples
        else:
            num_test_samples = len(ts_list) - num_train_samples - num_val_samples
    # print three num samples
    print('num of train samples:', num_train_samples)
    print('num of val samples:', num_val_samples)
    print('num of test samples:', num_test_samples)

    assert len(ts_list) == len(set(ts_list))
    assert len(bin_idx_list) == len(set(bin_idx_list))
    return ts2idx, seg_path_list, seg_split_list

def create_argo2_infos(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, token, num_process):
    for seg_i, seg_path in enumerate(seg_path_list):
        if seg_i % num_process != token:
            continue
        print(f'processing segment: {seg_i}/{len(seg_path_list)}')
        split = seg_split_list[seg_i]
        process_single_segment(seg_path, split, info_list, ts2idx, output_dir, save_bin)


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

        self.argo2_infos = []
        self.include_argo2_data(self.mode)
        self.evaluate_range = dataset_cfg.get("EVALUATE_RANGE", 200.0)

    def include_argo2_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Argoverse2 dataset')
        argo2_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                argo2_infos.extend(infos)

        self.argo2_infos.extend(argo2_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Argo2 dataset: %d' % (len(argo2_infos)))

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
            return len(self.argo2_infos) * self.total_epochs

        return len(self.argo2_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.argo2_infos)

        info = copy.deepcopy(self.argo2_infos[index])

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

        assert len(self.argo2_infos) == len(outputs)
        num_samples = len(outputs)
        print('\nGot {} samples'.format(num_samples))

        serialized_dts_list = []

        print('\nConvert predictions to Argoverse 2 format')
        for i in range(num_samples):
            out_i = outputs[i]
            log_id, ts = self.argo2_infos[i]['uuid'].split('/')
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
        lwh = boxes[:, [3, 4, 5]]
        yaw = boxes[:, 6]

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
        """Evaluation in Argo2 protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'Argo2'.
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
        argo2_root = self.root_path
        val_anno_path = osp.join(argo2_root, 'val_anno.feather')
        gts = read_feather(Path(val_anno_path))
        gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")

        valid_uuids_gts = gts.index.tolist()
        valid_uuids_dts = dts.index.tolist()
        valid_uuids = set(valid_uuids_gts) & set(valid_uuids_dts)
        gts = gts.loc[list(valid_uuids)].sort_index()

        categories = set(x.value for x in CompetitionCategories)
        categories &= set(gts["category"].unique().tolist())

        dataset_dir = Path(argo2_root) / 'sensor' / 'val'
        cfg = DetectionCfg(
            dataset_dir=dataset_dir,
            categories=tuple(sorted(categories)),
            max_range_m=self.evaluate_range,
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

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, default="/data/argo2/sensor")
    parser.add_argument('--output_dir', type=str, default="/data/argo2/processed")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    root = args.root_path
    output_dir = args.output_dir
    save_bin = True
    ts2idx, seg_path_list, seg_split_list = prepare(root)

    velodyne_dir = Path(output_dir) / 'training' / 'velodyne'
    if not velodyne_dir.exists():
        velodyne_dir.mkdir(parents=True, exist_ok=True)

    info_list = []
    create_argo2_infos(seg_path_list, seg_split_list, info_list, ts2idx, output_dir, save_bin, 0, 1)

    assert len(info_list) > 0

    train_info = [e for e in info_list if e['sample_idx'][0] == '0']
    val_info = [e for e in info_list if e['sample_idx'][0] == '1']
    test_info = [e for e in info_list if e['sample_idx'][0] == '2']
    trainval_info = train_info + val_info
    assert len(train_info) + len(val_info) + len(test_info) == len(info_list)

    # save info_list in under the output_dir as pickle file
    with open(osp.join(output_dir, 'argo2_infos_train.pkl'), 'wb') as f:
        pkl.dump(train_info, f)

    with open(osp.join(output_dir, 'argo2_infos_val.pkl'), 'wb') as f:
        pkl.dump(val_info, f)

    # save validation anno feather
    save_feather_path = os.path.join(output_dir, 'val_anno.feather')
    val_seg_path_list = [seg_path for seg_path in seg_path_list if 'val' in seg_path]
    assert len(val_seg_path_list) == len([i for i in seg_split_list if i == 'val'])

    seg_anno_list = []
    for seg_path in val_seg_path_list:
        seg_anno = read_feather(osp.join(seg_path, 'annotations.feather'))
        log_id = seg_path.split('/')[-1]
        seg_anno["log_id"] = log_id
        seg_anno_list.append(seg_anno)

    gts = pd.concat(seg_anno_list).reset_index()
    gts.to_feather(save_feather_path)
