import copy
import pickle
import numpy as np
from pathlib import Path
from ..semi_dataset import SemiDatasetTemplate
from .once_toolkits import Octopus

def split_once_semi_data(info_paths, data_splits, root_path, labeled_ratio, logger):
    once_pretrain_infos = []
    once_test_infos = []
    once_labeled_infos = []
    once_unlabeled_infos = []

    def check_annos(info):
        return 'annos' in info

    root_path = Path(root_path)

    train_split = data_splits['train']
    for info_path in info_paths[train_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            infos = list(filter(check_annos, infos))
            once_pretrain_infos.extend(copy.deepcopy(infos))
            once_labeled_infos.extend(copy.deepcopy(infos))

    test_split = data_splits['test']
    for info_path in info_paths[test_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            infos = list(filter(check_annos, infos))
            once_test_infos.extend(copy.deepcopy(infos))

    raw_split = data_splits['raw']
    for info_path in info_paths[raw_split]:
        info_path = root_path / info_path
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            once_unlabeled_infos.extend(copy.deepcopy(infos))

    logger.info('Total samples for ONCE pre-training dataset: %d' % (len(once_pretrain_infos)))
    logger.info('Total samples for ONCE testing dataset: %d' % (len(once_test_infos)))
    logger.info('Total samples for ONCE labeled dataset: %d' % (len(once_labeled_infos)))
    logger.info('Total samples for ONCE unlabeled dataset: %d' % (len(once_unlabeled_infos)))

    return once_pretrain_infos, once_test_infos, once_labeled_infos, once_unlabeled_infos

class ONCESemiDataset(SemiDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
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
        self.cam_names = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        self.cam_tags = ['top', 'top2', 'left_back', 'left_front', 'right_front', 'right_back', 'back']
        self.toolkits = Octopus(self.root_path)

        self.once_infos = infos

    def get_lidar(self, sequence_id, frame_id):
        return self.toolkits.load_point_cloud(sequence_id, frame_id)

    def get_image(self, sequence_id, frame_id, cam_name):
        return self.toolkits.load_image(sequence_id, frame_id, cam_name)

    def project_lidar_to_image(self, sequence_id, frame_id):
        return self.toolkits.project_lidar_to_image(sequence_id, frame_id)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_infos) * self.total_epochs

        return len(self.once_infos)

    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
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
            pred_dict['boxes_3d'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .once_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.once_infos]
        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)
        """
        eval_det_annos = copy.deepcopy(eval_gt_annos)
        for gt_anno in eval_det_annos:
            gt_anno['score'] = np.random.uniform(low=0.1, high=1,size=gt_anno['name'].shape[0])
            #gt_anno['score'] = np.ones(gt_anno['name'].shape[0])
            gt_anno['boxes_3d'][:, 0] += 0#np.random.uniform(low=0.1, high=1,size=gt_anno['name'].shape[0]) * 0.001
            gt_anno['boxes_3d'][:, 1] += 0#np.random.uniform(low=0.1, high=1,size=gt_anno['name'].shape[0]) * 0.001
        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)
        """
        return ap_result_str, ap_dict

class ONCEPretrainDataset(ONCESemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict

class ONCELabeledDataset(ONCESemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        assert 'annos' in info
        annos = info['annos']
        input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
        })

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)
        if teacher_dict is not None: teacher_dict.pop('num_points_in_gt', None)
        if student_dict is not None: student_dict.pop('num_points_in_gt', None)
        return tuple([teacher_dict, student_dict])

class ONCEUnlabeledDataset(ONCESemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.unlabeled_data_for)
        return tuple([teacher_dict, student_dict])

class ONCETestDataset(ONCESemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is False
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict