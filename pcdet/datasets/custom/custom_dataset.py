import copy
import pickle
import os

import numpy as np
from skimage import io

from . import custom_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils, object3d_custom
from ..dataset import DatasetTemplate

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.root_split_path = os.path.join(self.root_path, ('training' if self.split != 'test' else 'testing'))

        split_dir = os.path.join(self.root_path, 'ImageSets',(self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_custom_data(self.mode)
        self.ext = ext


    def include_custom_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        
        self.custom_infos.extend(custom_infos)

        if self.logger is not None:
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))
    

    def get_infos(self, num_workers=16, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        # Process single scene
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # no images, calibs are need to transform the labels

            type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list]) # 1-dimension
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])             
                annotations['location'] = np.concatenate([obj.loc.reshape(1,3) for obj in obj_list])
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list]) # 1-dimension

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = self.get_calib(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, (np.pi / 2 - rots[..., np.newaxis])], axis=1) # 2-dimension array
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                
                info['annos'] = annotations
            
            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)

        return list(infos)
                

    def get_calib(self, loc):
        """
        This calibration is different from the kitti dataset.
        The transform formual of labelCloud: ROOT/labelCloud/io/labels/kitti.py: import labels
            if self.transformed:
                centroid = centroid[2], -centroid[0], centroid[1] - 2.3
            dimensions = [float(v) for v in line_elements[8:11]]
            if self.transformed:
                dimensions = dimensions[2], dimensions[1], dimensions[0]
            bbox = BBox(*centroid, *dimensions)
        """
        loc_lidar = np.concatenate([np.array((float(loc_obj[2]), float(-loc_obj[0]), float(loc_obj[1]-2.3)), dtype=np.float32).reshape(1,3) for loc_obj in loc])
        return loc_lidar
                

    def get_label(self, idx):

        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_custom.get_objects_from_label(label_file)


    def get_lidar(self, idx, getitem):
        """
            Loads point clouds for a sample
                Args:
                    index (int): Index of the point cloud file to get.
                Returns:
                    np.array(N, 4): point cloud.
        """
        # get lidar statistics
        if getitem == True:
            lidar_file = self.root_split_path + '/velodyne/' + ('%s.bin' % idx)
        else:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None


    # Create gt database for data augmentation
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # For each .bin file
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx, False)
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
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N,7), Tensor
                pred_scores: (N), Tensor
                pred_lables: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_smaples):
            ret_dict = {
                'name': np.zeros(num_smaples), 'alpha' : np.zeros(num_smaples),
                'dimensions': np.zeros([num_smaples, 3]), 'location': np.zeros([num_smaples, 3]),
                'rotation_y': np.zero(num_smaples), 'score': np.zeros(num_smaples),
                'boxes_lidar': np.zeros([num_smaples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            # Define an empty template dict to store the prediction information, 'pred_scores.shape[0]' means 'num_samples'
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # If num_samples equals zero then return the empty dict
            if pred_scores.shape[0] == 0:
                return pred_dict

            # No calibration files

            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera[pred_boxes]

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
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

            # Output pred results to Output-path in .txt file 
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl: lidar -> camera

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
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)


    def __getitem__(self, index):
        """
        Function:
            Read 'velodyne' folder as pointclouds
            Read 'label_2' folder as labels
            Return type 'dict'
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)
        
        info = copy.deepcopy(self.custom_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': self.sample_id_list[index],
        }

        """
        Here infos was generated by get_infos
        """
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
        
        if 'points' in get_item_list:
            points = self.get_lidar(sample_idx, True)
            input_dict['points'] = points
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    # No evaluation
    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filenmae = save_path / ('custom_infos%s.pkl' % val_split)
    trainval_filename = save_path / 'custom_infos_trainval.pkl'
    test_filename = save_path / 'custom_infos_test.pkl'

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is save to %s' % train_filename)

    dataset.set_split('test')
    custom_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(custom_infos_test, f)
    print('Custom info test file is saved to %s' % test_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')

if __name__=='__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom'
        )
