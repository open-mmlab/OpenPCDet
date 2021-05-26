import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, object3d_neolix, box_utils
# from visual_utils import visualize_utils as V



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', label_path=None):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.sample_label_list = None
        if label_path is not None:
            self.label_ext = '.txt'
            data_label_list = glob.glob(str(label_path / f'*{self.label_ext}')) if label_path.is_dir() else [label_path]
            data_label_list.sort()
            self.sample_label_list = data_label_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            rs_ls = []
            with open(self.sample_file_list[index], 'r') as f:
                raw_rs = f.readlines()[11:]
            for p_line in raw_rs:
                p_ls = p_line.strip().split(" ")
                if len(p_ls) == 4:
                    if (p_ls[0] != "nan") or (p_ls[1] != "nan") or (p_ls[2] != "nan") or (p_ls[3] != "nan"):
                        # rs_ls.append([float(i) for i in p_ls[:3]] + [float(p_ls[3]) / 255])
                        rs_ls.append([float(i) for i in p_line.strip().split(" ")])
            points = np.array(rs_ls)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': self.sample_file_list[index].split('/')[-1].strip('.bin'),
            'gt_boxes': None
        }
        self.sample_label_list = None
        if self.sample_label_list is not None:
            obj_list = object3d_neolix.get_objects_from_label(self.sample_label_list[index])
            loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            dims = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])
            rots = np.array([obj.ry for obj in obj_list])
            h = dims[:, 2:3]
            loc[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc, dims, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            gt_names = np.array([obj.cls_type for obj in obj_list])

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


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

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index]
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )
            pred_boxes_camera = box_utils.boxes3d_lidar_to_neolix_camera(pred_boxes)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = np.zeros([pred_boxes.shape[0], 4])
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            class_dict = {'Vehicle': 0, 'Large_vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Bicycle': 4,
                          'Unknown_movable':5 , 'Unknown_unmovable': 6}
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if len(single_pred_dict['name']) == 0:
                continue

            if output_path is not None:
                label_path = output_path
                track_format = False
                if track_format:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    content = []
                    for idx in range(len(bbox)):
                        type_name = class_dict[single_pred_dict['name'][idx]]
                        prob_ls = list(np.zeros(7))
                        prob_ls[type_name] = single_pred_dict['score'][idx]

                        content.append([loc[idx][0], loc[idx][1], loc[idx][2], dims[idx][2],
                                        dims[idx][0], dims[idx][1], -np.pi / 2 - single_pred_dict['rotation_y'][idx],
                                        type_name] + prob_ls)
                    np.array(content).astype(np.float32).tofile(label_path + batch_dict['frame_id'][index] + '.bin')
                else:
                    output_path.mkdir(parents=True,exist_ok=True)
                    print('output', output_path)
                    cur_det_file = output_path / (frame_id + '.txt')
                    with open(cur_det_file, 'w') as f:
                        print(cur_det_file)
                        bbox = single_pred_dict['bbox']
                        loc = single_pred_dict['location']
                        dims = single_pred_dict['dimensions']  # lhw -> hwl
                        for idx in range(len(bbox)):
                            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                                  % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                     bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                     dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                     loc[idx][1], loc[idx][2]-dims[idx][1]/2, single_pred_dict['rotation_y'][idx],
                                     single_pred_dict['score'][idx]), file=f)

        return annos



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--label_path', type=str, default=None,
                        help='specify the point cloud data label or directory')
    parser.add_argument('--output_path', type=str, default=None,
                        help='the output path of inference')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # import pdb; pdb.set_trace()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, label_path=Path(args.label_path) if args.label_path else None
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    ## add by shl
    # dataset = demo_dataset.dataset
    # class_names = dataset.class_names
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            # pred_dicts, _ = model.forward(data_dict)
            with torch.no_grad():
                pred_dicts, ret_dict = model(data_dict)
            annos = demo_dataset.generate_prediction_dicts(
                data_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=Path(args.output_path)
            )
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0][:, :-1], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
