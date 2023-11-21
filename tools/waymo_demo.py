import argparse
import glob
import json
from pathlib import Path
import os

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset

NUMBER_OF_SCENES = 100


class DemoDataset(DatasetTemplate):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points[:, :5],
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main(model_path, cfg_path, tag):
    cfg_from_yaml_file(cfg_path, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    dataset = WaymoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )
    
    logger.info(f'Total number of samples: \t{len(dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=model_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vis_path = '/'.join(os.path.normpath(model_path).split(os.path.sep)[:-2]) + '/visualization/' + tag
    os.makedirs(vis_path, exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            if idx % 100 != 0:
                continue
            if idx  >= NUMBER_OF_SCENES * 100:
                break
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            frame_id = str(data_dict['frame_id'][0])

            torch.save(data_dict['points'][:,1:], os.path.join(vis_path, 'points_{}.pt'.format(frame_id)))
            torch.save(pred_dicts[0]['pred_boxes'], os.path.join(vis_path, 'pred_boxes_{}.pt'.format(frame_id)))
            torch.save(pred_dicts[0]['pred_scores'], os.path.join(vis_path, 'pred_scores_{}.pt'.format(frame_id)))
            torch.save(pred_dicts[0]['pred_labels'], os.path.join(vis_path, 'pred_labels_{}.pt'.format(frame_id)))
            torch.save(data_dict['gt_boxes'], os.path.join(vis_path, 'gt_boxes_{}.pt'.format(frame_id)))
            if 'gnn_edges_final' in pred_dicts[0]:
                torch.save(pred_dicts[0]['gnn_edges_final'],os.path.join(vis_path, 'gnn_edges{}.pt'.format(frame_id)))
                json.dump(pred_dicts[0]['edge_to_pred'] , open(os.path.join(vis_path, 'edge_to_predict{}.json'.format(frame_id)), 'w'))

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    
    model_path = "../output/cfgs/waymo_models/pv_rcnn/2023-10-19_12-02-51/ckpt/checkpoint_epoch_30.pth"
    cfg_path = "./cfgs/waymo_models/pv_rcnn.yaml"
    tag = "epoch_30"

    # model_path = "../output/cfgs/waymo_models/pv_rcnn_relation/2023-10-24_09-25-41/ckpt/checkpoint_epoch_25.pth"
    # cfg_path = "./cfgs/waymo_models/pv_rcnn_relation.yaml" 
    # tag = "epoch_25"
    main(model_path, cfg_path, tag)
