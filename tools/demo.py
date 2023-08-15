import argparse
import glob
from pathlib import Path
import time
import os

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


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
        print(data_file_list)
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
            'points': points,
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


def main():
    args, cfg = parse_config()
    #fig = mlab.figure(size=(2560,1500))
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("output.txt", "w") as file:
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)

                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

                for i,pred_dict in enumerate(pred_dicts):
                    boxes = pred_dict['pred_boxes'].detach().cpu().numpy()
                    scores = pred_dict['pred_scores'].detach().cpu().numpy()
                    labels = pred_dict['pred_labels'].detach().cpu().numpy()
                file.write(f"Frame: {idx+1}\n")
                with open(output_dir / f"frame_{idx+1}.txt", 'w') as output_file:
                    for box, score, label in zip(boxes, scores, labels):
                            adjusted_label = label - 1
                            if adjusted_label < len(cfg.CLASS_NAMES):  
                                output_file.write(
                                    f'Label Index: {adjusted_label}, '
                                    f'Class Name: {cfg.CLASS_NAMES[adjusted_label]}\n'
                                    f"Bounding box: {box}, "
                                    f"Class: {cfg.CLASS_NAMES[adjusted_label]}, "
                                    f"Score: {score}\n"
                                )
                            else:
                                output_file.write(
                                    f"Label index {adjusted_label} out of range. "
                                    f"Maximum index is {len(cfg.CLASS_NAMES) - 1}\n"
                                )


                        
                max_label = np.max(labels)
                if max_label >= len(cfg.CLASS_NAMES):
                    file.write(f"Max label ({max_label}) is greater than the number of classes ({len(cfg.CLASS_NAMES)})\n")
                else:
                    file.write("All predicted labels are within the valid range.\n")

                min_label = np.min(labels)
                if min_label == 0:
                    file.write("Labels start from 0.\n")
                else:
                    file.write("Labels start from 1.\n")

                if not OPEN3D_FLAG:
                    filename = f'/home/neha/OpenPCDet/data/results/image_{idx}.png'
                    mlab.savefig(filename)
                    mlab.close()
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
