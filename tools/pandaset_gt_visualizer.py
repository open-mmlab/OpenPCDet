#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:06:30 2022

@author: yagmur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:04:53 2022

@author: yagmur
"""

import argparse
import glob
from pathlib import Path
import os

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V
OPEN3D_FLAG = False    

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.pandaset.pandaset_dataset import PandasetDataset



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
   
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = PandasetDataset(
        dataset_cfg=cfg, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )

 
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

 
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info('Visualized sample name %s: \t' , demo_dataset[idx]["frame_idx"])
            data_dict = demo_dataset.collate_batch([data_dict])
                          
         #   print(data_dict['gt_boxes'][0,:,:7])
                
            V.draw_scenes(
                 points=data_dict['points'][:, 1:], ref_boxes=data_dict['gt_boxes'][0,:,:7]
             )

            if not OPEN3D_FLAG:
                 mlab.show(stop=True)

    logger.info('Inference done.')


if __name__ == '__main__':
    main()
