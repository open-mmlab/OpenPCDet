#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:39:02 2022

@author: yagmur
"""

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
import argparse
from pcdet.models import load_data_to_gpu
import torch
from pcdet.utils import common_utils
import os

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                    help='specify the config for demo')
parser.add_argument('--data_path', type=str, default='demo_data',
                    help='specify the point cloud data file or directory')
parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
parser.add_argument('--out_folder', type=str, default=None, help='save demo results to the output folder')


args = parser.parse_args()

cfg_from_yaml_file(args.cfg_file, cfg)


test_set, test_loader, sampler = build_dataloader(
       dataset_cfg=cfg.DATA_CONFIG,
       class_names=cfg.CLASS_NAMES,
       batch_size=1,
       dist=True, workers=4, training=False
   )

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

dataset = test_loader.dataset
class_names = dataset.class_names
det_annos = []

logger = common_utils.create_logger()
model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
model.cuda()
model.eval()

for i, batch_dict in enumerate(test_loader):
    load_data_to_gpu(batch_dict)
    with torch.no_grad():
        pred_dicts, ret_dict = model(batch_dict)

    annos = dataset.generate_prediction_dicts(
        batch_dict, pred_dicts, class_names,
        output_path=None 
    )
    
    print(pred_dicts)




    if not os.path.exists(args.out_folder + "/annotations/"):
          os.makedirs(args.out_folder + "/annotations/")
     
    for idx in range(len(annos)):    
     
        filename = args.out_folder + "/annotations/" + annos[idx]["frame_id"] + ".txt"
        annotation_file = open(filename, "w+")
    
        print(annos[idx]["name"])
        object_counts = len(annos[idx]["name"])
       
        print("predicted objects for this image", object_counts)

        for i in range(object_counts):                    
            print(annos[idx]["name"][i])
            annotation_file.write(annos[idx]["name"][i] + " ")
            annotation_file.write(str(round(annos[idx]['truncated'][i],4)) + " ")
            annotation_file.write(str(round(annos[idx]['occluded'][i],4)) + " ")
            annotation_file.write(str(round(annos[idx]['alpha'][i],4)) + " ")
            string = str(annos[idx]['bbox'][i]).replace("[", "")
            string = string.replace("]", "")
            annotation_file.write(string + " ")
            string = str(annos[idx]['dimensions'][i]).replace("[", "")
            string = string.replace("]", "")
            annotation_file.write(string + " ")
            string = str(annos[idx]['location'][i]).replace("[", "")
            string = string.replace("]", "")
            annotation_file.write(string + " ")
            annotation_file.write(str(round(annos[idx]['rotation_y'][i],4)) + " ")
            annotation_file.write(str(round(pred_dicts[idx]['pred_scores'][i].item(),4)) + " ")
            annotation_file.write("\n")


        annotation_file.close()