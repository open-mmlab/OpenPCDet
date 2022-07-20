import torch
from .detector3d_template import Detector3DTemplate
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os
import numpy as np
import time
from ...utils import common_utils
from pcdet.datasets.augmentor import augmentor_utils, database_sampler

class MPPNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):

        batch_dict['proposals_list'] = batch_dict['roi_boxes']


        for cur_module in self.module_list[:]:
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)

            if self.model_cfg.POST_PROCESSING.get('USE_MEMORYBANK',False):
                return pred_dicts, recall_dicts, batch_dict
            else:
                return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}  
        tb_dict ={}
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rcnn

        return loss, tb_dict, disp_dict

