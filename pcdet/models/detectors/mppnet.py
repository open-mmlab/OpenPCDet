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

        if self.model_cfg.ROI_HEAD.get('USE_BEV_FEAT',None):
            with torch.no_grad():
                for cur_module in self.module_list[:-1]:
                    batch_dict = cur_module(batch_dict)
            batch_dict =  self.module_list[-1](batch_dict)
            
        else:
            for cur_module in self.module_list[:]:
                batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:

            # if self.model_cfg.POST_PROCESSING.get('SAVE_BBOX',False):
            #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
               
            #     for bs_idx in range(batch_dict['batch_size']):

            #         cur_boxes = batch_dict['batch_box_preds'][bs_idx]
            #         cur_scores = batch_dict['batch_cls_preds'][bs_idx]
            #         cur_labels = batch_dict['roi_labels'][bs_idx]
            #         cur_superboxes = batch_dict['pred_superboxes'][bs_idx]
                    


            #         path = '/home/xschen/OpenPCDet_xuesong/iter_6933/%s/' % (batch_dict['frame_id'][bs_idx][:-4])

            #         if not os.path.exists(path):
            #             try:
            #                 os.makedirs(path)
            #             except:
            #                 pass
            #         bbox_path = path + '%s.npy' % (batch_dict['frame_id'][bs_idx][-3:])
            
            #         #cur_gtboxes = torch.cat([cur_gtboxes[k:k+1],torch.ones([1,1]).cuda()],-1)
            #         pred_boxes = torch.cat([cur_boxes,cur_scores,cur_labels[:,None],cur_superboxes],dim=-1)
            #         np.save(bbox_path, pred_boxes.cpu().numpy())

            # else:
            #     start_time = time.time()
            pred_dicts, recall_dicts = self.post_processing(batch_dict,nms=self.model_cfg.POST_PROCESSING.get('USE_NMS',True))

            # torch.cuda.empty_cache()
            return pred_dicts, recall_dicts, batch_dict

    def get_training_loss(self):
        disp_dict = {}  
        if self.model_cfg.ONLY_TRAIN_RCNN:
            tb_dict ={}
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rcnn
        elif self.model_cfg.ONLY_TRAIN_RPN:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss = loss_rpn
        else:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

