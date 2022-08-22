import torch
import os
import numpy as np
import copy
from ...utils import common_utils
from ..model_utils import model_nms_utils
from .detector3d_template import Detector3DTemplate
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.datasets.augmentor import augmentor_utils, database_sampler

class MPPNetE2E(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'backbone_2d', 'dense_head','roi_head'
        ]
        self.num_frames = 1 - self.dataset.dataset_cfg.SEQUENCE_CONFIG.SAMPLE_OFFSET[0]

    def reset_memorybank(self):
        self.memory_rois = None
        self.memory_labels = None
        self.memory_scores = None
        self.memory_feature = None

    def forward(self, batch_dict):

        if batch_dict['sample_idx'][0] ==0:
            self.reset_memorybank()
            batch_dict['memory_bank'] = {}
        else:
            batch_dict['memory_bank'] = {'feature_bank':self.memory_feature}

        for idx, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            if self.module_topology[idx] == 'dense_head':

                if self.memory_rois is None:
                    self.memory_rois = [batch_dict['rois']]*self.num_frames
                    self.memory_labels = [batch_dict['roi_labels'][:,:,None]]*self.num_frames
                    self.memory_scores = [batch_dict['roi_scores'][:,:,None]]*self.num_frames
                else:
                    self.memory_rois.pop()
                    self.memory_rois.insert(0,batch_dict['rois'])
                    self.memory_labels.pop()
                    self.memory_labels.insert(0,batch_dict['roi_labels'][:,:,None])
                    self.memory_scores.pop()
                    self.memory_scores.insert(0,batch_dict['roi_scores'][:,:,None])


                batch_dict['memory_bank'].update({'rois': self.memory_rois,
                                                'roi_labels': self.memory_labels,
                                                'roi_scores': self.memory_scores})


            if self.module_topology[idx] == 'roi_head':
                if self.memory_feature is None:
                    self.memory_feature = [batch_dict['geometory_feature_memory'][:,:64]]*self.num_frames

                else:
                    self.memory_feature.pop()
                    self.memory_feature.insert(0,batch_dict['geometory_feature_memory'][:,:64])


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.model_cfg.POST_PROCESSING.get('SAVE_BBOX',False):

                for bs_idx in range(batch_dict['batch_size']):

                    if batch_dict['final_box_dicts'][bs_idx]['pred_boxes'].shape[0] >0:
                        cur_vels = batch_dict['final_box_dicts'][bs_idx]['pred_boxes'][:,7:9]
                        cur_boxes = batch_dict['final_box_dicts'][bs_idx]['pred_boxes']
                        cur_scores = batch_dict['final_box_dicts'][bs_idx]['pred_scores']
                        cur_labels = batch_dict['final_box_dicts'][bs_idx]['pred_labels']

                        path = os.path.join(self.dataset.dataset_cfg.DATA_PATH, self.model_cfg.POST_PROCESSING.BBOX_SAVE_PATH, '%s' % (batch_dict['frame_id'][bs_idx][:-4]))

                        if not os.path.exists(path):
                            try:
                                os.makedirs(path)
                            except:
                                pass
                        bbox_path = os.path.join(path, '%s.npy' % (batch_dict['frame_id'][bs_idx][-3:]))
                
                        pred_boxes = torch.cat([cur_boxes[:,:7],cur_scores[:,None],cur_labels[:,None],cur_vels],dim=-1)
                        np.save(bbox_path, pred_boxes.cpu().numpy())
                    else:
                        pass
                       
            if self.model_cfg.POST_PROCESSING.get('USE_MEMORYBANK',False):
                return pred_dicts, recall_dicts, batch_dict
            else:
                return pred_dicts, recall_dicts


    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict


    def post_processing(self, batch_dict):

        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                try:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                except:
                    record_dict = {
                        'pred_boxes': torch.tensor([]),
                        'pred_scores': torch.tensor([]),
                        'pred_labels': torch.tensor([])
                    }
                    pred_dicts.append(record_dict)
                    continue

                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

                #########  Car DONOT Using NMS ###### 
                if post_process_cfg.get('NOT_APPLY_NMS_FOR_VEL',False):
                    
                    pedcyc_mask = final_labels !=1 
                    final_scores_pedcyc = final_scores[pedcyc_mask]
                    final_labels_pedcyc = final_labels[pedcyc_mask]
                    final_boxes_pedcyc = final_boxes[pedcyc_mask]

                    car_mask = (label_preds==1) & (cls_preds > post_process_cfg.SCORE_THRESH)
                    final_scores_car = cls_preds[car_mask]
                    final_labels_car = label_preds[car_mask]
                    final_boxes_car = box_preds[car_mask]

                    final_scores  = torch.cat([final_scores_car,final_scores_pedcyc],0)
                    final_labels  = torch.cat([final_labels_car,final_labels_pedcyc],0)
                    final_boxes  = torch.cat([final_boxes_car,final_boxes_pedcyc],0)

                #########  Car DONOT Using NMS ###### 

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            

            record_dict = {
                'pred_boxes': final_boxes[:,:7],
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

