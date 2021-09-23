import torch
import copy
import os

from .detector3d_template import Detector3DTemplate

# additional data augementation is implemented
from pcdet.datasets.augmentor.augmentor_utils import *
from .pv_rcnn import PVRCNN 

class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger=None, global_cfg=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset,logger=logger, global_cfg=global_cfg)
        # apply deepcopy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        
        # create model
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger, global_cfg=global_cfg)
        # no logger is feeded
        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        # detach parameters for semi supervised training
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # # data augement
        # self.thresh = model_cfg.THRESH
        # self.sem_thresh = model_cfg.SEM_THRESH
        # self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        # self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        # self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        # self.no_nms = model_cfg.NO_NMS
        # self.supervise_model = model_cfg.SUPERVISE_MODE

    def forward(self, batch_dict):
        """
        TRAINING PROCESS:
        1. 
        """
        if self.training:
            # 拿到带标签的输入数据
            mask = batch_dict['mask'].view(-1)
            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()

            batch_dict_ema = {}

            keys = list(batch_dict.keys())
            for k in keys:
                # 为半监督的训练提供训练的batch dict
                if k + '_ema' in keys:
                    continue
                if k.endwith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # train with unlabeled data
                # compute pseudo labels
                for cur_module in self.pv_rcnn_ema.module_list:
                    # train with EMA
                    batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema)
            
                pseudo_boxes = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0

                for ind in unlabeled_mask:
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_label']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        continue

                    # update the pseudo targets pseudo label的shape [N 8]
                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        # 经过filter之后得到的pseudo box的数量
                        max_pseudo_box_num = pseudo_box.shape[0]

                max_box_num = batch_dict['gt_boxes'].shape[1]
                # construct new training batch data
                if max_box_num >= max_pseudo_box_num:
                    # 如果gt框的数量大于pseudo label的数量
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        # 只是对pseudo label提供训练的数据
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        # 更新pseudo label的标签
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box

                else:
                    # pseudo label的数量多于gt label的数量
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                            device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        # 带标签的数据
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        # 无标签的数据
                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)
            
            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss()
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_loss = loss_rpn_cls[labeled_mask, ...].sum() + loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight
            
            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight
            loss_point = loss_point[labeled_mask, ...].sum()
            loss_rcnn_cls = loss_rcnn_cls[labeled_mask, ...].sum()

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum() + loss_rcnn_box[unlabeled_mask, ...].sum()
            
            loss = loss_rpn_loss + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box

            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key + '_labeled'] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + '_unlabeled'] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key + '_labeled'] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + '_unlabeled'] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'point_pos_num' in key:
                    tb_dict_[key + '_labeled'] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + '_unlabeled'] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]
            
            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict_, disp_dict
        
        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
    
    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
            
        logger.info('==> Loading parameters from checkpoint %s to %s ' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s ' % checkpoint['version'])
        
        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            
        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s ' % (key, str(state_dict[key].shape)))
        
        logger.info('==> Done (loaded %d / %d )' % (len(update_model_state), len(self.state_dict())))