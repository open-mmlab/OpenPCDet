from .detector3d_template import Detector3DTemplate
import os
import torch
import numpy as np

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:

            pred_dicts, recall_dicts = self.post_processing(batch_dict)

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
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}

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

        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
