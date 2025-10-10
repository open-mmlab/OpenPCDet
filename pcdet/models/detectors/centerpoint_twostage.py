from .detector3d_template import Detector3DTemplate


class CenterPointTwoStage(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)

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
        
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    # def post_processing(self, batch_dict):
    #     post_process_cfg = self.model_cfg.POST_PROCESSING
    #     batch_size = batch_dict['batch_size']
    #     # final_pred_dict = batch_dict['final_box_dicts']
    #     final_preds = batch_dict['batch_box_preds']
    #     recall_dict = {}
    #     for index in range(batch_size):
    #         pred_boxes = final_pred_dict[index]['pred_boxes']

    #         recall_dict = self.generate_recall_record(
    #             box_preds=pred_boxes,
    #             recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
    #             thresh_list=post_process_cfg.RECALL_THRESH_LIST
    #         )

    #     return final_pred_dict, recall_dict
