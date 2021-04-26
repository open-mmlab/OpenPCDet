from .detector3d_template import Detector3DTemplate


class RangeTemplate(Detector3DTemplate):
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
        weight_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        seg_weight = weight_dict['seg_weight']
        rpn_weight = weight_dict['rpn_weight']
        disp_dict = {}

        loss_seg = self.seg_head.get_loss()
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_seg * seg_weight + loss_rpn * rpn_weight
        return loss, tb_dict, disp_dict


class RSN(RangeTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

    def post_processing(self, batch_dict):
        pred_dicts = batch_dict['pred_dicts']
        recall_dict = {}
        batch_size = batch_dict['batch_size']
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                box_preds=pred_dicts[index]['pred_boxes'],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=self.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST
            )

        return pred_dicts, recall_dict
