from .detector3d_template import Detector3DTemplate


class CaDDN(Detector3DTemplate):
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

        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.vfe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth
        return loss, tb_dict, disp_dict
