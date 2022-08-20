from .detector3d_template import Detector3DTemplate
import torch

#128 151 178
class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.vfe, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list

        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(0)

        self.update_time_dict( {
                'VFE': [], #'PillarFeatureNet': [],
                'MapToBEV': [], #'PillarScatter': [],
                'RPN-finalize': [],
                'RPN-total': [],
                'Post-RPN': [],
                'PostProcess': [],})

    def forward(self, batch_dict):
        #for cur_module in self.module_list:
        #    batch_dict = cur_module(batch_dict)

        #self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        #self.measure_time_end('VFE')
        #self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        #self.measure_time_end('MapToBEV')
        self.measure_time_start('RPN-total')
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('RPN-total')
        self.measure_time_start('Post-RPN')
        self.measure_time_start('RPN-finalize')
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('RPN-finalize')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            self.measure_time_end('Post-RPN')

            return ret_dict, tb_dict, disp_dict
        else:
            self.measure_time_start('PostProcess')
            pred_dicts, recall_dicts = self.post_processing(batch_dict, False)
            self.measure_time_end('PostProcess')

            for dd in pred_dicts:
                for k,v in dd.items():
                    dd[k] = v.cpu()
            self.measure_time_end('Post-RPN')

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
