from .detector3d_template import Detector3DTemplate
import torch

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # Enabling benchmark gives a small boost (5ms)
        torch.backends.cudnn.benchmark = True
        # Enabling these doesnt speed up...
        #torch.backends.cuda.matmul.allow_tf32 = True
        #torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed(0)
        self.is_voxel_enc=True

        if self.model_cfg.get('BACKBONE_3D', None) is None:
            #pillar
            self.is_voxel_enc=False
            self.vfe, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],})
        else:
            #voxel
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Backbone3D':[],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],})
        self.post_processing_func = self.post_processing
        self.save_voxels = False
        self.sample_counter = 0

    def forward(self, batch_dict):
        #for cur_module in self.module_list:
        #    batch_dict = cur_module(batch_dict)

        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        self.measure_time_end('VFE')
        #print('points', batch_dict['points'].size())
        #print('voxels', batch_dict['voxels'].size())
        #print('voxel_coords', batch_dict['voxel_coords'].size())
        #print(batch_dict['voxel_coords'])
        #print(torch.sum(batch_dict['voxel_coords'][:,0]))

        if self.save_voxels:
            torch.save(batch_dict['voxel_coords'].cpu(),
                f'/root/shared_data/voxel_coords/voxelcoords_{self.sample_counter}.pt')
            self.sample_counter += 1

        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')
         
        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')
        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('Backbone2D')
        self.measure_time_start('CenterHead')
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('CenterHead')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # I don't wanna do this before final syncronization
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #return pred_dicts, recall_dicts
            return batch_dict

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
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def calibrate(self):
        batch_dict = self.load_data_with_ds_index(0)
        batch_dict = self.vfe(batch_dict)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        self.dense_head.calibrate(batch_dict)
        self.clear_stats()

        # I should't do that first because I need the tensors of center head to be preallocated
        super().calibrate()
