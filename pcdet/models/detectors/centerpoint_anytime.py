from .detector3d_template import Detector3DTemplate, pre_forward_hook
import torch
from sbnet.layers import ReduceMask

class CenterPointAnytime(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)

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
            self.is_voxel_enc=True
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Backbone3D':[],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],})

        ################################################################################
        self.tcount= torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.total_num_tiles = self.tcount[0] * self.tcount[1]

        #Tile prios are going to be updated dynamically, initially all tiles have equal priority
        self.tile_prios = torch.full((self.total_num_tiles,), \
                self.total_num_tiles//2, dtype=torch.long, device='cuda')
        #self.tile_prios = torch.randint(0, self.total_num_tiles, (self.total_num_tiles,), \
        #        dtype=torch.long, device='cuda')

        # This number will be determined by the scheduling algorithm initially for each input
        self.num_tiles_to_process = self.total_num_tiles.cpu().item()
        self.reduce_mask_stream = torch.cuda.Stream()
        ################################################################################

        print(self)

    def produce_reduce_mask(self, data_dict):
        tile_coords = data_dict['chosen_tile_coords']
        total_num_tiles = data_dict['total_num_tiles']
        tcount = self.model_cfg.TILE_COUNT
        batch_idx = torch.div(tile_coords, total_num_tiles, rounding_mode='trunc').short()
        row_col_idx = tile_coords - batch_idx * total_num_tiles
        row_idx = torch.div(row_col_idx, tcount[0], rounding_mode='trunc').short()
        col_idx = (row_col_idx - row_idx * tcount[1]).short()
        inds = torch.stack((batch_idx, col_idx, row_idx), dim=1) # should this stay as col row?
        counts = torch.full((1,), inds.size(0), dtype=torch.int32)
        return ReduceMask(inds, counts)


    def forward(self, batch_dict):
        for v in ('tcount','tile_prios','num_tiles_to_process', 'total_num_tiles'):
            batch_dict[v] = getattr(self, v)

        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        self.measure_time_end('VFE')

        # Produce the reduce mask in parallel in a seperate stream
        with torch.cuda.stream(self.reduce_mask_stream):
            batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)

        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.reduce_mask_stream.synchronize()
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

    def post_processing_pre(self, batch_dict):
        return (batch_dict,)

    def post_processing_post(self, pp_args):
        batch_dict = pp_args[0]
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

    def calibrate(self, batch_size=1):
        batch_dict = pre_forward_hook(self, ([i for i in range(batch_size)],))
        for v in ('tcount','tile_prios','num_tiles_to_process', 'total_num_tiles'):
            batch_dict[v] = getattr(self, v)

        batch_dict = self.vfe(batch_dict)
        with torch.cuda.stream(self.reduce_mask_stream):
            batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        self.reduce_mask_stream.synchronize()
        batch_dict = self.backbone_2d(batch_dict)
        self.dense_head.calibrate(batch_dict)
        self.clear_stats()
