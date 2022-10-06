from .detector3d_template import Detector3DTemplate, pre_forward_hook
import torch
from sbnet.layers import ReduceMask
from ...ops.cuda_projection import cuda_projection
import json

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
        self.num_tiles_to_process = int(self.total_num_tiles.cpu().item() / 100 * 30)
        #self.num_tiles_to_process = self.total_num_tiles.cpu().item()
        self.reduce_mask_stream = torch.cuda.Stream()
        ################################################################################

        ####Projection###
        self.enable_projection = False
        self.token_to_scene = {}
        self.token_to_ts = {}
        if self.enable_projection:
            with open('token_to_pos.json', 'r') as handle:
                self.token_to_pose = json.load(handle)
            #timestamps = [v['timestamp'] for v in self.token_to_pose.values()]
            #timestamps = torch.tensor(timestamps, dtype=torch.long)
            #timestamps -= torch.min(timestamps)
            #timestamps //= 1000 # make is ms

            for k, v in self.token_to_pose.items():
                cst, csr, ept, epr = v['cs_translation'],  v['cs_rotation'], \
                        v['ep_translation'], v['ep_rotation']
                # convert time stamps to seconds
                # 1 3 4 3 4
                self.token_to_pose[k] = torch.tensor((*cst, *csr, *ept, *epr), dtype=torch.float)
                self.token_to_ts[k] = torch.tensor((v['timestamp'],), dtype=torch.long)
                self.token_to_scene[k] = v['scene']

        self.past_detections = {'num_dets': []}
        # Poses include [ts(1) cst(3) csr(4) ept(3) epr(4)]
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)
        self.det_timeout_limit = int(0.11 * 1000000) # in microseconds
        self.prev_scene_token = ''
        #################

        print(self)

    def produce_reduce_mask(self, data_dict):
        tile_coords = data_dict['chosen_tile_coords']
        total_num_tiles = data_dict['total_num_tiles']
        tcount = self.model_cfg.TILE_COUNT
        batch_idx = torch.div(tile_coords, total_num_tiles, rounding_mode='trunc').short()
        row_col_idx = tile_coords - batch_idx * total_num_tiles
        row_idx = torch.div(row_col_idx, tcount[0], rounding_mode='trunc').short()
        col_idx = (row_col_idx - row_idx * tcount[1]).short()
        inds = torch.stack((batch_idx, col_idx, row_idx), dim=1)
        counts = torch.full((1,), inds.size(0), dtype=torch.int32)
        return ReduceMask(inds, counts)


    def forward(self, batch_dict):
        if self.enable_projection and batch_dict['batch_size'] == 1:
            self.latest_token = batch_dict['metadata'][0]['token']
            self.cur_pose = self.token_to_pose[self.latest_token]
            self.cur_ts = self.token_to_ts[self.latest_token]
            scene_token = self.token_to_scene[self.latest_token]
            if scene_token != self.prev_scene_token:
                self.projection_reset()

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
            pred_dicts = batch_dict['final_box_dicts']
            if self.enable_projection:
                # First, remove the outdated detections
                num_dets, dets_to_rm = self.past_detections['num_dets'], []
                while num_dets:
                    # timestamp comparison
                    if (self.cur_ts[0] - self.past_ts[len(dets_to_rm)]) <= self.det_timeout_limit:
                        break
                    dets_to_rm.append(num_dets.pop(0))
                if dets_to_rm:
                    self.past_poses = self.past_poses[len(dets_to_rm):]
                    self.past_ts = self.past_ts[len(dets_to_rm):]
                    dets_to_rm_sum = sum(dets_to_rm)
                    for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx', 'tile_inds'):
                        self.past_detections[k] = self.past_detections[k][dets_to_rm_sum:]

                # Project now
                projected_boxes=None
                if self.past_poses.size(0) > 0:
#                    print(self.past_detections['pred_boxes'].size())
#                    print(self.past_detections['pred_boxes'])
#                    print(self.past_detections['pose_idx'].size())
#                    print(self.past_detections['pose_idx'])
#                    print(self.past_poses.size())
#                    print(self.past_poses)
#                    print(self.cur_pose.size())
#                    print(self.cur_pose)

                    # all these cuda calls might not be good
                    mask, projected_boxes = cuda_projection.project_past_detections(
                            batch_dict['chosen_tile_coords'],
                            self.past_detections['tile_inds'],
                            self.past_detections['pred_boxes'],
                            self.past_detections['pose_idx'],
                            self.past_poses.cuda(),
                            self.cur_pose.cuda(),
                            self.past_ts.cuda(),
                            self.cur_ts.item())

                    nanmask = torch.isfinite(projected_boxes)
                    nanmask = torch.all(nanmask, dim=1)
                    mask = torch.logical_and(mask, nanmask)

                    projected_boxes = projected_boxes[mask]
                    print('projected boxes:', projected_boxes.size())
                    projected_scores = self.past_detections['pred_scores'][mask]
                    projected_labels = self.past_detections['pred_labels'][mask]

                # Second, append new detections
                num_dets = pred_dicts[0]['pred_labels'].size(0)
                self.past_detections['num_dets'].append(num_dets)
                # Append the current pose
                self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
                self.past_ts = torch.cat((self.past_ts, self.cur_ts))
                # Append the pose idx for the detection that will be added
                past_poi = self.past_detections['pose_idx'] - len(dets_to_rm)
                poi = torch.full((num_dets,), self.past_poses.size(0)-1,
                    dtype=past_poi.dtype, device=past_poi.device)
                self.past_detections['pose_idx'] = torch.cat((past_poi, poi))
                for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'tile_inds'):
                    self.past_detections[k] = torch.cat((self.past_detections[k], pred_dicts[0][k]))

                self.prev_scene_token = scene_token

                # append the projected detections
                if projected_boxes is not None:
                    pred_dicts[0]['pred_boxes'] = torch.cat((pred_dicts[0]['pred_boxes'], 
                        projected_boxes))
                    pred_dicts[0]['pred_scores'] = torch.cat((pred_dicts[0]['pred_scores'],
                        projected_scores))
                    pred_dicts[0]['pred_labels'] = torch.cat((pred_dicts[0]['pred_labels'],
                        projected_labels))

                    batch_dict['final_box_dicts'] = pred_dicts  # needed?

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

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.past_detections = self.get_empty_det_dict()
        self.past_detections['num_dets'] = []
        self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long,
            device=self.past_detections["pred_labels"].device)
        self.past_detections['tile_inds'] = torch.zeros([0], dtype=torch.long,
            device=self.past_detections["pred_labels"].device)
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)

    def calibrate(self, batch_size=1):
        ep = self.enable_projection
        self.enable_projection = False
        pred_dicts = super().calibrate(batch_size)
        self.projection_reset()
        self.enable_projection = ep
        return pred_dicts
