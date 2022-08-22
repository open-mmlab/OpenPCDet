from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.functional as F
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.mppnet_utils import build_transformer, PointNet, MLP
from .target_assigner.proposal_target_layer import ProposalTargetLayer
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules

class MPPNetHeadE2E(RoIHeadTemplate):
    def __init__(self,model_cfg, num_class=1,**kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        # self.proposal_target_layer = ProposalTargetLayerMPPNet(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.use_time_stamp = self.model_cfg.get('USE_TIMESTAMP',None)
        self.num_lidar_points = self.model_cfg.Transformer.num_lidar_points
        self.avg_stage1_score = self.model_cfg.get('AVG_STAGE1_SCORE', None)

        self.nhead = model_cfg.Transformer.nheads
        self.num_enc_layer = model_cfg.Transformer.enc_layers
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.num_groups = model_cfg.Transformer.num_groups

        self.grid_size = model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.num_proxy_points = model_cfg.Transformer.num_proxy_points
 
        self.seqboxembed = PointNet(8,model_cfg=self.model_cfg)
        self.jointembed = MLP(self.hidden_dim*(self.num_groups+1), model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4)


        num_radius = len(self.model_cfg.ROI_GRID_POOL.POOL_RADIUS)
        self.up_dimension_geometry = MLP(input_dim = 29, hidden_dim = 64, output_dim =hidden_dim//num_radius, num_layers = 3)
        self.up_dimension_motion = MLP(input_dim = 30, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)

        self.transformer = build_transformer(model_cfg.Transformer)

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
                nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
                mlps=self.model_cfg.ROI_GRID_POOL.MLPS,
                use_xyz=True,
                pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
                )

        self.class_embed = nn.ModuleList()
        self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim, 1))

        self.bbox_embed = nn.ModuleList()
        for _ in range(self.num_groups):
            self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4))

        if self.model_cfg.Transformer.use_grid_pos.enabled:
            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                self.grid_index = torch.cat([i.reshape(-1,1)for i in torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size), torch.arange(self.grid_size))],1).float().cuda()
                self.grid_pos_embeded = MLP(input_dim = 3, hidden_dim = 256, output_dim = hidden_dim, num_layers = 2)
            else:
                self.pos = nn.Parameter(torch.zeros(1, self.num_grid_points, 256))

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.bbox_embed.layers[-1].weight, mean=0, std=0.001)

    def get_corner_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)  # (BxN, 2x2x2, 3)
        local_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()


        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        if isinstance(grid_size,list):
            faked_features = rois.new_ones((grid_size[0], grid_size[1], grid_size[2]))
            grid_size = torch.tensor(grid_size).float().cuda()
        else:
            faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = torch.div((dense_idx + 0.5), grid_size) * local_roi_size.unsqueeze(dim=1) - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 2x2x2, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 2x2x2, 3)
        return roi_grid_points

    def get_proxy_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  
        local_roi_grid_points = common_utils.rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    def roi_grid_pool(self, batch_size, rois, point_coords, point_features,batch_dict=None,batch_cnt=None,batch_voxel_cnt=None,src=None,effi=True):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """

        num_frames = batch_dict['num_frames']
        num_rois = rois.shape[2]*rois.shape[1]



        global_roi_proxy_points, local_roi_proxy_points = self.get_proxy_points_of_roi(
            rois.permute(0,2,1,3).contiguous(), grid_size=self.grid_size
        )  

        

        if not effi:
            global_roi_proxy_points = global_roi_proxy_points.view(batch_size, -1, 3)
            point_coords = point_coords.view(point_coords.shape[0]*num_frames,point_coords.shape[1]//num_frames,point_coords.shape[-1])
            xyz = point_coords[:, :, 0:3].view(-1,3)
            num_points = point_coords.shape[1]
            num_proxy_points = self.num_proxy_points

            if batch_cnt is None:
                xyz_batch_cnt = torch.tensor([num_points]*num_rois*batch_size).cuda().int()
            else:
                xyz_batch_cnt = torch.tensor(batch_cnt).cuda().int()

            new_xyz_batch_cnt = torch.tensor([num_proxy_points]*num_rois*batch_size).cuda().int()
            new_xyz = global_roi_proxy_points.view(-1, 3)
        else:

            num_points = point_coords.shape[1]
            num_proxy_points = self.num_proxy_points

            xyz = point_coords[:, :, 0:3].view(-1,3)
            if batch_cnt is None:
                xyz_batch_cnt = torch.tensor([point_coords.shape[1]]*rois.shape[2]*batch_size).cuda().int() 
            else:
                xyz_batch_cnt = torch.tensor(batch_cnt).cuda().int()
            new_xyz = torch.cat([i[0] for i in global_roi_proxy_points.chunk(rois.shape[2],0)],0)
            new_xyz_batch_cnt = torch.tensor([self.num_proxy_points]*rois.shape[2]*batch_size).cuda().int()
        
        _, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.view(-1,point_features.shape[-1]).contiguous(),
        )  

        if effi:
            features = pooled_features.view(
                point_features.shape[0], self.num_proxy_points,
                pooled_features.shape[-1]
            ).contiguous()  
        else:
            features = pooled_features.view(
                point_features.shape[0], num_frames*self.num_proxy_points,
                pooled_features.shape[-1]
            ).contiguous() 

        return features,global_roi_proxy_points.view(batch_size*rois.shape[2], num_frames*num_proxy_points,3).contiguous()

    def get_grid_points_of_roi(self, rois, grid_size):
        # import pdb;pdb.set_trace()
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        local_roi_grid_points = common_utils.rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_batch_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 2x2x2, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 2x2x2, 3)
        return roi_grid_points

    @staticmethod
    def reorder_rois_by_superbox(batch_size, pred_dicts, use_superbox=False):
        """
        Args:
            final_box_dicts: (batch_size), list of dict
                pred_boxes: (N, 7 + C)
                pred_scores: (N)
                pred_labels: (N)
            batch_box_preds: (batch_size, num_rois, 7 + C)
        """
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])
            if use_superbox:
                rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_supboxes']
            else:
                rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
        return rois
    
    def spherical_coordinate(self, src, diag_dist):

        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def box_spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 24)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def reorder_rois(batch_size, batch_dict):
        moving_mask = batch_dict['moving_mask']

        moving_rois = batch_dict['rois'].new_zeros(batch_dict['rois'].shape)
        moving_roi_scores = batch_dict['rois'].new_zeros((batch_size, 128))
        moving_roi_labels = batch_dict['rois'].new_zeros((batch_size, 128)).long()

        static_rois = batch_dict['rois'].new_zeros(batch_dict['rois'].shape)
        static_roi_scores = batch_dict['rois'].new_zeros((batch_size, 128))
        static_roi_labels = batch_dict['rois'].new_zeros((batch_size, 128)).long()

        for bs_idx in range(batch_size):
            mask = moving_mask[bs_idx]

            moving_rois[bs_idx, mask, :] = batch_dict['rois'][bs_idx,mask,:]
            # moving_roi_scores[bs_idx, mask] = batch_dict['rois_scores'][bs_idx]
            moving_roi_labels[bs_idx, mask] = batch_dict['rois_labels'][bs_idx]

            static_rois[bs_idx, ~mask, :] = batch_dict['rois'][bs_idx,~mask,:]
            # static_roi_scores[bs_idx, ~mask] = batch_dict['rois_scores'][bs_idx,~mask]
            static_roi_labels[bs_idx, ~mask] = batch_dict['rois_labels'][bs_idx,~mask]
        return moving_rois,moving_roi_labels, static_rois, static_roi_labels

    def compute_mask(self, xyz, radius, dist=None):

        # import pdb;pdb.set_trace()
        xyz = xyz.view(xyz.shape[0]*4,xyz.shape[1]//4,-1)
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
            token_mask = torch.ones(mask.shape[0],1,mask.shape[-1]).bool().cuda()
            mask = torch.cat([token_mask,mask],1)
            token_mask = torch.ones(mask.shape[0],mask.shape[1],1).bool().cuda()
            mask = torch.cat([token_mask,mask],-1)
            mask[:,0,0] = False
        return mask, dist

    def transform_prebox2current(self,pred_boxes3d,pose_pre,pose_cur):

        expand_bboxes = np.concatenate([pred_boxes3d[:,:3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)
        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        expand_bboxes_global = np.concatenate([bboxes_global[:,:3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        bboxes_pre2cur = np.concatenate([bboxes_pre2cur, pred_boxes3d[:,3:9]],axis=-1)
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] + np.arctan2(pose_pre[..., 1, 0], pose_pre[..., 0,0])
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] - np.arctan2(pose_cur[..., 1, 0], pose_cur[..., 0,0])

        return torch.from_numpy(bboxes_pre2cur).cuda()

    def generate_trajectory(self,cur_batch_boxes,proposals_list,roi_scores_list,batch_dict):
        frame1 = cur_batch_boxes
        trajectory_rois = cur_batch_boxes[:,None,:,:].repeat(1,batch_dict['rois'].shape[-2],1,1)
        trajectory_roi_scores = torch.zeros_like(batch_dict['roi_scores'].permute(0,2,1))
        trajectory_rois[:,0,:,:]= frame1
        trajectory_roi_scores[:,0,:] = batch_dict['roi_scores'][:,:,0]
        effective_length = torch.zeros([batch_dict['batch_size'],batch_dict['rois'].shape[-2],trajectory_rois.shape[2]])
        effective_length[:,0] = 1
        matching_table = (trajectory_rois.new_ones([trajectory_rois.shape[1],trajectory_rois.shape[2]]) * -1).long()
        for i in range(1,batch_dict['rois'].shape[-2]):
            frame = torch.zeros_like(cur_batch_boxes)

            frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] + trajectory_rois[:,i-1,:,7:9]

            frame[:,:,2:] = trajectory_rois[:,i-1,:,2:]

            for idx in range( batch_dict['batch_size']):
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame[idx][:,:7], proposals_list[idx,i,:,:7])
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                effective_length[idx,i,fg_inds] = 1
                matching_table[i,fg_inds] = gt_assignment[fg_inds]
                trajectory_rois[idx,i,:,:][fg_inds] = proposals_list[idx][i][gt_assignment[fg_inds]]
                trajectory_roi_scores[idx,i,:][fg_inds] = roi_scores_list[idx][i][gt_assignment[fg_inds]]

        batch_dict['effi_length'] = effective_length
    
        return trajectory_rois,trajectory_roi_scores,effective_length


    def crop_current_frame_points(self, src, batch_size,trajectory_rois,num_rois,num_sample, batch_dict):

        for bs_idx in range(batch_size):

            cur_batch_boxes = trajectory_rois[bs_idx,0,:,:7].view(-1,7)
            cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.1
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
            time_mask = cur_points[:,-1].abs() < 1e-3
            cur_points = cur_points[time_mask]
            dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))

            mask = point_mask
            sampled_idx = torch.topk(mask.float(),128)[1]
            sampled_idx_buffer = sampled_idx[:, 0:1].repeat(1, 128)  # (num_rois, npoints)
            roi_idx = torch.arange(num_rois)[:, None].repeat(1, 128)
            sampled_mask = mask[roi_idx, sampled_idx]  # (num_rois, 128)
            sampled_idx_buffer[sampled_mask] = sampled_idx[sampled_mask]

            src[bs_idx] = cur_points[sampled_idx_buffer][:,:,:5] # (num_rois, npoints)
            empty_flag = sampled_mask.sum(-1)==0
            src[bs_idx,empty_flag] = 0

        return src

    def trajectories_auxiliary_branch(self,trajectory_rois):

        time_stamp = torch.ones([trajectory_rois.shape[0],trajectory_rois.shape[1],trajectory_rois.shape[2],1]).cuda()
        for i in range(time_stamp.shape[1]):
            time_stamp[:,i,:] = i*0.1 

        box_seq = torch.cat([trajectory_rois[:,:,:,:7],time_stamp],-1)
        # box_seq_time = box_seq

        if self.model_cfg.USE_BOX_ENCODING.NORM_T0:
            # canonical transformation
            box_seq[:, :, :,0:3]  = box_seq[:, :, :,0:3] - box_seq[:, 0:1, :, 0:3]


        roi_ry = box_seq[:,:,:,6] % (2 * np.pi)
        roi_ry_t0 = roi_ry[:,0] 
        roi_ry_t0 = roi_ry_t0.repeat(1,box_seq.shape[1])

        # transfer LiDAR coords to local coords
        box_seq = common_utils.rotate_points_along_z(
            points=box_seq.view(-1, 1, box_seq.shape[-1]), angle=-roi_ry_t0.view(-1)
        ).view(box_seq.shape[0],box_seq.shape[1], -1, box_seq.shape[-1])

        if self.model_cfg.USE_BOX_ENCODING.ALL_YAW_T0:
            box_seq[:, :, :, 6]  =  0

        else:
            box_seq[:, 0:1, :, 6]  =  0
            box_seq[:, 1:, :, 6]  =  roi_ry[:, 1:, ] - roi_ry[:,0:1]


        batch_rcnn = box_seq.shape[0]*box_seq.shape[2]


        box_reg, box_feat, _ = self.seqboxembed(box_seq.permute(0,2,3,1).contiguous().view(batch_rcnn,box_seq.shape[-1],box_seq.shape[1]))
        
        return box_reg, box_feat


    def get_proposal_aware_motion_feature(self,proxy_point,batch_size,trajectory_rois,num_rois,batch_dict):

        time_stamp   = torch.ones([proxy_point.shape[0],proxy_point.shape[1],1]).cuda()
        padding_zero = torch.zeros([proxy_point.shape[0],proxy_point.shape[1],2]).cuda()
        proxy_point_padding = torch.cat([padding_zero,time_stamp],-1)

        num_time_coding = trajectory_rois.shape[1]

        for i in range(num_time_coding):
            proxy_point_padding[:,i*self.num_proxy_points:(i+1)*self.num_proxy_points,-1] = i*0.1


        ######### use T0 Norm ########
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,0,:,:].contiguous())  
        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  
        corner_points = corner_points.view(batch_size * num_rois, -1)
        corner_add_center_points = torch.cat([corner_points, trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,:3]], dim = -1)

        pos_fea = proxy_point[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1) 

        lwh = trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,proxy_point.shape[1],1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        ######### use T0 Norm ########

        proxy_point_padding = torch.cat([pos_fea,proxy_point_padding],-1)
        proxy_point_motion_feat = self.up_dimension_motion(proxy_point_padding)

        return proxy_point_motion_feat

    @staticmethod
    def reorder_rois_for_refining(pred_bboxes):

        num_max_rois = max([len(bbox) for bbox in pred_bboxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        ordered_bboxes = torch.zeros([len(pred_bboxes),num_max_rois,pred_bboxes[0].shape[-1]]).cuda()

        for bs_idx in range(ordered_bboxes.shape[0]):
            ordered_bboxes[bs_idx,:len(pred_bboxes[bs_idx])] = pred_bboxes[bs_idx]
        return ordered_bboxes


    def transform_prebox_to_current_vel(self,pred_boxes3d,pose_pre,pose_cur):

        expand_bboxes = np.concatenate([pred_boxes3d[:,:3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)
        expand_vels = np.concatenate([pred_boxes3d[:,7:9], np.zeros((pred_boxes3d.shape[0], 1))], axis=-1)
        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        vels_global = np.dot(expand_vels, pose_pre[:3,:3].T)
        moved_bboxes_global = copy.deepcopy(bboxes_global)
        moved_bboxes_global[:,:2] = moved_bboxes_global[:,:2] - 0.1*vels_global[:,:2]

        expand_bboxes_global = np.concatenate([bboxes_global[:,:3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        expand_moved_bboxes_global = np.concatenate([moved_bboxes_global[:,:3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]

        moved_bboxes_pre2cur = np.dot(expand_moved_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        vels_pre2cur = np.dot(vels_global, np.linalg.inv(pose_cur[:3,:3].T))[:,:2]
        bboxes_pre2cur = np.concatenate([bboxes_pre2cur, pred_boxes3d[:,3:7],vels_pre2cur],axis=-1)
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] + np.arctan2(pose_pre[1, 0], pose_pre[0,0])
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] - np.arctan2(pose_cur[1, 0], pose_cur[0,0])
        bboxes_pre2cur[:,7:9] = moved_bboxes_pre2cur[:,:2] - bboxes_pre2cur[:,:2]
        return bboxes_pre2cur[None,:,:]

    def generate_trajectory(self,cur_batch_boxes,proposals_list,batch_dict):
 
        trajectory_rois = cur_batch_boxes[:,None,:,:].repeat(1,batch_dict['rois'].shape[-2],1,1)
        trajectory_rois[:,0,:,:]= cur_batch_boxes
        valid_length = torch.zeros([batch_dict['batch_size'],batch_dict['rois'].shape[-2],trajectory_rois.shape[2]])
        valid_length[:,0] = 1
        num_frames = batch_dict['rois'].shape[-2]
        matching_table = (trajectory_rois.new_ones([trajectory_rois.shape[1],trajectory_rois.shape[2]]) * -1).long()

        for i in range(1,num_frames):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] + trajectory_rois[:,i-1,:,7:9]
            frame[:,:,2:] = trajectory_rois[:,i-1,:,2:]

            for bs_idx in range( batch_dict['batch_size']):
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame[bs_idx,:,:7], proposals_list[bs_idx,i,:,:7])
                max_overlaps, traj_assignment = torch.max(iou3d, dim=1)
                
                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                    
                valid_length[bs_idx,i,fg_inds] = 1
                matching_table[i,fg_inds] = traj_assignment[fg_inds]

                trajectory_rois[bs_idx,i,fg_inds,:] = proposals_list[bs_idx,i,traj_assignment[fg_inds]]

            batch_dict['valid_length'] = valid_length
        
        return trajectory_rois,valid_length, matching_table

    def get_proposal_aware_geometry_feature(self,src, batch_size,trajectory_rois,num_rois,batch_dict):
        
        i = 0 # only current frame
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,i,:,:].contiguous()) 

        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1]) 
        corner_points = corner_points.view(batch_size * num_rois, -1)
        trajectory_roi_center = trajectory_rois[:,i,:,:].contiguous().reshape(batch_size * num_rois, -1)[:,:3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim = -1)
        proposal_aware_feat = src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points,:3].repeat(1,1,9) - \
                                corner_add_center_points.unsqueeze(1).repeat(1,self.num_lidar_points,1) 

        lwh = trajectory_rois[:,i,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,proposal_aware_feat.shape[1],1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist = diag_dist.unsqueeze(-1))
        # proposal_aware_feat_list.append(proposal_aware_feat)

        # proposal_aware_feat = torch.cat(proposal_aware_feat_list,dim=1)
        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:,:,3:]], dim = -1)
        src_gemoetry = self.up_dimension_geometry(proposal_aware_feat) 
        proxy_point_geometry, proxy_points = self.roi_grid_pool(batch_size,trajectory_rois,src,src_gemoetry,batch_dict,batch_cnt=None)
        return proxy_point_geometry,proxy_points


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        if 'memory_bank' in batch_dict.keys():

            rois_list = []
            memory_list = copy.deepcopy(batch_dict['memory_bank'])

            for idx in range(len(memory_list['rois'])):

                rois = torch.cat([batch_dict['memory_bank']['rois'][idx][0],
                                    batch_dict['memory_bank']['roi_scores'][idx][0],
                                    batch_dict['memory_bank']['roi_labels'][idx][0]],-1)

                rois_list.append(rois)


            batch_rois = self.reorder_rois_for_refining(rois_list)
            batch_dict['roi_scores'] = batch_rois[None,:,:,9]
            batch_dict['roi_labels'] = batch_rois[None,:,:,10]
           
            proposals_list = []

            for i in range(self.model_cfg.Transformer.num_frames):
                pose_pre = batch_dict['poses'][0,i*4:(i+1)*4,:]
                pred2cur = self.transform_prebox_to_current_vel(batch_rois[i,:,:9].cpu().numpy(),pose_pre=pose_pre.cpu().numpy(),
                pose_cur=batch_dict['poses'][0,:4,:].cpu().numpy())
                proposals_list.append(torch.from_numpy(pred2cur).cuda().float())
            batch_rois = torch.cat(proposals_list,0)
            batch_dict['proposals_list'] = batch_rois[None,:,:,:9]

            batch_dict['rois'] = batch_rois.unsqueeze(0).permute(0,2,1,3)
            num_rois = batch_dict['rois'].shape[1]
            batch_dict['num_frames'] = batch_dict['rois'].shape[2]
            roi_labels_list = copy.deepcopy(batch_dict['roi_labels'])

            batch_dict['roi_scores'] = batch_dict['roi_scores'].permute(0,2,1)
            batch_dict['roi_labels'] = batch_dict['roi_labels'][:,0,:].long()
            proposals_list = batch_dict['proposals_list']
            batch_size = batch_dict['batch_size']
            cur_batch_boxes = copy.deepcopy(batch_dict['rois'].detach())[:,:,0]
            batch_dict['cur_frame_idx'] = 0

        else:
 
            batch_dict['rois'] = batch_dict['proposals_list'].permute(0,2,1,3)
            assert batch_dict['rois'].shape[0] ==1
            num_rois = batch_dict['rois'].shape[1]
            batch_dict['num_frames'] = batch_dict['rois'].shape[2]
            roi_labels_list = copy.deepcopy(batch_dict['roi_labels'])

            batch_dict['roi_scores'] = batch_dict['roi_scores'].permute(0,2,1)
            batch_dict['roi_labels'] = batch_dict['roi_labels'][:,0,:].long()
            proposals_list = batch_dict['proposals_list']
            batch_size = batch_dict['batch_size']
            cur_batch_boxes = copy.deepcopy(batch_dict['rois'].detach())[:,:,0]
            batch_dict['cur_frame_idx'] = 0

        trajectory_rois,effective_length,matching_table = self.generate_trajectory(cur_batch_boxes,proposals_list,batch_dict)


        batch_dict['has_class_labels'] = True
        batch_dict['trajectory_rois'] = trajectory_rois


        rois = batch_dict['rois']
        num_rois = batch_dict['rois'].shape[1]

        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            empty_mask = batch_dict['rois'][:,:,0,:6].sum(-1)==0
            batch_dict['valid_traj_mask'] = ~empty_mask

        num_sample = self.num_lidar_points 

        src = rois.new_zeros(batch_size, num_rois, num_sample, 5)

        src = self.crop_current_frame_points(src, batch_size, trajectory_rois, num_rois, num_sample, batch_dict)

        src = src.view(batch_size * num_rois, -1, src.shape[-1])

        src_geometry_feature,proxy_points = self.get_proposal_aware_geometry_feature(src,batch_size,trajectory_rois,num_rois,batch_dict)

        src_motion_feature = self.get_proposal_aware_motion_feature(proxy_points,batch_size,trajectory_rois,num_rois,batch_dict)


        if batch_dict['sample_idx'][0] >=1:

            src_repeat = src_geometry_feature[:,None,:64,:].repeat([1,trajectory_rois.shape[1],1,1])
            src_before = src_repeat[:,1:,:,:].clone() #[bs,traj,num_roi,C]
            valid_length = batch_dict['num_frames'] -1 if batch_dict['sample_idx'][0] > batch_dict['num_frames'] -1 \
                            else int(batch_dict['sample_idx'][0].item())
            num_max_rois = max(trajectory_rois.shape[2], *[i.shape[0] for i in batch_dict['memory_bank']['feature_bank']])
            feature_bank = self.reorder_memory(batch_dict['memory_bank']['feature_bank'][:valid_length],num_max_rois)
            effective_length = effective_length[0,1:1+valid_length].bool() #rm dim of bs
            for i in range(valid_length):
                src_before[:,i][effective_length[i]] = feature_bank[i,matching_table[1+i][effective_length[i]]]

            src_geometry_feature = torch.cat([src_repeat[:,:1],src_before],1).view(src_geometry_feature.shape[0],-1,
                                              src_geometry_feature.shape[-1])

        else:

            src_geometry_feature = src_geometry_feature.repeat([1,trajectory_rois.shape[1],1])

        batch_dict['geometory_feature_memory'] = src_geometry_feature[:,:64]


        src = src_geometry_feature + src_motion_feature

        
        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            src[empty_mask.view(-1)] = 0

        if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
            pos = self.grid_pos_embeded(self.grid_index.cuda())[None,:,:]
            pos = torch.cat([torch.zeros(1,1,self.hidden_dim).cuda(),pos],1)
        else:
            pos=None

        hs, tokens = self.transformer(src,pos=pos)
        point_cls_list = []

        for i in range(3):
            point_cls_list.append(self.class_embed[0](tokens[i][0]))

        point_cls = torch.cat(point_cls_list,0)


        hs = hs.permute(1,0,2).reshape(hs.shape[1],-1)
    
        _, feat_box = self.trajectories_auxiliary_branch(trajectory_rois)

        joint_reg = self.jointembed(torch.cat([hs,feat_box],-1))

        rcnn_cls = point_cls
        rcnn_reg = joint_reg

        if not self.training:
            batch_dict['rois'] = batch_dict['rois'][:,:,0].contiguous()
            rcnn_cls = rcnn_cls[-rcnn_cls.shape[0]//self.num_enc_layer:]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            batch_dict['cls_preds_normalized'] = False
            if self.avg_stage1_score:
                stage1_score = batch_dict['roi_scores'][:,:,:1]
                batch_cls_preds = F.sigmoid(batch_cls_preds)
                if self.model_cfg.get('IOU_WEIGHT', None):
                    batch_box_preds_list = []
                    roi_labels_list = []
                    batch_cls_preds_list = []
                    for bs_idx in range(batch_size):
                        car_mask = batch_dict['roi_labels'][bs_idx] ==1
                        batch_cls_preds_car = batch_cls_preds[bs_idx].pow(self.model_cfg.IOU_WEIGHT[0])* \
                                              stage1_score[bs_idx].pow(1-self.model_cfg.IOU_WEIGHT[0])
                        batch_cls_preds_car = batch_cls_preds_car[car_mask][None]
                        batch_cls_preds_pedcyc = batch_cls_preds[bs_idx].pow(self.model_cfg.IOU_WEIGHT[1])* \
                                                 stage1_score[bs_idx].pow(1-self.model_cfg.IOU_WEIGHT[1])
                        batch_cls_preds_pedcyc = batch_cls_preds_pedcyc[~car_mask][None]
                        cls_preds = torch.cat([batch_cls_preds_car,batch_cls_preds_pedcyc],1)
                        box_preds = torch.cat([batch_dict['batch_box_preds'][bs_idx][car_mask],
                                                     batch_dict['batch_box_preds'][bs_idx][~car_mask]],0)[None]
                        roi_labels = torch.cat([batch_dict['roi_labels'][bs_idx][car_mask],
                                                batch_dict['roi_labels'][bs_idx][~car_mask]],0)[None]
                        batch_box_preds_list.append(box_preds)
                        roi_labels_list.append(roi_labels)
                        batch_cls_preds_list.append(cls_preds)
                    batch_dict['batch_box_preds'] = torch.cat(batch_box_preds_list,0)
                    batch_dict['roi_labels'] = torch.cat(roi_labels_list,0)
                    batch_cls_preds = torch.cat(batch_cls_preds_list,0)
                    
                else:
                    batch_cls_preds = torch.sqrt(batch_cls_preds*stage1_score)
                batch_dict['cls_preds_normalized']  = True

            batch_dict['batch_cls_preds'] = batch_cls_preds

        return batch_dict

    def reorder_memory(self, memory,num_max_rois):

        ordered_memory = memory[0].new_zeros([len(memory),num_max_rois,memory[0].shape[1],memory[0].shape[2]])
        for bs_idx in range(len(memory)):
            ordered_memory[bs_idx,:len(memory[bs_idx])] = memory[bs_idx]
        return ordered_memory



    def generate_predicted_boxes(self, batch_size, rois, cls_preds=None, box_preds=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)
        Returns:
        """
        code_size = self.box_coder.code_size

        if cls_preds is not None:
            batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        else:
            batch_cls_preds = None
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds,rois[:,:,7:]/1.5],-1) #for superbox
        return batch_cls_preds, batch_box_preds