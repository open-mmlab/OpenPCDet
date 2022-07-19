from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import itertools
from numpy import *
import copy
import time
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import torch.nn.functional as F
from ...utils import box_coder_utils, common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.mppnet_utils import build_transformer

import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules


class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1,outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel==256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(pts_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x,  self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(256 * x)
        self.bn4 = nn.BatchNorm1d(self.output_channel)



    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x)) 

        x = torch.max(x_ori, 2, keepdim=True)[0]

        x = x.view(-1, self.output_channel)
        return x, x_ori

class PointNet(nn.Module):
    def __init__(self, pts_dim, x, CLS_NUM,joint_feat=False,model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT

        if joint_feat:
            self.feat = None
            times=5

            self.fc_s1 = nn.Linear(channels*times, 256)
            self.fc_s2 = nn.Linear(256, 3, bias=False)
            self.fc_c1 = nn.Linear(channels*times, 256)
            self.fc_c2 = nn.Linear(256, CLS_NUM, bias=False)
            self.fc_ce1 = nn.Linear(channels*times, 256)
            self.fc_ce2 = nn.Linear(256, 3, bias=False)
            self.fc_hr1 = nn.Linear(channels*times, 256)
            self.fc_hr2 = nn.Linear(256, 1, bias=False)

        else:
            # channels=256
            times=1
            self.feat = PointNetfeat(pts_dim, x)

            self.fc1 = nn.Linear(512 * x, 256 * x)
            self.fc2 = nn.Linear(256 * x, channels)

            self.pre_bn = nn.BatchNorm1d(pts_dim)
            self.bn1 = nn.BatchNorm1d(256 * x)
            self.bn2 = nn.BatchNorm1d(channels)
            self.relu = nn.ReLU()

            self.fc_s1 = nn.Linear(channels*times, 256)
            self.fc_s2 = nn.Linear(256, 3, bias=False)
            self.fc_c1 = nn.Linear(channels*times, 256)
            self.fc_c2 = nn.Linear(256, CLS_NUM, bias=False)
            self.fc_ce1 = nn.Linear(channels*times, 256)
            self.fc_ce2 = nn.Linear(256, 3, bias=False)
            self.fc_hr1 = nn.Linear(channels*times, 256)
            self.fc_hr2 = nn.Linear(256, 1, bias=False)

    def forward(self, x, feat=None):

        if self.joint_feat:
            if len(feat.shape) > 2:
                feat = torch.max(feat, 2, keepdim=True)[0]
                x = feat.view(-1, self.output_channel)
                x = F.relu(self.bn1(self.fc1(x)))
                feat = F.relu(self.bn2(self.fc2(x)))
            else:
                feat = feat
            feat_traj = None
        else:
            x, feat_traj = self.feat(self.pre_bn(x))
            x = F.relu(self.bn1(self.fc1(x)))
            feat = F.relu(self.bn2(self.fc2(x)))

        x = F.relu(self.fc_c1(feat))
        logits = self.fc_c2(x)

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return logits, torch.cat([centers, sizes, headings],-1),feat,feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MPPNetHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, voxel_size, point_cloud_range, num_class=1,**kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.time_stamp = self.model_cfg.get('USE_TIMESTAMP',None)
        self.num_points = self.model_cfg.Transformer.num_points
        self.avg_stage1 = self.model_cfg.get('AVG_STAGE_1', None)

        self.nhead = model_cfg.Transformer.nheads
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.num_groups = model_cfg.Transformer.num_groups

        self.grid_size = model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.num_grid_points = model_cfg.ROI_GRID_POOL.GRID_SIZE**3
        self.num_key_points = self.num_grid_points
        # import pdb;pdb.set_trace()
        self.seqboxemb = PointNet(8,1,1,model_cfg=self.model_cfg)
        if self.model_cfg.get('USE_MLP_JOINTEMB',None):
            self.jointemb = MLP(self.hidden_dim*5, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4)
        else:
            self.jointemb = PointNet(1,1,1,joint_feat=True,model_cfg=self.model_cfg)


        dim=30 

        self.up_dimension_voxel = MLP(input_dim = dim, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)

        input_dim = 29 
        if self.model_cfg.Transformer.num_frames==4:
            self.up_dimension = MLP(input_dim = input_dim, hidden_dim = 64, output_dim =hidden_dim//2, num_layers = 3)
        else:
            self.up_dimension = MLP(input_dim = input_dim, hidden_dim = 64, output_dim =hidden_dim, num_layers = 3)

        self.transformer = build_transformer(model_cfg.Transformer)

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
                nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
                mlps=self.model_cfg.ROI_GRID_POOL.MLPS,
                use_xyz=True,
                pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
                )

            
                    

        self.class_embed = nn.ModuleList()
        self.bbox_embed = nn.ModuleList()
        self.num_pred = 4 

        for i in range(self.num_pred):
            self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim, 1))
            self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4))

        if self.model_cfg.Transformer.use_grid_pos.enabled:
            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                self.gridindex = torch.cat([i.reshape(-1,1)for i in torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size), torch.arange(self.grid_size))],1).float().cuda()

                self.gridposembeding = MLP(input_dim = 3, hidden_dim = 256, output_dim = hidden_dim, num_layers = 2)

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
        # pdb.set_trace()

        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
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

    def roi_grid_pool(self, batch_size, rois, point_coords, point_features,batch_dict=None,batch_cnt=None,batch_voxel_cnt=None,src=None,effi=False):


        num_frames = batch_dict['num_frames']
        num_rois = rois.shape[2]*rois.shape[1]


        if len(point_coords.shape)==3:
            global_roi_grid_points, local_roi_grid_points = self.get_grid_points_of_roi(
                rois.permute(0,2,1,3).contiguous(), grid_size=self.grid_size
            )  # (BxN, 6x6x6, 3)
        else:
            global_roi_grid_points, local_roi_grid_points = self.get_grid_points_of_roi(
                rois, grid_size=self.grid_size
            )  # (BxN, 6x6x6, 3)
        # order [xxxxyyyyzzzz ....] not [xyz...,xyz..xyz..xyz]

        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)
  
        if len(point_coords.shape)==3:
            point_coords = point_coords.view(point_coords.shape[0]*num_frames,point_coords.shape[1]//num_frames,point_coords.shape[-1])
            xyz = point_coords[:, :, 0:3].view(-1,3)
        else:
            xyz = point_coords[:,:3]

        
        num_points = point_coords.shape[1]
        num_key_points = self.num_key_points

        if batch_cnt is None:
            xyz_batch_cnt = torch.tensor([num_points]*num_rois*batch_size).cuda().int()
        else:
            xyz_batch_cnt = torch.tensor(batch_cnt).cuda().int()

        new_xyz_batch_cnt = torch.tensor([num_key_points]*num_rois*batch_size).cuda().int()
        new_xyz = global_roi_grid_points.view(-1, 3)

        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.view(-1,point_features.shape[-1]).contiguous(),
        )  # (M1 + M2 ..., C)

        if len(point_coords.shape)==3:
            if effi:
                features = pooled_features.view(
                    point_features.shape[0], self.num_key_points,
                    pooled_features.shape[-1]
                ).contiguous()  # (BxN, 6x6x6, C)
            else:
                features = pooled_features.view(
                    point_features.shape[0], num_frames*self.num_key_points,
                    pooled_features.shape[-1]
                ).contiguous()  # (BxN, 6x6x6, C)


        elif pooled_features.shape[-1]==256:
            features = pooled_features.view(batch_size, num_frames, rois.shape[2],num_key_points,256).permute(0,2,1,3,4).contiguous().view(-1,num_frames*num_key_points,256)
        else:
            features = pooled_features.view(batch_size, num_frames, rois.shape[2],num_key_points,60).permute(0,2,1,3,4).contiguous() 
            features = features.view(batch_size*rois.shape[2], num_frames*num_key_points,2,30)
            features = features.view(batch_size*rois.shape[2], num_frames*num_key_points*2,30).contiguous()  # (BxN, 6x6x6, C)
        return features,global_roi_grid_points.view(batch_size*rois.shape[2], num_frames*num_key_points,3).contiguous(), None

    def get_grid_points_of_roi(self, rois, grid_size):

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

    def transform_prebox2current(self,pred_boxes3d,pose_pre,pose_cur):

        expand_bboxes = np.concatenate([pred_boxes3d[:,:3], np.ones((pred_boxes3d.shape[0], 1))], axis=-1)
        bboxes_global = np.dot(expand_bboxes, pose_pre.T)[:, :3]
        expand_bboxes_global = np.concatenate([bboxes_global[:,:3],np.ones((bboxes_global.shape[0], 1))], axis=-1)
        bboxes_pre2cur = np.dot(expand_bboxes_global, np.linalg.inv(pose_cur.T))[:, :3]
        bboxes_pre2cur = np.concatenate([bboxes_pre2cur, pred_boxes3d[:,3:9]],axis=-1)
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] + np.arctan2(pose_pre[..., 1, 0], pose_pre[..., 0,0])
        bboxes_pre2cur[:,6]  = bboxes_pre2cur[..., 6] - np.arctan2(pose_cur[..., 1, 0], pose_cur[..., 0,0])

        return torch.from_numpy(bboxes_pre2cur).cuda()

    def crop_current_frame_points(self, src, batch_size,trajectory_rois,num_rois,num_sample, batch_dict):

        for bs_idx in range(batch_size):

            cur_batch_boxes = trajectory_rois[bs_idx,0,:,:7].view(-1,7)
            cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.1
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
            dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
            point_mask = (dis <= cur_radiis.unsqueeze(-1))
            time_mask_list = []

            # sampled_idx = torch.topk(mask.float(),128)[1]
            # sampled_idx_buffer = sampled_idx[:, 0:1].repeat(1, 128)  # (num_rois, npoints)
            # roi_idx = torch.arange(num_rois)[:, None].repeat(1, 128)
            # sampled_mask = mask[roi_idx, sampled_idx]  # (num_rois, 128)
            # sampled_idx_buffer[sampled_mask] = sampled_idx[sampled_mask]

            # src = cur_points[sampled_idx_buffer][:,:,:5] # (num_rois, npoints)
            # empty_flag = sampled_mask.sum(-1)==0
            # src[empty_flag] = 0

            for roi_box_idx in range(0, num_rois):
                start3 = time.time()
                cur_roi_points = cur_points[point_mask[roi_box_idx]]
                time_mask = cur_roi_points[:,-1].abs() < 1e-3
                cur_roi_points = cur_roi_points[time_mask]
                time_mask_list.append(time.time() - start3)
                if cur_roi_points.shape[0] > self.num_points:
                    # random.seed(0)
                    # choice = np.random.choice(cur_roi_points.shape[0], self.num_points, replace=True)
                    # cur_roi_points_sample = cur_roi_points[choice]
                    cur_roi_points_sample = cur_roi_points[:128]

                elif cur_roi_points.shape[0] == 0:
                    cur_roi_points_sample = cur_roi_points.new_zeros(self.num_points, 6)
                    batch_dict['nonempty_mask'][bs_idx, roi_box_idx] = False

                else:
                    empty_num = num_sample - cur_roi_points.shape[0]
                    add_zeros = cur_roi_points.new_zeros(empty_num, 6)
                    add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                    cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                if not self.time_stamp:
                    cur_roi_points_sample = cur_roi_points_sample[:,:-1]

                src[bs_idx, roi_box_idx, :self.num_points, :] = cur_roi_points_sample

        src = src.repeat([1,1,trajectory_rois.shape[1],1])

        return src

    def crop_previous_frame_points(self,src,batch_size,trajectory_rois,num_rois,effective_length,batch_dict):
        for bs_idx in range(batch_size):
            
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]


            for idx in range(1,trajectory_rois.shape[1]):
            
                time_mask = (cur_points[:,-1] - idx*0.1).abs() < 1e-3
                cur_time_points = cur_points[time_mask]
                cur_batch_boxes = trajectory_rois[bs_idx,idx,:,:7].view(-1,7)

                cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.1
                if not self.training and cur_batch_boxes.shape[0] > 32:
                    length_iter= cur_batch_boxes.shape[0]//32
                    dis_list = []
                    for i in range(length_iter+1):
                        dis = torch.norm((cur_time_points[:,:2].unsqueeze(0) - cur_batch_boxes[32*i:32*(i+1),:2].unsqueeze(1).repeat(1,cur_time_points.shape[0],1)), dim = 2)
                        dis_list.append(dis)
                    dis = torch.cat(dis_list,0)
                else:
                    dis = torch.norm((cur_time_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_time_points.shape[0],1)), dim = 2)
                
                point_mask = (dis <= cur_radiis.unsqueeze(-1)).view(trajectory_rois.shape[2],-1)

                for roi_box_idx in range(0, num_rois):
                # for idx in range(1,trajectory_rois.shape[1]):
                    if not effective_length[bs_idx,idx,roi_box_idx]:
                            continue

                    cur_roi_points = cur_time_points[point_mask[roi_box_idx]]



                    if cur_roi_points.shape[0] > self.num_points:
                        # random.seed(0)
                        # choice = np.random.choice(cur_roi_points.shape[0], self.num_points, replace=True)
                        # cur_roi_points_sample = cur_roi_points[choice]
                        cur_roi_points_sample = cur_roi_points[:128]

                    elif cur_roi_points.shape[0] == 0:
                        cur_roi_points_sample = cur_roi_points.new_zeros(self.num_points, 6)
                        batch_dict['nonempty_mask'][bs_idx, roi_box_idx] = False

                    else:
                        empty_num = self.num_points - cur_roi_points.shape[0]
                        add_zeros = cur_roi_points.new_zeros(empty_num, 6)
                        add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                        cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                

                    if not self.time_stamp:
                        cur_roi_points_sample = cur_roi_points_sample[:,:-1]

                    src[bs_idx, roi_box_idx, self.num_points*idx:self.num_points*(idx+1), :] = cur_roi_points_sample

        return src

    def get_proposal_aware_geometry_feature(self,src, batch_size,trajectory_rois,num_rois,batch_dict):
        pos_fea_list = []
        for i in range(trajectory_rois.shape[1]):

            corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,i,:,:].contiguous())  # (BxN, 2x2x2, 3)

            corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)
            corner_points = corner_points.view(batch_size * num_rois, -1)
            corner_add_center_points = torch.cat([corner_points, trajectory_rois[:,i,:,:].contiguous().reshape(batch_size * num_rois, -1)[:,:3]], dim = -1)
            pos_fea = src[:,i*self.num_points:(i+1)*self.num_points,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1).repeat(1,self.num_points,1)  # 27 维
            pos_fea_ori = pos_fea

            if self.model_cfg.get('USE_SPHERICAL_COOR',True):
                lwh = trajectory_rois[:,i,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,pos_fea.shape[1],1)
                diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
                pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))

            
            pos_fea_list.append(pos_fea)
            # pos_fea_ori_list.append(pos_fea_ori)

        pos_fea = torch.cat(pos_fea_list,dim=1)
        src_res = torch.cat([pos_fea, src[:,:,3:]], dim = -1)
        src_gemoetry = self.up_dimension(src_res) # [bs,num_points,num_feat]
        proxy_point_geometry, proxy_points, _ = self.roi_grid_pool(batch_size,trajectory_rois,src,src_gemoetry,batch_dict,batch_cnt=None)
        return proxy_point_geometry,proxy_points

    def get_proposal_aware_motion_feature(self,proxy_point,batch_size,trajectory_rois,num_rois,batch_dict):


        time_stamp   = torch.ones([proxy_point.shape[0],proxy_point.shape[1],1]).cuda()
        padding_zero = torch.zeros([proxy_point.shape[0],proxy_point.shape[1],2]).cuda()
        proxy_point_padding = torch.cat([padding_zero,time_stamp],-1)

        num_time_coding = trajectory_rois.shape[1]

        for i in range(num_time_coding):
            proxy_point_padding[:,i*self.num_key_points:(i+1)*self.num_key_points,-1] = i*0.1


        ######### use T0 Norm ########
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,0,:,:].contiguous())  # (BxN, 2x2x2, 3)
        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)
        corner_points = corner_points.view(batch_size * num_rois, -1)
        corner_add_center_points = torch.cat([corner_points, trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,:3]], dim = -1)

        pos_fea = proxy_point[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1) # 

        lwh = trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,proxy_point.shape[1],1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        ######### use T0 Norm ########

        proxy_point_padding = torch.cat([pos_fea,proxy_point_padding],-1)
        proxy_point_motion_feat = self.up_dimension_voxel(proxy_point_padding)

        return proxy_point_motion_feat

    def trajectories_auxiliary_branch(self,trajectory_rois):

        time_stamp = torch.ones([trajectory_rois.shape[0],trajectory_rois.shape[1],trajectory_rois.shape[2],1]).cuda()
        for i in range(time_stamp.shape[1]):
            time_stamp[:,i,:] = i*0.1 

        box_seq = torch.cat([trajectory_rois[:,:,:,:7],time_stamp],-1)

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

        box_cls,  box_reg, feat_traj, _ = self.seqboxemb(box_seq.permute(0,2,3,1).contiguous().view(batch_rcnn,box_seq.shape[-1],box_seq.shape[1]))
        
        return box_cls, box_reg, feat_traj

    def generate_trajectory(self,cur_batch_boxes,proposals_list,roi_labels_list,roi_scores_list,batch_dict):
        frame1 = cur_batch_boxes
        trajectory_rois = cur_batch_boxes[:,None,:,:].repeat(1,batch_dict['rois'].shape[-2],1,1)
        trajectory_roi_scores = torch.zeros_like(batch_dict['roi_scores'].permute(0,2,1))
        trajectory_rois[:,0,:,:]= frame1
        trajectory_roi_scores[:,0,:] = batch_dict['roi_scores'][:,:,0]
        effective_length = torch.zeros([batch_dict['batch_size'],batch_dict['rois'].shape[-2],trajectory_rois.shape[2]])
        effective_length[:,0] = 1
        for i in range(1,batch_dict['rois'].shape[-2]):
            frame = torch.zeros_like(cur_batch_boxes)
            frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] + trajectory_rois[:,i-1,:,7:9]
            frame[:,:,2:] = trajectory_rois[:,i-1,:,2:]

            for idx in range( batch_dict['batch_size']):
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame[idx][:,:7], proposals_list[idx,i,:,:7])
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
                
                fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                    
                effective_length[idx,i,fg_inds] = 1

                trajectory_rois[idx,i,:,:][fg_inds] = proposals_list[idx][i][gt_assignment[fg_inds]]
                trajectory_roi_scores[idx,i,:][fg_inds] = roi_scores_list[idx][i][gt_assignment[fg_inds]]

            batch_dict['effi_length'] = effective_length
        
        return trajectory_rois,trajectory_roi_scores,effective_length

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        start_time = time.time()
        batch_dict['rois'] = batch_dict['proposals_list'].permute(0,2,1,3)#[:,:,[4,3,0,2,1]]
        num_rois = batch_dict['rois'].shape[1]
        batch_dict['num_frames'] = batch_dict['rois'].shape[2]
        roi_scores_list = copy.deepcopy(batch_dict['roi_scores'])
        roi_labels_list = copy.deepcopy(batch_dict['roi_labels'])

        batch_dict['roi_scores'] = batch_dict['roi_scores'].permute(0,2,1)
        batch_dict['roi_labels'] = batch_dict['roi_labels'][:,0,:].long()
        proposals_list = batch_dict['proposals_list']
        batch_size = batch_dict['batch_size']
        cur_batch_boxes = copy.deepcopy(batch_dict['rois'].detach())[:,:,0]
        batch_dict['cur_frame_idx'] = 0
        batch_dict['matching_index'] = []
        batch_dict['fg_inds'] = []
    
        trajectory_rois,trajectory_roi_scores,effective_length \
             = self.generate_trajectory(cur_batch_boxes,proposals_list,roi_labels_list,roi_scores_list,batch_dict)

        batch_dict['traj_memory'] = trajectory_rois
        batch_dict['has_class_labels'] = True
        batch_dict['trajectory_rois'] = trajectory_rois
        batch_dict['trajectory_roi_scores'] = trajectory_roi_scores
        batch_dict['gt_boxes'] = batch_dict['gt_boxes']

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois'].permute(0,2,1,3).contiguous()
            targets_dict['rois']  = targets_dict['rois'][:,batch_dict['cur_frame_idx']]
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            targets_dict['trajectory_rois'][:,batch_dict['cur_frame_idx'],:,:] = batch_dict['rois'][:,:,batch_dict['cur_frame_idx'],:]
            trajectory_rois = targets_dict['trajectory_rois']
            effective_length = targets_dict['effi_length']


        rois = batch_dict['rois']
        num_rois = batch_dict['rois'].shape[1]
        batch_dict['traj_time'] = time.time() - start_time
        start_time = time.time()
        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            empty_mask = batch_dict['rois'][:,:,0,:6].sum(-1)==0
            batch_dict['valid_traj_mask'] = ~empty_mask
            if self.training:
                targets_dict['empty_mask'] = empty_mask

        num_sample = self.num_points 

        src = rois.new_zeros(batch_size, num_rois, num_sample, 5) # For waymo, the last dim is not used

        batch_dict['nonempty_mask'] = rois.new_ones(batch_size,num_rois).bool()

        src = self.crop_current_frame_points(src, batch_size, trajectory_rois, num_rois, num_sample, batch_dict)

        src = self.crop_previous_frame_points(src, batch_size,trajectory_rois,num_rois,effective_length,batch_dict)

        src = src.view(batch_size * num_rois, -1, src.shape[-1])  # (b*128, 256, 6)

        src_geometry_feature,proxy_points = self.get_proposal_aware_geometry_feature(src,batch_size,trajectory_rois,num_rois,batch_dict)

        src_motion_feature = self.get_proposal_aware_motion_feature(proxy_points,batch_size,trajectory_rois,num_rois,batch_dict)

        src = src_geometry_feature + src_motion_feature

        box_cls,  box_reg, feat_box = self.trajectories_auxiliary_branch(trajectory_rois)
        
        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            src[empty_mask.view(-1)] = 0

        if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
            pos = self.gridposembeding(self.gridindex.cuda())[None,:,:]
            pos = torch.cat([torch.zeros(1,1,self.hidden_dim).cuda(),pos],1)

        else:
            grid_pos = None
            pos=None

        start_time = time.time()
        hs, tokens, mlp_merge = self.transformer(src,pos=pos, num_frames = batch_dict['num_frames'])
        batch_dict['casa_time'] = time.time() - start_time
        head_time = time.time()
        point_cls_list = []
        point_reg_list = []


        
        # index_list = [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
        # for i in range(3):
        #     point_cls_list.append(self.class_embed[0](tokens[index_list[0][i]]))


        # for i in range(hs.shape[0]):
        #     for j in range(3):
        #         point_reg_list.append(self.bbox_embed[i](tokens[index_list[i][j]]))



        # point_cls = torch.cat(point_cls_list,0)
        # point_reg = torch.cat(point_reg_list,0)

        # # if self.model_cfg.Transformer.joint_dim==1024:
        # hs = hs.permute(1,0,2).reshape(hs.shape[1],-1)
        

        for i in range(3):
            point_cls_list.append(self.class_embed[0](tokens[i][0]))

        for i in range(hs.shape[0]):
            for j in range(3):
                point_reg_list.append(self.bbox_embed[i](tokens[j][i]))

        point_cls = torch.cat(point_cls_list,0)

        point_reg = torch.cat(point_reg_list,0)
        hs = hs.permute(1,0,2).reshape(hs.shape[1],-1)
        
        if self.model_cfg.get('USE_MLP_JOINTEMB',None):
            joint_cls = None
            joint_reg = self.jointemb(torch.cat([hs,feat_box],-1))
        else:
            joint_cls, joint_reg, _, _ = self.jointemb(None,torch.cat([hs,feat_box],-1))
        
        if self.model_cfg.get('USE_POINT_AS_JOINT_CLS',True):
            rcnn_cls = point_cls
        else:
            rcnn_cls = joint_cls

        rcnn_reg = joint_reg

        if not self.training:
            batch_dict['rois'] = batch_dict['rois'][:,:,0].contiguous()
            
            rcnn_cls = rcnn_cls[-rcnn_cls.shape[0]//3:]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            batch_dict['cls_preds_normalized'] = False
            if self.avg_stage1:
                stage1_score = batch_dict['roi_scores'][:,:,:1]
                batch_cls_preds = F.sigmoid(batch_cls_preds)
                if self.model_cfg.get('IOU_WEIGHT', None):
                    car_mask = batch_dict['roi_labels'] ==1
                    batch_cls_preds_car = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[0])*stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[0])
                    batch_cls_preds_car = batch_cls_preds_car[car_mask][None]
                    batch_cls_preds_pedcyc = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[1])*stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[1])
                    batch_cls_preds_pedcyc = batch_cls_preds_pedcyc[~car_mask][None]
                    batch_cls_preds = torch.cat([batch_cls_preds_car,batch_cls_preds_pedcyc],1)
                    batch_box_preds = torch.cat([batch_dict['batch_box_preds'][car_mask],batch_dict['batch_box_preds'][~car_mask]],0)[None]
                    batch_dict['batch_box_preds'] = batch_box_preds.view(batch_size, -1, batch_box_preds.shape[-1])
                    roi_labels = torch.cat([batch_dict['roi_labels'][car_mask],batch_dict['roi_labels'][~car_mask]],0)[None]
                    batch_dict['roi_labels'] = roi_labels.view(batch_size, -1)
                    batch_cls_preds = batch_cls_preds.view(batch_size, -1, 1)
                    
                else:
                    batch_cls_preds = torch.sqrt(batch_cls_preds*stage1_score)
                batch_dict['cls_preds_normalized']  = True
            batch_dict['batch_cls_preds'] = batch_cls_preds
                

        else:
            targets_dict['batch_size'] = batch_size
            targets_dict['nonempty_mask'] = batch_dict['nonempty_mask']
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg


            if self.model_cfg.USE_BOX_ENCODING.ENABLED:
                targets_dict['box_reg'] = box_reg
                targets_dict['box_cls'] = box_cls
                targets_dict['point_reg'] = point_reg
                targets_dict['point_cls'] = point_cls
            self.forward_ret_dict = targets_dict
        batch_dict['transformer_time'] = time.time()- start_time
        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        if self.model_cfg.USE_CLS_LOSS:
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_cls
            tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        tb_dict['num_pos'] = self.forward_ret_dict['num_pos']
        # import pdb;pdb.set_trace()
        return rcnn_loss, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        nonempty_mask = forward_ret_dict['nonempty_mask'].view(-1)
        batch_size = forward_ret_dict['batch_size']

        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)

        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
            roi_boxes3d = forward_ret_dict['trajectory_rois']
        else:
            roi_boxes3d = forward_ret_dict['rois']

        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):

            if self.model_cfg.get('USE_NONEMPTY_MASK',None):
                fg_mask_t0 = (forward_ret_dict['reg_valid_mask'][:,0] > 0).view(-1) & nonempty_mask
                fg_sum_t0 = fg_mask_t0.long().sum().item()
            else:
                fg_mask_t0 = (forward_ret_dict['reg_valid_mask'][:,0] > 0).view(-1)
                fg_sum_t0 = fg_mask_t0.long().sum().item()
            
            fg_mask = (reg_valid_mask > 0)
            fg_sum = fg_mask.long().sum().item()
        
        else:
            if self.model_cfg.get('USE_NONEMPTY_MASK',None):
                fg_mask = (reg_valid_mask > 0) & nonempty_mask
                fg_sum = fg_mask.long().sum().item()
            else:
                fg_mask = (reg_valid_mask > 0)
                fg_sum = fg_mask.long().sum().item()

        

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':


            if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
             
                rois_anchor = roi_boxes3d.clone().detach()[:,:,:,:7].contiguous().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                
                reg_targets = self.box_coder.encode_torch(
                    gt_boxes3d_ct.view(-1 , code_size), rois_anchor
                    )

                reg_targets_shape = reg_targets.view(batch_size,-1,code_size)
                reg_targets_t0 = reg_targets_shape[:,:reg_targets_shape.shape[1]//4,:].contiguous().view(-1, code_size).unsqueeze(dim=0)

                rcnn_loss_reg = self.reg_loss_func(
                    rcnn_reg.unsqueeze(dim=0),
                    reg_targets_t0,
                )  # [B, M, 7]
                rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size//4, -1) * fg_mask_t0.unsqueeze(dim=-1).float()).sum() / max(fg_sum_t0, 1)
                rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['usebox_reg_weight'][0]

            else:

                rois_anchor = roi_boxes3d.clone().detach()[:,:,:7].contiguous().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                reg_targets = self.box_coder.encode_torch(
                    gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
                    )
                rcnn_loss_reg = self.reg_loss_func(
                    rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0),
                )  # [B, M, 7]
                rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['usebox_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
  
            if self.model_cfg.USE_AUG_LOSS:
                point_reg = forward_ret_dict['point_reg']  # (rcnn_batch_size, C)

                groups = point_reg.shape[0]//reg_targets.shape[0]
                if groups != 1 :
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                        point_reg[i*slice:(i+1)*slice].view(slice, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)  # [B, M, 7]
                        point_loss_reg = (point_loss_reg.view(slice, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                        point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['usebox_reg_weight'][2]
                        
                        point_loss_regs += point_loss_reg
                    point_loss_regs = point_loss_regs / groups
                    tb_dict['point_loss_reg'] = point_loss_regs.item()
                    rcnn_loss_reg += point_loss_regs 

                else:
                    point_loss_reg = self.reg_loss_func(point_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)  # [B, M, 7]
                    point_loss_reg = (point_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                    point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['usebox_reg_weight'][2]
                    tb_dict['point_loss_reg'] = point_loss_reg.item()
                    rcnn_loss_reg += point_loss_reg

                if self.model_cfg.USE_BOX_ENCODING.ENABLED:
                    seqbox_reg = forward_ret_dict['box_reg']  # (rcnn_batch_size, C)
                    if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
                        seqbox_loss_reg = self.reg_loss_func(seqbox_reg.unsqueeze(dim=0),reg_targets_t0,)  # [B, M, 7]
                        seqbox_loss_reg = (seqbox_loss_reg.view(-1, code_size) * fg_mask_t0.unsqueeze(dim=-1).float()).sum() / max(fg_sum_t0, 1)
                    else:
                        seqbox_loss_reg = self.reg_loss_func(seqbox_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)
                        seqbox_loss_reg = (seqbox_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                    seqbox_loss_reg = seqbox_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['usebox_reg_weight'][1]
                    tb_dict['seqbox_loss_reg'] = seqbox_loss_reg.item()
                    rcnn_loss_reg += seqbox_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
                    fg_rcnn_reg = rcnn_reg[fg_mask_t0]
                    fg_roi_boxes3d = roi_boxes3d[:,0,:,:7].contiguous().view(-1, code_size)[fg_mask_t0]
                else:
                    fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                    fg_roi_boxes3d = roi_boxes3d[:,:,:7].contiguous().view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                if self.model_cfg.get('FIX_CORNER_BUG',None):
                    corner_loss_func = loss_utils.get_corner_loss_lidar_v2
                else:
                    corner_loss_func = loss_utils.get_corner_loss_lidar_v1

                if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
                    loss_corner = corner_loss_func(
                        rcnn_boxes3d[:, 0:7],
                        gt_of_rois_src.view(batch_size,-1,gt_of_rois_src.shape[-1])[:,:num_rois,:].contiguous().view(-1,gt_of_rois_src.shape[-1])[fg_mask_t0][:, 0:7]
                    )
                else:
                    loss_corner = corner_loss_func(
                        rcnn_boxes3d[:, 0:7],
                        gt_of_rois_src[fg_mask][:, 0:7])

                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']



                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()

                if self.model_cfg.get('USE_POINT_CORNER_LOSS',None):

                    point_reg = forward_ret_dict['point_reg']  # (rcnn_batch_size, C)
                    fg_point_reg = point_reg.view(rcnn_batch_size, -1)[fg_mask]
                    fg_roi_boxes3d = roi_boxes3d[:,:,:7].contiguous().view(-1, code_size)[fg_mask]

                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    point_boxes3d = self.box_coder.decode_torch(
                        fg_point_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors.repeat(1,4,1)
                    ).view(-1, code_size)

                    point_boxes3d = common_utils.rotate_points_along_z(point_boxes3d.unsqueeze(dim=1), roi_ry.repeat(4)).squeeze(dim=1)
                    point_boxes3d[:, 0:3] += roi_xyz.repeat(4,1)

                    
                    point_loss_corner = corner_loss_func(point_boxes3d[:, 0:7],gt_of_rois_src[fg_mask][:, 0:7].repeat(4,1))  # [B, M, 7]

                    point_loss_corner = point_loss_corner.mean()
                    point_loss_corner = point_loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                    rcnn_loss_reg += point_loss_corner
                    tb_dict['point_loss_corner'] = point_loss_corner.item()

                    

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        # nonempty_mask = forward_ret_dict['nonempty_mask'].view(-1)
        if self.model_cfg.USE_BOX_ENCODING.ENABLED:
            point_cls = forward_ret_dict['point_cls']

        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)


        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':

            rcnn_cls_flat = rcnn_cls.view(-1)

            groups = rcnn_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
            if groups != 1:
                rcnn_loss_cls = 0
                slice = rcnn_cls_labels.shape[0]
                for i in range(groups):
                    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat[i*slice:(i+1)*slice]), rcnn_cls_labels.float(), reduction='none')

                    cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                    rcnn_loss_cls = rcnn_loss_cls + (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

                rcnn_loss_cls = rcnn_loss_cls / groups
            
            else:

                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
                cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            if not self.model_cfg.get('USE_POINT_AS_JOINT_CLS',True):
                point_cls_flat = point_cls.view(-1)
                groups = point_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
                if groups != 1:
                    point_loss_cls = 0
                    slice = point_cls_flat.shape[0] // self.num_groups
                    for i in range(groups):
                        batch_loss_point_cls = F.binary_cross_entropy(torch.sigmoid(point_cls_flat[i*slice:(i+1)*slice]), rcnn_cls_labels.float(), reduction='none')
                        cls_valid_mask = (rcnn_cls_labels >= 0).float()
                        point_loss_cls = point_loss_cls + (batch_loss_point_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
                    
                    point_loss_cls  = point_loss_cls / groups


                else:

                    point_cls_flat = point_cls.view(-1)
                    #box_cls_flat   = box_cls.view(-1)
                    batch_loss_point_cls = F.binary_cross_entropy(torch.sigmoid(point_cls_flat), rcnn_cls_labels.float(), reduction='none')
                    cls_valid_mask = (rcnn_cls_labels >= 0).float()
                    point_loss_cls = (batch_loss_point_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)


        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        else:
            raise NotImplementedError

        if self.model_cfg.USE_BOX_ENCODING.ENABLED:
      
            if self.model_cfg.get('USE_POINT_AS_JOINT_CLS',True):
                rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight'] 
            else:
                rcnn_loss_cls = (rcnn_loss_cls+ point_loss_cls) * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight'] 

        else:
            rcnn_loss_cls = (rcnn_loss_cls) * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict


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
        # batch_cls_preds: (B, N, num_class or 1)
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
        batch_box_preds = torch.cat([batch_box_preds,rois[:,:,7:]],-1) #for superbox
        return batch_cls_preds, batch_box_preds
