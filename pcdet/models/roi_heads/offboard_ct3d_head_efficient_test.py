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
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
# from ..dense_heads.sparse_hybrid_head import SparseHybridHead
from ..model_utils.ctrans import build_transformer
import numba as nb
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils


class PointNetfeat(nn.Module):
    def __init__(self, pts_dim, x=1,outchannel=512,transformer=True):
        super(PointNetfeat, self).__init__()
        if outchannel==256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(pts_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x,  self.output_channel, 1)
        #self.conv3 = torch.nn.Conv1d(512 * x, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(256 * x)
        self.bn4 = nn.BatchNorm1d(self.output_channel)
        self.transformer = transformer
        if transformer:
            self.attn = nn.MultiheadAttention(512, 4, dropout=0.1)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x)) # NOTE: should not put a relu [bs,C,N]
        if self.transformer:
            query = x_ori[:,:,0:1].permute(2,0,1)
            key= value = x_ori[:,:,1:].permute(2,0,1)
            x = self.attn(query,key,value)[0].permute(1,2,0).contiguous()
        else:
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

            if (model_cfg.Transformer.num_frames == 4 and model_cfg.Transformer.p4conv_merge  and model_cfg.Transformer.merge_groups==1) or (model_cfg.Transformer.time_attn_type in ['mlp_merge','trans_merge'] \
                and model_cfg.Transformer.merge_groups==1) or model_cfg.Transformer.use_1_frame or model_cfg.Transformer.use_mlp_query_decoder: 
                # if model_cfg.Transformer.pyramid:
                #     times = 3
                # else:
                times = 2
            elif model_cfg.Transformer.merge_groups==2: # model_cfg.Transformer.joint_dim==1024 and model_cfg.Transformer.num_frames > 4:
                times=3
            else: # model_cfg.Transformer.joint_dim==1024 and model_cfg.Transformer.num_frames > 4:
                #### 这种写法默认只要不是1group 都是 1024 dim ####
                # if model_cfg.Transformer.pyramid:
                #     times=9
                # else:
                times=5

            if model_cfg.Transformer.hidden_dim==512:
                times = 2*(times -1) +1

            if not model_cfg.Transformer.get('frame1_reg',True):
                times = times -1

            # if model_cfg.TRANS_INPUT ==128:
            #     times=6
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
            self.feat = PointNetfeat(pts_dim, x, transformer=model_cfg.USE_BOX_ENCODING.TRANS_POOL)

            self.fc1 = nn.Linear(512 * x, 256 * x)
            self.fc2 = nn.Linear(256 * x, channels)

            self.pre_bn = nn.BatchNorm1d(pts_dim)
            self.bn1 = nn.BatchNorm1d(256 * x)
            self.bn2 = nn.BatchNorm1d(channels)
            self.relu = nn.ReLU()
            # NOTE: should there put a BN?
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

class RegMLP(nn.Module):
    def __init__(self):
        super().__init__()
        times = 1
        self.fc_s1 = nn.Linear(256*times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_ce1 = nn.Linear(256*times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(256*times, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)

    def forward(self, feat):

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return torch.cat([centers, sizes, headings],-1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

class PointNetTimeMerge(nn.Module):
    def __init__(self, pts_dim, x, CLS_NUM,joint_feat=True,model_cfg=None):
        super(PointNetTimeMerge, self).__init__()
        self.joint_feat = joint_feat
        # times = 5
        # if joint_feat:
        #     self.feat = None

        times = 4
        # else:
        #     times=1
        #     self.feat = PointNetfeat(pts_dim, x)

        #     self.fc1 = nn.Linear(512 * x, 256 * x)
        #     self.fc2 = nn.Linear(256 * x, 256)

        #     self.pre_bn = nn.BatchNorm1d(pts_dim)
        #     self.bn1 = nn.BatchNorm1d(256 * x)
        #     self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        # NOTE: should there put a BN?
        self.fc_s1 = nn.Linear(256*times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_c1 = nn.Linear(256*times, 256)
        self.fc_c2 = nn.Linear(256, CLS_NUM, bias=False)
        self.fc_ce1 = nn.Linear(256*times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(256*times, 256)
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


@nb.jit(nopython=True,parallel=True)
def croproipoint(src,trajectory_rois,cur_points,point_mask, batch_size,num_rois,num_sample,num_points,nonempty_mask):

    # for bs_idx in range(batch_size):

    #     start1 = time.time()
    #     cur_batch_boxes = trajectory_rois[bs_idx,0,:,:7].reshape(-1,7)
    #     cur_radiis = np.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
    #     cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:].cpu().numpy()
        # dis = np.linalg.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
        # point_mask = (dis <= cur_radiis.unsqueeze(-1))
        # batch_dict['point_mask_time'] = time.time() - start1
        # start2 = time.time()
        # time_mask_list = []
        # if num_rois >=128:
        #     import pdb;pdb.set_trace()
        #     print(num_rois)
    for roi_box_idx in range(0, num_rois):
        # start3 = time.time()
        cur_roi_points = cur_points[point_mask[roi_box_idx]]
        # time_mask = np.abs(cur_roi_points[:,-1]) < 1e-3
        # cur_roi_points = cur_roi_points[time_mask]
        # time_mask_list.append(time.time() - start3)
        if cur_roi_points.shape[0] > 0:
            np.random.seed(0)
            choice = np.random.choice(cur_roi_points.shape[0], num_points, replace=True)
            cur_roi_points_sample = cur_roi_points[choice]

            # elif cur_roi_points.shape[0] == 0:
            #     cur_roi_points_sample = np.zeros([num_points, 6])
            #     nonempty_mask[0, roi_box_idx] = False

            # else:
            #     empty_num = num_sample - cur_roi_points.shape[0]
            #     add_zeros = np.zeros([empty_num, 6])
            #     add_zeros = np.tile(cur_roi_points[0],(empty_num, 1))
            #     cur_roi_points_sample = np.concatenate([cur_roi_points, add_zeros],  0)

            cur_roi_points_sample = cur_roi_points_sample[:,:-1]
            src[0, roi_box_idx, :num_points, :] = cur_roi_points_sample

    return src

class OffboardHeadCT3DEffiTEST(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, voxel_size, point_cloud_range, num_class=1,**kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.time_stamp = self.model_cfg.get('USE_TIMESTAMP',None)
        assert not (self.time_stamp and model_cfg.Transformer.use_learn_time_token)
        # self.window_stride = kwargs['window_stride']
        # if self.model_cfg.USE_CT3D_NORM:
        #     self.pointemb = PointNet(30,1,1)
        # else:
        #     self.pointemb = PointNet(12,1,1)
        if self.model_cfg.USE_BOX_ENCODING.ENABLED:
            self.seqboxemb = PointNet(8,1,1,model_cfg=self.model_cfg)
            if self.model_cfg.get('USE_MLP_JOINTEMB',None):
                self.jointemb = MLP(256*5, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4)
            else:
                self.jointemb = PointNet(1,1,1,joint_feat=True,model_cfg=self.model_cfg)
        else:
            self.pointemb = PointNetTimeMerge(1,1,1,joint_feat=True,model_cfg=self.model_cfg)

        self.num_points = self.model_cfg.Transformer.num_points
        self.avg_stage1 = self.model_cfg.get('AVG_STAGE_1', None)
        num_queries = model_cfg.Transformer.num_queries
        self.masking_radius = model_cfg.Transformer.masking_radius
        self.nhead = model_cfg.Transformer.nheads
        hidden_dim = model_cfg.TRANS_INPUT
        self.hidden_dim = model_cfg.TRANS_INPUT
        self.merge_groups = model_cfg.Transformer.merge_groups
        self.use_box_pos = model_cfg.Transformer.use_box_pos
        self.use_voxel_point_feat = model_cfg.Transformer.use_voxel_point_feat
        self.half_up_dimension = model_cfg.Transformer.half_up_dimension
        self.query_embed = nn.Embedding(1, model_cfg.Transformer.hidden_dim)
        self.grid_size = model_cfg.ROI_GRID_PVRCNNPOOL.GRID_SIZE
        if isinstance(self.grid_size,list):
            self.num_grid_points = self.grid_size[0]*self.grid_size[1]*self.grid_size[2]
        else:
            self.num_grid_points = model_cfg.ROI_GRID_PVRCNNPOOL.GRID_SIZE**3
        self.num_key_points = self.num_grid_points
        self.use_learn_time_token = self.model_cfg.Transformer.use_learn_time_token
        self.use_reflection = self.model_cfg.get('USE_REFLECTION', True)
        self.use_traj_pos_emb = self.model_cfg.Transformer.get('use_traj_pos_emb', False)
        self.use_center_init_token = self.model_cfg.get('USE_CENTER_INIT_TOKEN', False)
        self.pyramid = self.model_cfg.Transformer.pyramid
        self.use_regmlp = self.model_cfg.get('USE_REGMLP', None)
        if self.use_center_init_token.enabled:
            if self.use_center_init_token.init_type == 'xyz':
                self.up_dimension_token = MLP(input_dim = 28, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
            elif self.use_center_init_token.init_type == 'xyzyaw':
                self.up_dimension_token = MLP(input_dim = 29, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
            elif self.use_center_init_token.init_type == 'xyzwhlyaw':
                self.up_dimension_token = MLP(input_dim = 32, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
            elif self.use_center_init_token.init_type == 'box':
                if self.use_center_init_token.share_token:
                    self.up_dimension_token = MLP(input_dim = 8, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
                else:
                    self.up_dimension_token = MLP(input_dim = 31, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
            else:
                raise NotImplementedError

        if self.use_reflection:
            res_dim = 0
        else:
            res_dim = 2

        if (not self.use_learn_time_token) and (not self.time_stamp):
            dim=30 - res_dim
        else:
            dim=27

        if self.model_cfg.Transformer.use_voxel_point_feat.get('feat_add',None):
            # dim = 28
            self.up_dimension_voxel = MLP(input_dim = dim, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
        else:
            self.up_dimension_voxel = MLP(input_dim = dim, hidden_dim = 64, output_dim = hidden_dim//2, num_layers = 3)

        if self.time_stamp:
            input_dim = 30 - res_dim
        else:
            input_dim = 29 - res_dim
        if self.use_voxel_point_feat.enabled and self.use_voxel_point_feat.share_up_dim and not self.use_reflection:
            self.up_dimension = MLP(input_dim = 27, hidden_dim = 64, output_dim = 128, num_layers = 3)
        elif self.model_cfg.VOXELIZE_POINT or self.use_voxel_point_feat.enabled or self.half_up_dimension:
            self.up_dimension = MLP(input_dim = input_dim, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)
        else:
            self.up_dimension = MLP(input_dim = input_dim, hidden_dim = 64, output_dim = hidden_dim, num_layers = 3)

        if not self.model_cfg.SHARE_LOCAL_VOXEL:
            self.up_dimension_local = MLP(input_dim = 27, hidden_dim = 64, output_dim = hidden_dim//2, num_layers = 3)

        if self.use_box_pos.enabled:
            # self.reduce_box_feat = MLP(input_dim = 512, hidden_dim = 64, output_dim = 32, num_layers = 3)
            if self.use_box_pos.only_use_xyzyaw:
                self.up_dimension_bbox = MLP(input_dim = 4, hidden_dim = 64, output_dim = hidden_dim, num_layers = 4)
            elif self.use_box_pos.box_res_emb:
                self.up_dimension_bbox = MLP(input_dim = 7 , hidden_dim = 64, output_dim = hidden_dim, num_layers = 4)
            else:
                self.up_dimension_bbox = MLP(input_dim = 9, hidden_dim = 64, output_dim = hidden_dim, num_layers = 4)

        self.up_dimension_bbox = MLP(input_dim = 7 , hidden_dim = 64, output_dim = hidden_dim, num_layers = 4)
        
        self.add_cls_token = model_cfg.Transformer.add_cls_token
        self.use_grid_points = model_cfg.Transformer.use_grid_points
        self.transformer = build_transformer(model_cfg.Transformer)


        if self.use_grid_points:
            if self.model_cfg.VECTOR_POOL:
                self.roi_grid_pool_layer = pointnet2_stack_modules.VectorPoolAggregationModuleMSG(128,model_cfg.ROI_GRID_POOL)
            else:

                if self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_METHOD=='pyramid_pool':
                    self.roi_grid_pool_layer = pointnet2_stack_modules.PyramidModuleV2(
                        input_channels=128,
                        nradius=self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_RADIUS,
                        nsamples=self.model_cfg.ROI_GRID_PVRCNNPOOL.NSAMPLE,
                        grid_sizes= self.model_cfg.ROI_GRID_PVRCNNPOOL.GRID_SIZE,
                        num_heads = self.model_cfg.ROI_GRID_PVRCNNPOOL.NHEAD,
                        head_dims = 128//self.model_cfg.ROI_GRID_PVRCNNPOOL.NHEAD,
                    )

                else:
                    if self.model_cfg.ROI_GRID_PVRCNNPOOL.ONE_BALL:
                        outchanel = self.model_cfg.ROI_GRID_PVRCNNPOOL.OUTCHANNEL
                        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
                                radii=self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_RADIUS2,
                                nsamples=self.model_cfg.ROI_GRID_PVRCNNPOOL.NSAMPLE2,
                                mlps=[[outchanel,outchanel]],
                                use_xyz=True,
                                # ignore_empty_voxel= self.model_cfg.ROI_GRID_PVRCNNPOOL.IGNORE_EMPTY_VOXEL,
                                pool_method=self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_METHOD,
                                # config = self.model_cfg.ROI_GRID_PVRCNNPOOL
                                )
                    else:
                        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
                                radii=self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_RADIUS,
                                nsamples=self.model_cfg.ROI_GRID_PVRCNNPOOL.NSAMPLE,
                                mlps=self.model_cfg.ROI_GRID_PVRCNNPOOL.MLPS,
                                use_xyz=True,
                                # ignore_empty_voxel= self.model_cfg.ROI_GRID_PVRCNNPOOL.IGNORE_EMPTY_VOXEL,
                                pool_method=self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_METHOD,
                                # config = self.model_cfg.ROI_GRID_PVRCNNPOOL
                                )
                    

        self.class_embed = nn.ModuleList()
        self.bbox_embed = nn.ModuleList()
        # import pdb;pdb.set_trace()
        self.num_pred = 4 if 'casc' in model_cfg.Transformer.use_decoder.name and not model_cfg.Transformer.time_attn_type == 'mlp_merge' else 1
        # if model_cfg.Transformer.pyramid:
        #     model_cfg.Transformer.hidden_dim = model_cfg.Transformer.hidden_dim*2

        if self.num_pred > 1:

            for i in range(self.num_pred):
                self.class_embed.append(nn.Linear(model_cfg.Transformer.hidden_dim, 1))
                if self.use_regmlp:
                    self.bbox_embed.append(RegMLP())
                else:
                    self.bbox_embed.append(MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4))
        else:
            self.class_embed = nn.Linear(model_cfg.Transformer.hidden_dim, 1)
            self.bbox_embed = MLP(model_cfg.Transformer.hidden_dim, model_cfg.Transformer.hidden_dim, self.box_coder.code_size * self.num_class, 4)

        if self.model_cfg.USE_BEV_FEAT.CONCAT:
            self.merge_bev_point = MLP(256*2, 256, 256, 3)

        if self.model_cfg.USE_FUSION_LOSS:
            self.timefusion = PointNetTimeMerge(1,1,1,model_cfg=self.model_cfg)
            self.timefusion_class_embed = nn.Linear(model_cfg.Transformer.hidden_dim, 1)
        if self.model_cfg.Transformer.use_grid_pos.enabled:
            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                if isinstance(self.grid_size, list):
                    self.gridindex = torch.cat([i.reshape(-1,1)for i in torch.meshgrid(torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1]), torch.arange(self.grid_size[2]))],1).float().cuda()
                else:
                    self.gridindex = torch.cat([i.reshape(-1,1)for i in torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size), torch.arange(self.grid_size))],1).float().cuda()
                
                if self.model_cfg.Transformer.use_grid_pos.norm:
                    self.gridindex = self.gridindex -1.5
                if self.model_cfg.Transformer.use_grid_pos.token_center:
                    self.gridindex = torch.cat([torch.tensor([0,0,0]).view(1,3).cuda(),self.gridindex],0)
                self.gridposembeding = MLP(input_dim = 3, hidden_dim = 256, output_dim = hidden_dim, num_layers = 2)
                self.gridposembeding_pyramid = MLP(input_dim = 3, hidden_dim = 256, output_dim = hidden_dim*2, num_layers = 3)
            else:
                self.pos = nn.Parameter(torch.zeros(1, self.num_grid_points, 256))

        # self.croppoint = roipoint_pool3d_utils.RoIPointPool3d(128,[1.0,1.0,1.0])

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





        # from tools.visual_utils import ss_visual_utils as V
        # pos_fea = voxel_point[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1) # 27 维度

        # lwh = rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,voxel_point.shape[1],1)
        # diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        # pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))

        if len(point_coords.shape)==3:
            global_roi_grid_points, local_roi_grid_points = self.get_grid_points_of_roi(
                rois.permute(0,2,1,3).contiguous(), grid_size=self.grid_size
            )  # (BxN, 6x6x6, 3)
        else:
            global_roi_grid_points, local_roi_grid_points = self.get_grid_points_of_roi(
                rois, grid_size=self.grid_size
            )  # (BxN, 6x6x6, 3)
        # import pdb;pdb.set_trace()
        # V.draw_scenes(global_roi_grid_points.view(-1,3))
        # order [xxxxyyyyzzzz ....] not [xyz...,xyz..xyz..xyz]
        if effi:
            cur_frame_index = list(range(0,global_roi_grid_points.shape[0],4))

        # global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3) # used to determine KNN with raw point global xyz

        if 'transformer' in self.model_cfg.ROI_GRID_PVRCNNPOOL.POOL_METHOD:
            pos_fea_list = []
            ## from xxxyyyzzz to xyzxyzxyz
            local_roi_grid_points = local_roi_grid_points.view(-1,4,self.num_key_points,3).permute(1,0,2,3).contiguous().view(-1,self.num_key_points,3)
            for i in range(rois.shape[1]):
                _, local_corner_points = self.get_corner_points_of_roi(rois[:,i,:,:].contiguous())  # (BxN, 2x2x2, 3)
                local_corner_points = local_corner_points.view(batch_size,rois.shape[2], -1, local_corner_points.shape[-1])  # (B, N, 2x2x2, 3)
                local_corner_points = local_corner_points.view(batch_size*rois.shape[2], -1)
                local_corner_add_center_points = torch.cat([local_corner_points, torch.zeros([local_corner_points.shape[0], 3]).cuda()], dim = -1)
                local_grid_pos_fea = local_roi_grid_points[i*batch_size*rois.shape[2]:(i+1)*batch_size*rois.shape[2],:,:3].repeat(1,1,9)  - local_corner_add_center_points.unsqueeze(1)

                lwh = rois[:,i,:,:].reshape(batch_size * rois.shape[2], -1)[:,3:6].unsqueeze(1).repeat(1,1,1)
                diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
                pos_fea = self.spherical_coordinate(local_grid_pos_fea, diag_dist = diag_dist.unsqueeze(-1))
                pos_fea_list.append(pos_fea.unsqueeze(1))
            

            local_grid_pos_fea = torch.cat(pos_fea_list,1).view(batch_size, -1, 27)

            if self.model_cfg.SHARE_LOCAL_VOXEL:
                ## To reuse feat, not use time_stamp here ##
                local_grid_pos_fea = torch.cat([local_grid_pos_fea,torch.zeros(local_grid_pos_fea.shape[0],local_grid_pos_fea.shape[1],2).cuda()],-1)
                local_grid_pos_fea = self.up_dimension(local_grid_pos_fea)
            else:
                local_grid_pos_fea = self.up_dimension_local(local_grid_pos_fea)
        else:
            local_grid_pos_fea = None

        #import pdb;pdb.set_trace()
        if not effi:
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
        else:

            num_points = point_coords.shape[1]
            num_key_points = self.num_key_points

            # point_coords = point_coords.view(point_coords.shape[0],point_coords.shape[1],point_coords.shape[-1])
            xyz = point_coords[:, :, 0:3].view(-1,3)
            if batch_cnt is None:
                xyz_batch_cnt = torch.tensor([point_coords.shape[1]]*rois.shape[2]*batch_size).cuda().int() 
            else:
                xyz_batch_cnt = torch.tensor(batch_cnt).cuda().int()
            new_xyz = torch.cat([i[0] for i in global_roi_grid_points.chunk(rois.shape[2],0)],0)
            new_xyz_batch_cnt = torch.tensor([self.num_key_points]*rois.shape[2]*batch_size).cuda().int()
        
        # import pdb;pdb.set_trace()
        # start = time.time()
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.view(-1,point_features.shape[-1]).contiguous(),
            # grid_feat = local_grid_pos_fea
        )  # (M1 + M2 ..., C)
        # batch_dict['grid_time'] = time.time() - start
        # batch_dict['group_time'] = group_time
        # batch_dict['mlp_time'] = mlp_time
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
            # empty_ball_mask = empty_ball_mask.view(point_features.shape[0], num_frames, self.num_key_points).contiguous()  # (BxN, 6x6x6, C)

        elif pooled_features.shape[-1]==256:
            features = pooled_features.view(batch_size, num_frames, rois.shape[2],num_key_points,256).permute(0,2,1,3,4).contiguous().view(-1,num_frames*num_key_points,256)
            # features = features.view(batch_size*rois.shape[2], num_frames*num_key_points*2,6).contiguous()  # (BxN, 6x6x6, C)
        else:
            features = pooled_features.view(batch_size, num_frames, rois.shape[2],num_key_points,60).permute(0,2,1,3,4).contiguous() 
            features = features.view(batch_size*rois.shape[2], num_frames*num_key_points,2,30)
            features = features.view(batch_size*rois.shape[2], num_frames*num_key_points*2,30).contiguous()  # (BxN, 6x6x6, C)
        return features,global_roi_grid_points.view(batch_size*rois.shape[2], num_frames*num_key_points,3).contiguous(), None

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



    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        start_time1 = time.time()
        batch_dict['rois'] = batch_dict['proposals_list'].permute(0,2,1,3)#[:,:,[4,3,0,2,1]]
        num_rois = batch_dict['rois'].shape[1]
        batch_dict['num_frames'] = batch_dict['rois'].shape[2]
        roi_scores_list = copy.deepcopy(batch_dict['roi_scores'])
        batch_dict['roi_scores'] = batch_dict['roi_scores'].permute(0,2,1)
        batch_dict['roi_labels'] = batch_dict['roi_labels'][:,0,:].long()
        proposals_list = batch_dict['proposals_list']
        batch_size = batch_dict['batch_size']
        cur_batch_boxes = copy.deepcopy(batch_dict['rois'].detach())[:,:,0]

        batch_dict['matching_index'] = []
        batch_dict['fg_inds'] = []
        repeat_frame = 1


        if self.model_cfg.USE_TRAJ:
            frame1 = cur_batch_boxes
            trajectory_rois = cur_batch_boxes[:,None,:,:].repeat(1,batch_dict['rois'].shape[-2],1,1)
            trajectory_roi_scores = torch.zeros_like(batch_dict['roi_scores'].permute(0,2,1))
            trajectory_rois[:,0,:,:]= frame1
            trajectory_roi_scores[:,0,:] = batch_dict['roi_scores'][:,:,0]
            # if batch_dict['sample_idx'][0] ==16:
            # import pdb;pdb.set_trace()
            effective_length = torch.zeros([batch_dict['batch_size'],batch_dict['rois'].shape[-2],trajectory_rois.shape[2]])
            effective_length[:,0] = 1
            # matching_dict = []
            matching_table = (trajectory_rois.new_ones([trajectory_rois.shape[1],trajectory_rois.shape[2]]) * -1).long()
            for i in range(1,batch_dict['rois'].shape[-2]):
                frame = torch.zeros_like(cur_batch_boxes)
                # if batch_dict['use_speed_bbox'][0]:
                frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] + trajectory_rois[:,i-1,:,7:9]
                # else:
                #     frame[:,:,0:2] = trajectory_rois[:,i-1,:,0:2] + trajectory_rois[:,i-1,:,7:9]/1.5
                frame[:,:,2:] = trajectory_rois[:,i-1,:,2:]

                # if i==20:
                #     import pdb
                #     pdb.set_trace()

                for idx in range( batch_dict['batch_size']):
                    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(frame[idx][:,:7], proposals_list[idx,i,:,:7])
                    max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
                    fg_inds = ((max_overlaps >= 0.5)).nonzero().view(-1)
                    effective_length[idx,i,fg_inds] = 1
                    t0_pre_idx_dict  = dict(map(lambda x,y:[x,y],fg_inds.cpu().numpy().tolist(),gt_assignment[fg_inds].cpu().numpy().tolist()))
                    matching_table[i,fg_inds] = gt_assignment[fg_inds]
                    # for k in effective_length[idx,i].nonzero()[:,0]:
                    #     if not k.item() in t0_pre_idx_dict.keys():
                    #         import pdb;pdb.set_trace()
                    # matching_dict.append(t0_pre_idx_dict)
                    # batch_dict['matching_index'].append(gt_assignment[fg_inds].cpu().numpy())
                    # batch_dict['fg_inds'].append(fg_inds.cpu().numpy())
                    trajectory_rois[idx,i,:,:][fg_inds] = proposals_list[idx][i][gt_assignment[fg_inds]]
                    trajectory_roi_scores[idx,i,:][fg_inds] = roi_scores_list[idx][i][gt_assignment[fg_inds]]

            batch_dict['effi_length'] = effective_length

        """
        if batch_dict['sample_idx'][0]-16 >=repeat_frame:
            from tools.visual_utils import ss_visual_utils as V
            trajectory_rois_copy = copy.deepcopy(trajectory_rois[:,0:1,:,:]).repeat([1,16,1,1])
            trajectory_rois_before = trajectory_rois_copy[:,1:]
            traj_memory2cur = traj_memory2cur[:,:15,:,:]

            for i in effective_legth.sum(0).nonzero()[:,0]:
                if i==20:
                    import pdb
                    pdb.set_trace()
                memory_used = effective_legth[:,i].bool()
                frame_idx = memory_used.nonzero()[0,0]
                idx = batch_dict['fg_inds'][frame_idx].tolist().index(i) 
                memory_idx = batch_dict['matching_index'][frame_idx][idx]
                trajectory_rois_before[:,memory_used,i] = traj_memory2cur[:,memory_used,memory_idx]
                # num_roi,_,C = batch_dict['grid_feature_memory'].shape
                # feat_before = batch_dict['grid_feature_memory'][:,:64*15].reshape(num_roi,15,-1,C)[memory_idx,memory_used,:,:]
                # src_before[i,memory_used,:,:] = feat_before 
            traj = torch.cat([trajectory_rois[:,0:1,:,:],trajectory_rois_before],1)
        """
        # import pdb
        # pdb.set_trace()
        # match_list = []
        # fg_list = []
        # for k, i in enumerate(batch_dict['matching_index']):
        #     match_list.extend(list(i.cpu().numpy()))
        #     fg_list.extend(list(batch_dict['fg_inds'][k].cpu().numpy()))

        # if batch_dict['sample_idx'][0] > 16:
        #     from tools.visual_utils import ss_visual_utils as V
        #     import pdb
        #     pdb.set_trace()
        #     matched_traj = trajectory_rois[0,0].sum(-1) != trajectory_rois[0,-1].sum(-1)
            # V.draw_scenes(batch_dict['points'][:,1:], trajectory_rois[0,0,:,:7],traj_memory2cur)
        batch_dict['traj_memory'] = trajectory_rois
        batch_dict['has_class_labels'] = True
        batch_dict['trajectory_rois'] = trajectory_rois
        batch_dict['trajectory_roi_scores'] = trajectory_roi_scores
        # batch_dict['gt_boxes'] = batch_dict['gt_norm_boxes']

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois'].permute(0,2,1,3).contiguous()
            targets_dict['rois']  = targets_dict['rois'][:,0]
            batch_dict['roi_scores'] = targets_dict['roi_scores']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # trajectory_roi_scores = targets_dict['trajectory_roi_scores']
            targets_dict['trajectory_rois'][:,0,:,:] = batch_dict['rois'][:,:,0,:]
            trajectory_rois = targets_dict['trajectory_rois']
            effective_length = targets_dict['effi_length']


        # import pdb;pdb.set_trace()
        if not self.training and self.model_cfg.get('MAX_ROIS',None):
            trajectory_rois = trajectory_rois[:,:,:self.model_cfg.MAX_ROIS]
            batch_dict['roi_scores'] = batch_dict['roi_scores'][:,:self.model_cfg.MAX_ROIS]
            batch_dict['roi_labels'] = batch_dict['roi_labels'][:,:self.model_cfg.MAX_ROIS]
            batch_dict['rois'] = batch_dict['rois'][:,:self.model_cfg.MAX_ROIS]

        rois = batch_dict['rois']
        num_rois = batch_dict['rois'].shape[1]

        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            empty_mask = batch_dict['rois'][:,:,0,:6].sum(-1)==0
            batch_dict['valid_traj_mask'] = ~empty_mask
            if self.training:
                targets_dict['empty_mask'] = empty_mask

        num_sample = self.num_points #* trajectory_rois.shape[1]
        # num_sample = self.num_points * (trajectory_rois.shape[1]-1)
        num_key_sample = self.num_key_points * trajectory_rois.shape[1]

        src = rois.new_zeros(batch_size, num_rois, num_sample, 5) # For waymo, the last dim is not used



        # import pdb;pdb.set_trace()
        timestamp_start = 0.0 # batch_dict['points'][:,-1].min()
        batch_dict['nonempty_mask'] = rois.new_ones(batch_size,num_rois).bool()
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        #def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):

        batch_dict['traj_time'] = time.time() - start_time1
        start_time = time.time()

        # import pdb;pdb.set_trace()
        # cur_batch_boxes = trajectory_rois[0,0,:,:7].view(1,-1,7)
        # cur_radiis =  (cur_batch_boxes[0,:,3:5].max(-1)[0]*0.2)[:,None].repeat(1,3)
        # src, empty_flag = self.croppoint(batch_dict['points'][None,:,1:4],batch_dict['points'][None,:,4:6],cur_batch_boxes)
        # # src, empty_flag = roipoint_pool3d_utils.RoIPointPool3dFunction.apply(batch_dict['points'][None,:,1:4],batch_dict['points'][None,:,4:6],cur_batch_boxes,cur_radiis,128)

        # batch_dict['rois_crop_time'] = 0
        # batch_dict['time_mask_time'] = 0
        # import pdb;pdb.set_trace()
        ##### crop current point ####
        if self.model_cfg.USE_RADIUS_CROP:
            for bs_idx in range(batch_size):

                start1 = time.time()
                cur_batch_boxes = trajectory_rois[bs_idx,0,:,:7].view(-1,7)
                cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.1
                cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]
                time_mask = cur_points[:,-1].abs() < 1e-3
                cur_points = cur_points[time_mask]
                dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                point_mask = (dis <= cur_radiis.unsqueeze(-1))
                mask = point_mask
                batch_dict['point_mask_time'] = time.time() - start1
                start2 = time.time()
                time_mask_list = []

                # dist = (A[:, None] - B(None, :)).norm(dim=-1)  # (num_rois, npoints)
                # mask = (dist < thresh)  # (num_rois, npoints)

                # sorted_idx = point_mask.float().argsort(dim=-1, descending=True)  # (num_rois, npoints), T, T, T, F, F, F, F
                # sampled_idx = sorted_idx[:, :128]  # (num_rois, 128)
                # roi_idx = torch.arange(num_rois)[:, None].repeat(1, 128)
                # sampled_mask = point_mask[roi_idx, sampled_idx]  # (num_rois, 128)

                # sampled_idx[~sampled_mask] = sampled_idx[:, 0:1]  # repeat the first idx

                # sorted_idx = mask.float().argsort(dim=-1, descending=True)  # (num_rois, npoints), T, T, T, F, F, F, F
                # sampled_idx = sorted_idx[:, :128]  # (num_rois, 128)
                # sampled_idx_buffer = sorted_idx[:, 0:1].repeat(1, 128)  # (num_rois, npoints)


                # sampled_idx = torch.topk(mask.float(),128)[1]
                # sampled_idx_buffer = sampled_idx[:, 0:1].repeat(1, 128)  # (num_rois, npoints)
                # roi_idx = torch.arange(num_rois)[:, None].repeat(1, 128)
                # sampled_mask = mask[roi_idx, sampled_idx]  # (num_rois, 128)
                # sampled_idx_buffer[sampled_mask] = sampled_idx[sampled_mask]

                # src = cur_points[sampled_idx_buffer][:,:,:5] # (num_rois, npoints)
                # empty_flag = sampled_mask.sum(-1)==0
                # src[empty_flag] = 0


                # batch_dict['crop_time'] = time.time()- start2

                # return sampled_idx # (num_rois, npoints)
                #croproipoint(src,trajectory_rois,cur_points,point_mask, batch_size,num_rois,num_sample,num_points,nonempty_mask):
                # nonemptymask = np.ones([batch_size,num_rois])
                # src = croproipoint(src.cpu().numpy(),trajectory_rois.cpu().numpy(),cur_points.cpu().numpy(),point_mask.cpu().numpy(), batch_size,num_rois,num_sample,128,nonemptymask)
                # src = torch.from_numpy(src).cuda()
                # if num_rois >=128:
                # import pdb;pdb.set_trace()
                #     print(num_rois)
                # start3 = time.time()
                # cur_roi_points = cur_points[point_mask[roi_box_idx]]
                # time_mask = cur_roi_points[:,-1].abs() < 1e-3
                # cur_roi_points = cur_roi_points[time_mask]
                # time_mask_list.append(time.time() - start3)
                # point_mask[roi_box_idx].nonzero().reshape(-1)

                for roi_box_idx in range(0, num_rois):
                    cur_roi_points = cur_points[point_mask[roi_box_idx]]
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
                
                # from tools.visual_utils import ss_visual_utils as V
                # import pdb
                # pdb.set_trace()
                # V.draw_scenes(src.view(-1,5))


                # if cur_roi_points.shape[0] >= num_sample:
                #     random.seed(0)
                #     index = np.random.randint(cur_roi_points.shape[0], size=num_sample)
                #     cur_roi_points_sample = cur_roi_points[index]

                # elif cur_roi_points.shape[0] == 0:
                #     cur_roi_points_sample = cur_roi_points.new_zeros(num_sample, 4)

                # else:
                #     empty_num = num_sample - cur_roi_points.shape[0]
                #     add_zeros = cur_roi_points.new_zeros(empty_num, 4)
                #     add_zeros = cur_roi_points[0].repeat(empty_num, 1)
                #     cur_roi_points_sample = torch.cat([cur_roi_points, add_zeros], dim = 0)

                # src[bs_idx, roi_box_idx, :, :] = cur_roi_points_sample


            batch_dict['rois_crop_time'] = 0 #time.time() - start2
            batch_dict['time_mask_time'] = 0 #np.array(time_mask_list).mean()
        else:
            cur_batch_boxes = trajectory_rois[0,0,:,:7].view(1,-1,7)
            cur_radiis =  (cur_batch_boxes[0,:,3:5].max(-1)[0]*0.2)[:,None].repeat(1,3)
            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == 0)][:,1:]
            time_mask = cur_points[:,-1].abs() < 1e-3
            cur_points = cur_points[time_mask]
            # src, empty_flag = self.croppoint(batch_dict['points'][None,:,1:4],batch_dict['points'][None,:,4:6],cur_batch_boxes)
            src, empty_flag = roipoint_pool3d_utils.RoIPointPool3dFunction.apply(cur_points[None,:,0:3],cur_points[None,:,3:5],cur_batch_boxes,cur_radiis,128)
            batch_dict['point_mask_time'] = 0
            batch_dict['rois_crop_time'] = 0
            batch_dict['time_mask_time'] = 0
        ##### crop current point ####
        # src = src.repeat([1,1,trajectory_rois.shape[1],1])
        # import pdb;pdb.set_trace()
        # src = sample_points[:,:,:,:3]
        gem_start = time.time()
        src = src.view(batch_size * num_rois, -1, src.shape[-1])
        src_ori = src
        i=0
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,0,:,:].contiguous())  # (BxN, 2x2x2, 3)

        corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)
        corner_points = corner_points.view(batch_size * num_rois, -1)
        corner_add_center_points = torch.cat([corner_points, trajectory_rois[:,i,:,:].contiguous().reshape(batch_size * num_rois, -1)[:,:3]], dim = -1)
        pos_fea = src[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1).repeat(1,self.num_points,1)  # 27 维
        # pos_fea_ori = pos_fea

        # if self.model_cfg.get('USE_SPHERICAL_COOR',True):
        lwh = trajectory_rois[:,i,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,pos_fea.shape[1],1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        src = torch.cat([pos_fea, src[:,:,3:]], dim = -1)
        batch_dict['gem_time'] = time.time() - gem_start
        batch_dict['crop_time'] = time.time()- start_time
        start_time = time.time()
        mask=None
        # pos_fea_list.append(pos_fea)
        # pos_fea_ori_list.append(pos_fea_ori)

        # pos_fea = torch.cat(pos_fea_list,dim=1)
        # pos_fea_ori = torch.cat(pos_fea_ori_list,dim=1)






        #.view(batch_size * num_rois, -1, src.shape[-1]) 
        # from tools.visual_utils import ss_visual_utils as V
        # import pdb
        # pdb.set_trace()
        ##### crop previous point ####
        # crop 之前帧的点大概200ms ##
        """
        for bs_idx in range(batch_size):
            # if self.time_stamp:

            cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:,1:]

            if self.model_cfg.get('USE_TRAJ_SAMPLING',None):
                cur_batch_boxes = trajectory_rois[bs_idx,:,:,:7].view(-1,7)
                #cur_batch_boxes = copy.deepcopy(cur_supboxes[bs_idx,:,0,:7])
                # if self.model_cfg.USE_RADIUS_CROP:
                cur_radiis = torch.sqrt((cur_batch_boxes[:,3]/2) ** 2 + (cur_batch_boxes[:,4]/2) ** 2) * 1.2
                if not self.training and cur_batch_boxes.shape[0] > 32:
                    length_iter= cur_batch_boxes.shape[0]//32
                    dis_list = []
                    for i in range(length_iter+1):
                        dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[32*i:32*(i+1),:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                        dis_list.append(dis)
                    dis = torch.cat(dis_list,0)
                else:
                    dis = torch.norm((cur_points[:,:2].unsqueeze(0) - cur_batch_boxes[:,:2].unsqueeze(1).repeat(1,cur_points.shape[0],1)), dim = 2)
                
                point_mask = (dis <= cur_radiis.unsqueeze(-1)).view(trajectory_rois.shape[1],trajectory_rois.shape[2],-1)
       

            for roi_box_idx in range(0, num_rois):

                if not effective_length[bs_idx,1:,roi_box_idx].sum():
                    continue

                for idx in range(1,trajectory_rois.shape[1]):
                    if not effective_length[bs_idx,idx,roi_box_idx]:
                            continue

                    cur_roi_points = cur_points[point_mask[idx,roi_box_idx]]
                    time_mask = (cur_roi_points[:,-1] - idx*0.1).abs() < 1e-3
                    cur_roi_points = cur_roi_points[time_mask]


                    if cur_roi_points.shape[0] > 0:
                        random.seed(0)
                        choice = np.random.choice(cur_roi_points.shape[0], self.num_points, replace=True)
                        cur_roi_points_sample = cur_roi_points[choice]
                        if self.training and self.model_cfg.get('USE_RANDOM_NOISE',None) and idx > 0:
                            enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                            if enable: 
                                cur_roi_points_sample[:,:3] += torch.from_numpy(np.random.random([cur_roi_points_sample.shape[0],3])*0.05).cuda()


                    else:

                        if self.use_reflection:
                            cur_roi_points_sample = cur_roi_points.new_zeros(self.num_points, 6)
                        else:
                            cur_roi_points_sample = cur_roi_points.new_zeros(self.num_points, 4)

                        if self.model_cfg.get('USE_CENTER_PADDING',None):
                            cur_roi_points_sample[:,:3] = trajectory_rois[bs_idx,idx,roi_box_idx,:][0:3].repeat(self.num_points, 1)
            
                        if idx==0:
                            batch_dict['nonempty_mask'][bs_idx, roi_box_idx] = False
                

                    if not self.time_stamp:
                        cur_roi_points_sample = cur_roi_points_sample[:,:-1]

                    src[bs_idx, roi_box_idx, self.num_points*idx:self.num_points*(idx+1), :] = cur_roi_points_sample
        
        
        ##### crop previous point ####
        
        
        # import pdb;pdb.set_trace()

        # src = src.view(batch_size * num_rois, -1, src.shape[-1])  # (b*128, 256, 6)

        # if batch_dict['sample_idx'][0] >= 4:
        #     from tools.visual_utils import ss_visual_utils as V
        #     import pdb
        #     pdb.set_trace()
        #     V.draw_scenes(batch_dict['points'][:,1:], trajectory_rois[0,0,:,:7],traj_memory2cur)
            # src_ori_memory = batch_dict['src_ori_memory']

        # src_ori = src
        # batch_dict['src_ori_memory'] = src_ori

        # batch_dict['crop_time'] = time.time()- start_time
        # start_time = time.time()

        # mask=None


        pos_fea_list = []
        trajectory_res_list= []
        pos_fea_ori_list = []
        for i in range(trajectory_rois.shape[1]):

            if self.model_cfg.get('USE_MID_CONTEXT',None):
                #assert batch_dict['points'][:,-1].min() < 0 , 'Should Check Sequence Clip for mid context'
                mid = int(trajectory_rois.shape[1]/2)
                corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,mid,:,:].contiguous())  # (BxN, 2x2x2, 3)

            else:
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
            pos_fea_ori_list.append(pos_fea_ori)

        pos_fea = torch.cat(pos_fea_list,dim=1)
        pos_fea_ori = torch.cat(pos_fea_ori_list,dim=1)
        # if batch_dict['sample_idx'][0] >= 4:
            # import pdb;pdb.set_trace()
            # pos_fea_memory = batch_dict['pos_fea_memory']

            # from tools.visual_utils import ss_visual_utils as V
            # import pdb
            # pdb.set_trace()
            # V.draw_scenes(batch_dict['points'][:,1:], trajectory_rois[0,0,:,:7],traj_memory2cur)
        batch_dict['pos_fea_memory'] = pos_fea
        batch_dict['pos_fea_ori_memory'] = pos_fea_ori
        
        """

        # point_features_list.append(point_bev_features)

        if self.model_cfg.USE_BOX_ENCODING.ENABLED:

            if self.model_cfg.USE_BOX_ENCODING.USE_SINGLEBOX:
                time_stamp = torch.zeros([trajectory_rois.shape[0],trajectory_rois.shape[1],trajectory_rois.shape[2],1]).cuda()
                box_seq = torch.cat([trajectory_rois[:,0:1,:,:7].repeat(1,4,1,1),time_stamp],-1)
            else:
                time_stamp = torch.ones([trajectory_rois.shape[0],trajectory_rois.shape[1],trajectory_rois.shape[2],1]).cuda()
                for i in range(time_stamp.shape[1]):
                    time_stamp[:,i,:] *= i*0.1 

                box_seq = torch.cat([trajectory_rois[:,:,:,:7],time_stamp],-1)
                box_seq_time = box_seq

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

            # box_seq = box_seq.permute(0,2,3,1).contiguous().view(batch_rcnn,box_seq.shape[-1],box_seq.shape[1])

            box_cls,  box_reg, feat_box,feat_box_traj = self.seqboxemb(box_seq.permute(0,2,3,1).contiguous().view(batch_rcnn,box_seq.shape[-1],box_seq.shape[1]))
            


        motion_feat = None

        # import pdb;pdb.set_trace()
        src = self.up_dimension(src) # [bs,num_points,num_feat]


        #TODO:  use local box norm

        center_token = None

        batch_dict['feature_time'] = time.time() -  start_time
        start_time = time.time()



        effi=True
        if effi:
            # start_time = time.time()
            src, voxel_point,empty_ball_mask = self.roi_grid_pool(batch_size,trajectory_rois,src_ori,src,batch_dict,batch_cnt=None,src=src_ori,effi=effi)
            # src=src.repeat(1,trajectory_rois.shape[1],1)
            batch_dict['grid_time'] = time.time() - start_time
        else:
            src, voxel_point,empty_ball_mask = self.roi_grid_pool(batch_size,trajectory_rois,src_ori,src,batch_dict,batch_cnt=None,src=src_ori,effi=effi)

        # src = src.repeat([1,trajectory_rois.shape[1],1])


        start5 = time.time()
        if batch_dict['sample_idx'][0] >=1:

            src_repeat = src[:,None,:64,:].repeat([1,trajectory_rois.shape[1],1,1])
            src_before = src_repeat[:,1:,:,:].clone() #[bs,traj,num_roi,C]
            valid_length = self.model_cfg.Transformer.num_frames -1 if batch_dict['sample_idx'][0] > self.model_cfg.Transformer.num_frames -1  else int(batch_dict['sample_idx'][0].item())

            # for i in effective_length[0,1:].sum(0).nonzero()[:,0]:
            #     for j in range(valid_length):
            #         if effective_length[0,j+1,i]:
            #             memory_idx = matching_dict[j][i.item()]
            #             src_before[i,j] = batch_dict['feature_bank'][j][memory_idx]#[:,:valid_length*64]#.reshape(16,64,256)[j]


            num_max_rois = max(trajectory_rois.shape[2], *[i.shape[0] for i in batch_dict['feature_bank']])
            feature_bank = self.reorder_memory(batch_dict['feature_bank'][:valid_length],num_max_rois)
            effective_length = effective_length[0,1:1+valid_length].bool() #rm dim of bs
            for i in range(valid_length):
                src_before[:,i][effective_length[i]] = feature_bank[i,matching_table[1+i][effective_length[i]]]


            src = torch.cat([src_repeat[:,:1],src_before],1).view(src.shape[0],-1,src.shape[-1])

        else:
            src = src.repeat([1,trajectory_rois.shape[1],1])
        batch_dict['match_time'] = time.time() - start5
        
        batch_dict['src_ori_memory'] = src_ori

        batch_dict['grid_feature_memory'] = src[:,:64]
        batch_dict['voxelize_time1'] = time.time() - start_time
        pading_time = time.time()
        # import pdb;pdb.set_trace()
        # time_stamps   = torch.chunk(src.new_ones([voxel_point.shape[0],voxel_point.shape[1],1]),1,trajectory_rois.shape[1])
        time_stamp = torch.cat([src.new_ones([voxel_point.shape[0],self.num_key_points,1])*i*0.1 for i in range(trajectory_rois.shape[1])],1)
        padding_zero = src.new_zeros([voxel_point.shape[0],voxel_point.shape[1],2])
        voxel_point_padding = torch.cat([padding_zero,time_stamp],-1)
        # if self.model_cfg.Transformer.use_one_coding_for_later:
        #     num_time_coding = 4
        # else:
        # num_time_coding = trajectory_rois.shape[1]

        # for i in range(num_time_coding):
        #     voxel_point_padding[:,i*self.num_key_points:(i+1)*self.num_key_points,-1] = i*0.1

        batch_dict['padding_time'] = time.time() - pading_time
        start4 = time.time()
        ######### use T0 Norm ########
        # corner_points, _ = self.get_corner_points_of_roi(trajectory_rois[:,0,:,:].contiguous())  # (BxN, 2x2x2, 3)
        # corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])  # (B, N, 2x2x2, 3)
        # corner_points = corner_points.view(batch_size * num_rois, -1)
        # corner_add_center_points = torch.cat([corner_points, trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,:3]], dim = -1)

        # from tools.visual_utils import ss_visual_utils as V
        pos_fea = voxel_point[:,:,:3].repeat(1,1,9) - corner_add_center_points.unsqueeze(1) # 27 维度

        lwh = trajectory_rois[:,0,:,:].reshape(batch_size * num_rois, -1)[:,3:6].unsqueeze(1).repeat(1,voxel_point.shape[1],1)
        diag_dist = (lwh[:,:,0]**2 + lwh[:,:,1]**2 + lwh[:,:,2]**2) ** 0.5
        pos_fea = self.spherical_coordinate(pos_fea, diag_dist = diag_dist.unsqueeze(-1))
        ######### use T0 Norm ########
        batch_dict['posfea_time'] = time.time() - start4

        voxel_point_padding = torch.cat([pos_fea,voxel_point_padding],-1)
        voxel_point_feat = self.up_dimension_voxel(voxel_point_padding)

        src = src + voxel_point_feat

        
        if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
            src[empty_mask.view(-1)] = 0

        batch_dict['voxelize_time2'] = time.time() - start_time
        start_time = time.time()
        box_pos = None

        if self.model_cfg.Transformer.use_grid_pos.enabled:
            grid_pos = None
            if self.model_cfg.Transformer.use_grid_pos.init_type == 'index':
                pos = self.gridposembeding(self.gridindex.cuda())[None,:,:]
                if not self.model_cfg.Transformer.use_grid_pos.token_center:
                    pos = torch.cat([torch.zeros(1,1,self.hidden_dim).cuda(),pos],1)
            else:
                grid_pos = None
                pos = torch.cat([torch.zeros(1,1,256).cuda(),self.pos],1)

            if self.pyramid:
                pos_pyramid = self.gridposembeding_pyramid(self.gridindex[:8].cuda())
                pos_pyramid = torch.cat([torch.zeros(1,self.hidden_dim*2).cuda(),pos_pyramid],0)[None,:,:]
            else:
                pos_pyramid =None


        else:
            grid_pos = None
            pos=None


        if self.model_cfg.USE_BOX_ENCODING.ENABLED:
            # output
            if 'ct3d' not in self.model_cfg.Transformer.name:
                # import pdb;pdb.set_trace()
                if self.model_cfg.Transformer.use_1_frame:
                    src = src[:,:src.shape[1]//4]

                start_time = time.time()
                hs, tokens, mlp_merge = self.transformer(src,mask,pos_pyramid = pos_pyramid, pos=pos, box_pos=box_pos, 
                                                      num_frames = batch_dict['num_frames'], center_token=center_token,
                                                      empty_ball_mask=empty_ball_mask)
                batch_dict['casa_time'] = time.time() - start_time
                head_time = time.time()
                point_cls_list = []
                point_reg_list = []
                if hs.shape[0] > 1:
                    if self.add_cls_token:
                        cls = cls.permute(1,0,2).squeeze(1)
                        point_cls = self.class_embed[0](cls)
                        for i in range(hs.shape[0]):
                            point_reg_list.append(self.bbox_embed[i](hs[i]))
                        point_reg = torch.cat(point_reg_list,0)
                        hs = hs[0:1]
                    else:
                        # import pdb;pdb.set_trace()
                        if self.model_cfg.get('USE_3LAYER_LOSS',None):

                            if self.model_cfg.Transformer.merge_groups==2:
                                index_list = [[0,2,4],[1,3,5]]
                            else:
                                index_list = [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]

                            for i in range(3):
                                point_cls_list.append(self.class_embed[0](tokens[index_list[0][i]]))
                            if self.model_cfg.get('SHARE_REG_HEAD',None):
                                for i in range(hs.shape[0]):
                                    for j in range(3):
                                        point_reg_list.append(self.bbox_embed[0](tokens[index_list[i][j]]))
                            else:
                                for i in range(hs.shape[0]):
                                    for j in range(3):
                                        point_reg_list.append(self.bbox_embed[i](tokens[index_list[i][j]]))
                        else:
                            for i in range(hs.shape[0]):
                                point_cls_list.append(self.class_embed[i](hs[i]))
                                point_reg_list.append(self.bbox_embed[i](hs[i]))

                        if self.model_cfg.Transformer.frame1_cls and not self.model_cfg.get('USE_3LAYER_LOSS',None):
                            point_cls = point_cls_list[0]
                        else:
                            point_cls = torch.cat(point_cls_list,0)
                        if self.model_cfg.Transformer.get('frame1_reg',True):
                            point_reg = torch.cat(point_reg_list,0)
                        else:
                            point_reg = torch.cat(point_reg_list[1:],0)
                        if self.model_cfg.Transformer.joint_dim==1024:
                            hs = hs.permute(1,0,2).reshape(hs.shape[1],-1)
                        else:
                            hs = hs[0:1]
                    
                else:
                    point_cls = self.class_embed[0](hs[0])
                    point_reg = self.bbox_embed[0](hs[0])
                # if point_cls.shape[1] >1 :
                #     point_cls = point_cls.mean(1,True)
                #     point_reg = point_reg.mean(1,True)
                #     hs = hs.mean(1)
            else:
                pos = torch.zeros([src.shape[0],src.shape[1],self.model_cfg.Transformer.hidden_dim]).cuda().float()
                hs,_= self.transformer(src, self.query_embed.weight, pos)
                point_cls = self.class_embed[0](hs)[-1].squeeze(1)
                point_reg = self.bbox_embed[0](hs)[-1].squeeze(1)

            if  self.model_cfg.Transformer.dec_layers > 1:
                hs = hs[-1].squeeze()

            # import pdb;pdb.set_trace()
            if self.model_cfg.get('USE_MLP_JOINTEMB',None):
                joint_cls = None
                joint_reg = self.jointemb(torch.cat([hs,feat_box],-1))
            else:
                joint_cls, joint_reg, _, _ = self.jointemb(None,torch.cat([hs,feat_box],-1))
            
            if self.model_cfg.USE_FUSION_LOSS:
                fusion_cls = self.timefusion_class_embed(mlp_merge.squeeze())
                if self.model_cfg.Transformer.fusion_type == 'mlp':
                    _, fusion_reg, _, _ = self.timefusion(None,torch.cat([mlp_merge.squeeze(),feat_box],-1))
                else:
                    _, fusion_reg, _, _ = self.timefusion(None,torch.cat([mlp_merge.permute(1,0,2).contiguous().view(mlp_merge.shape[1],-1),feat_box],-1))
            
            if self.model_cfg.get('USE_POINT_AS_JOINT_CLS',True):
                rcnn_cls = point_cls
            else:
                rcnn_cls = joint_cls

            rcnn_reg = joint_reg

        else:
            # Transformer
            if 'ct3d' not in self.model_cfg.Transformer.name:
                hs, cls, mlp_merge = self.transformer(src,mask, pos=pos, box_pos=box_pos, num_frames = batch_dict['num_frames'], center_token=center_token)
                point_cls_list = []
                point_reg_list = []

                for i in range(hs.shape[0]):
                    point_cls_list.append(self.class_embed[i](hs[i]))
                    point_reg_list.append(self.bbox_embed[i](hs[i]))

                point_cls = point_cls_list[0]

                point_reg = torch.cat(point_reg_list,0)

                if self.model_cfg.Transformer.joint_dim==1024:
                    hs = hs.permute(1,0,2).reshape(hs.shape[1],-1)

                joint_cls, joint_reg, _, _ = self.pointemb(None,hs.squeeze())
                rcnn_cls = point_cls
                rcnn_reg = joint_reg
            else:
                pos = torch.zeros([src.shape[0],src.shape[1],self.model_cfg.Transformer.hidden_dim]).cuda().float()
                hs,_= self.transformer(src, self.query_embed.weight, pos)
                rcnn_cls = self.class_embed[0](hs)[-1].squeeze(1)
                rcnn_reg = self.bbox_embed[0](hs)[-1].squeeze(1)

        # batch_dict['voxelize_time'] = time.time() - start_time

        if not self.training:
            batch_dict['rois'] = batch_dict['rois'][:,:,0].contiguous()
            
            # if rcnn_cls.shape[0] != rcnn_reg.shape[0]:
            #     rcnn_cls = torch.cat(rcnn_cls[:,None,:].chunk(4,0),1).mean(1)
            if self.model_cfg.get('USE_3LAYER_LOSS',None):
                rcnn_cls = rcnn_cls[-rcnn_cls.shape[0]//3:]
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            if self.model_cfg.USE_CLS_LOSS:
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
                        batch_dict['roi_labels'] = torch.cat([batch_dict['roi_labels'][car_mask],batch_dict['roi_labels'][~car_mask]],0)[None]
                        batch_cls_preds = batch_cls_preds.view(batch_size, -1, 1)

                    else:
                        batch_cls_preds = torch.sqrt(batch_cls_preds*stage1_score)
                    batch_dict['cls_preds_normalized']  = True
                batch_dict['batch_cls_preds'] = batch_cls_preds
                
            else:
                batch_dict['batch_cls_preds'] = batch_dict['roi_scores'][:,:,:1]
                batch_dict['cls_preds_normalized'] = True
        else:
            targets_dict['batch_size'] = batch_size
            targets_dict['nonempty_mask'] = batch_dict['nonempty_mask']
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            if self.model_cfg.USE_FUSION_LOSS:
                targets_dict['fusion_cls'] = fusion_cls
                targets_dict['fusion_reg'] = fusion_reg

            if self.model_cfg.USE_BOX_ENCODING.ENABLED:
                targets_dict['box_reg'] = box_reg
                targets_dict['box_cls'] = box_cls
                targets_dict['point_reg'] = point_reg
                targets_dict['point_cls'] = point_cls
            self.forward_ret_dict = targets_dict
        batch_dict['head_time'] = time.time() - head_time
        batch_dict['transformer_time'] = time.time()- start_time
        batch_dict['forward_time'] = time.time() - start_time1
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

        if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
            gt_boxes3d_ct = forward_ret_dict['gt_traj_of_rois'][..., 0:code_size]
            num_rois = gt_boxes3d_ct.shape[1]//4
            gt_of_rois_src = forward_ret_dict['gt_traj_of_rois_src'][..., 0:code_size].view(-1, code_size)
        else:
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
  
            if self.model_cfg.USE_FUSION_LOSS:
                fusion_reg = forward_ret_dict['fusion_reg']  # (rcnn_batch_size, C)
                fusion_loss_reg = self.reg_loss_func(
                            fusion_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)  # [B, M, 7]
                fusion_loss_reg = (fusion_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                fusion_loss_reg = fusion_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
                tb_dict['fusion_loss_reg'] = fusion_loss_reg.item()
                rcnn_loss_reg += fusion_loss_reg

            # import pdb;pdb.set_trace()
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

                    
                    # groups = point_reg.shape[0]//reg_targets.shape[0]
                    # if groups != 1 :
                    #     point_loss_corners = 0
                    #     slice = point_reg.shape[0] // groups
                    #     for i in range(groups):
                    point_loss_corner = corner_loss_func(point_boxes3d[:, 0:7],gt_of_rois_src[fg_mask][:, 0:7].repeat(4,1))  # [B, M, 7]

                    point_loss_corner = point_loss_corner.mean()
                    point_loss_corner = point_loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']
                    # point_loss_corners += point_loss_reg

                    rcnn_loss_reg += point_loss_corner
                    tb_dict['point_loss_corner'] = point_loss_corner.item()

                    

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        nonempty_mask = forward_ret_dict['nonempty_mask'].view(-1)
        # if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
        #     empty_mask = forward_ret_dict['empty_mask'].view(-1)
        if self.model_cfg.USE_BOX_ENCODING.ENABLED:
            point_cls = forward_ret_dict['point_cls']

        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        if self.model_cfg.USE_FUSION_LOSS:
            fusion_cls = forward_ret_dict['fusion_cls']
            fusion_cls_flat = fusion_cls.view(-1)
            batch_loss_fusion_cls = F.binary_cross_entropy(torch.sigmoid(fusion_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            fusion_loss_cls = (batch_loss_fusion_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            tb_dict = {'fusion_loss_cls': fusion_loss_cls.item()}

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':

            rcnn_cls_flat = rcnn_cls.view(-1)
            if self.model_cfg.get('USE_TRAJ_GT_ASSIGN',None):
                rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][:,0].contiguous().view(-1)

            # import pdb;pdb.set_trace()
            groups = rcnn_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
            if groups != 1:
                rcnn_loss_cls = 0
                slice = rcnn_cls_labels.shape[0]
                for i in range(groups):
                    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat[i*slice:(i+1)*slice]), rcnn_cls_labels.float(), reduction='none')
                    # if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
                    #     cls_valid_mask = ((rcnn_cls_labels >= 0) & ~empty_mask).float() 
                    # else:
                    cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                    rcnn_loss_cls = rcnn_loss_cls + (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

                rcnn_loss_cls = rcnn_loss_cls / groups
            
            else:

                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
                # if self.model_cfg.get('USE_TRAJ_EMPTY_MASK',None):
                #     cls_valid_mask = ((rcnn_cls_labels >= 0) & ~empty_mask).float() 
                # else:
                cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            if not self.model_cfg.get('USE_POINT_AS_JOINT_CLS',True):
                point_cls_flat = point_cls.view(-1)
                groups = point_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
                if groups != 1:
                    point_loss_cls = 0
                    slice = point_cls_flat.shape[0] // self.merge_groups
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

            if self.model_cfg.get('USE_BOX_CLS', False):
                box_cls = forward_ret_dict['box_cls']
                box_cls_flat = box_cls.view(-1)
                #box_cls_flat   = box_cls.view(-1)
                batch_loss_box_cls = F.binary_cross_entropy(torch.sigmoid(box_cls_flat), rcnn_cls_labels.float(), reduction='none')
                # cls_valid_mask = (rcnn_cls_labels >= 0).float()
                box_loss_cls = (batch_loss_box_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
                rcnn_loss_cls += box_loss_cls
                tb_dict = {'box_loss_cls': box_loss_cls.item()}

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

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride,num_rois):
        x_idxs = (keypoints[:, :, 0] - self.model_cfg.POINT_CLOUD_RANGE[0]) / self.model_cfg.VOXEL_SIZE[0]
        y_idxs = (keypoints[:, :, 1] - self.model_cfg.POINT_CLOUD_RANGE[1]) / self.model_cfg.VOXEL_SIZE[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features.view(batch_size*num_rois,-1,point_bev_features.shape[-1])

    def reorder_memory(self, memory,num_max_rois):

            # num_max_rois = max([len(bbox) for bbox in pred_bboxes])
            # num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error 
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
        batch_box_preds = torch.cat([batch_box_preds,rois[:,:,7:]/1.5],-1) #for superbox
        return batch_cls_preds, batch_box_preds


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans
