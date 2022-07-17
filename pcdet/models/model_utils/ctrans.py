from os import getgrouplist
import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from copy import deepcopy
from einops import rearrange, repeat
from torch import nn, einsum
import matplotlib.pyplot as plt
#import cv2

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y

def with_pos_embed(tensor, pos: Optional[Tensor]):
    if pos is None:
        return tensor
    else:
        if tensor.shape == pos.shape:
            return tensor + pos
        else:
            index = tensor.shape[0]
            pos = pos[:index]
            return tensor + pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SEMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layer_list = []
        layer_list.append(nn.Linear(input_dim, output_dim))
        layer_list.append(SELayer(output_dim))
        layer_list.append(nn.Linear(output_dim, output_dim))
        layer_list.append(nn.Linear(output_dim, output_dim))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SpatialMixerBlock(nn.Module):

    def __init__(self,hidden_dim,grid_size,channels,config=None):
        super().__init__()


        self.mixer_x = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_y = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_z = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = FeedForward(channels,2*channels)
        self.config = config
        self.grid_size = grid_size

    def forward(self, src):

        src_3d = src.permute(1,2,0).contiguous().view(src.shape[1],src.shape[2],self.grid_size,self.grid_size,self.grid_size) #[xyz]>> merge order is inverse >>xyz
        if self.config.get('order', 'zyx') =='xyz':
            src_3d = src_3d.permute(0,1,4,3,2).contiguous() #[zyx]>> merge order is inverse >>xyz
        mixed_x = self.mixer_x(src_3d) #[0,1,2,3,4]
        # mixed_x = src_3d.permute(0,2,3,4,1) + self.ffn(mixed_x)
        mixed_x = src_3d + mixed_x
        mixed_x = self.norm_x(mixed_x.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_y = self.mixer_y(mixed_x.permute(0,1,2,4,3)).permute(0,1,2,4,3).contiguous()
        # mixed_y = mixed_x.permute(0,2,3,4,1)  + self.ffn(mixed_y.permute(0,2,3,4,1))
        mixed_y =  mixed_x + mixed_y
        mixed_y = self.norm_y(mixed_y.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_z = self.mixer_z(mixed_y.permute(0,1,4,3,2)).permute(0,1,4,3,2).contiguous()
        # mixed_z = mixed_y.permute(0,2,3,4,1) + self.ffn(mixed_z.permute(0,2,3,4,1))
        mixed_z =  mixed_y + mixed_z
        mixed_z = self.norm_z(mixed_z.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        src_mixer = mixed_z.view(src.shape[1],src.shape[2],-1).permute(2,0,1)
        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)


        return src_mixer

class SpatialMixerBlockV2(nn.Module):

    def __init__(self,hidden_dim):
        super().__init__()


        self.mixer_x = MLP(input_dim = 4, hidden_dim = hidden_dim, output_dim = 4, num_layers = 3)
        self.mixer_y = MLP(input_dim = 4, hidden_dim = hidden_dim, output_dim = 4, num_layers = 3)
        self.mixer_z = MLP(input_dim = 4, hidden_dim = hidden_dim, output_dim = 4, num_layers = 3)
        self.norm_x = nn.LayerNorm(256)
        self.norm_y = nn.LayerNorm(256)
        self.norm_z = nn.LayerNorm(256)
        self.norm_channel = nn.LayerNorm(256)
        self.ffn = FeedForward(256,512)

    def forward(self, src):

        src_3d = src.permute(1,2,0).contiguous().view(src.shape[1],src.shape[2],4,4,4)
        mixed_x = self.mixer_x(src_3d) #[0,1,2,3,4]
        # mixed_x = src_3d.permute(0,2,3,4,1) + self.ffn(mixed_x)
        mixed_x = src_3d + mixed_x
        mixed_x = self.norm_x(mixed_x.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_y = self.mixer_y(mixed_x.permute(0,1,2,4,3)).permute(0,1,2,4,3).contiguous()
        # mixed_y = mixed_x.permute(0,2,3,4,1)  + self.ffn(mixed_y.permute(0,2,3,4,1))
        mixed_y =  mixed_x + mixed_y
        mixed_y = self.norm_y(mixed_y.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_z = self.mixer_z(mixed_y.permute(0,1,4,3,2)).permute(0,1,4,3,2).contiguous()
        # mixed_z = mixed_y.permute(0,2,3,4,1) + self.ffn(mixed_z.permute(0,2,3,4,1))
        mixed_z =  mixed_y + mixed_z
        mixed_z = self.norm_z(mixed_z.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        src_mixer = mixed_z.view(src.shape[1],src.shape[2],-1).permute(2,0,1)
        # src_mixer = src_mixer + src_mixer
        # src_mixer = self.norm_channel(src_mixer)
        src_mixer = self.ffn(src_mixer)


        return src_mixer

class TimeMixerBlock(nn.Module):

    def __init__(self):
        super().__init__()


        self.mixer_x = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
        self.mixer_y = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
        self.mixer_z = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
        self.norm_x = nn.LayerNorm(256)
        self.norm_y = nn.LayerNorm(256)
        self.norm_z = nn.LayerNorm(256)
        self.norm_channel = nn.LayerNorm(256)
        self.ffn = FeedForward(256,512)

    def forward(self, src):
        # import pdb;pdb.set_trace()
        mixed_x = self.mixer_x(src) #[0,1,2,3,4]
        # mixed_x = src_3d.permute(0,2,3,4,1) + self.ffn(mixed_x)
        mixed_x = src + mixed_x #torch.Size([64, 128, 256, 4])
        mixed_x = self.norm_x(mixed_x.permute(0,1,3,2))

        src_mixer = (mixed_x + self.ffn(mixed_x)).permute(0,2,1,3).contiguous().view(src.shape[0],-1,src.shape[2])
        src_mixer = self.norm_channel(src_mixer)

        return src_mixer

def MLP_v2(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class B():
    count = 0
    def __init__(self):
        B.count += 1


class C():
    count = 0

    def __init__(self):
        self._increase_count()

    @classmethod
    def _increase_count(cls):
        cls.count += 1

def show_feature_map(feature_map):
    B()
    #import pdb;pdb.set_trace()
    #feature_map = feature_map.squeeze(0)
    feature_map = feature_map.detach().cpu().numpy()
    feature_map_num = feature_map.shape[0]
    norm_img = feature_map[0] #values.squeeze().sum(0)
    norm_img = norm_img - norm_img.min()
    norm_img = (norm_img / norm_img.max())*255
    # norm_img[94,94] = 255
    heat_img = cv2.applyColorMap(norm_img.astype('uint8'), cv2.COLORMAP_JET) 
    heat_img1 = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/home/xschen/repos/OpenPCDet_xuesong/train_500_box3_layer%d_attn.png'% B.count,heat_img1)
    cv2.imwrite('/home/xschen/repos/OpenPCDet_xuesong/train_500_box3_layer%d_token2point.png'% B.count,heat_img1[0:1,:])
    cv2.imwrite('/home/xschen/repos/OpenPCDet_xuesong/train_500_box3_layer%d_point2token.png'% B.count,heat_img1[:,:1])

class CrossAttn(nn.Module):
    count = 0
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,
                 use_channel_weight=False,time_attn=False,time_attn_type=False, 
                 use_motion_attn=False,share_head=True,channel_time=False,use_grid_pos= True,
                 fusion_init_token=False,merge_groups=None,num_frames=None,group_concat=None,
                 use_mlp_as_query=None,src_as_value=False):
        super().__init__()

        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.time_attn_type = time_attn_type
        self.use_motion_attn = use_motion_attn
        self.time_attn = time_attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.share_head = share_head
        self.channel_time = channel_time
        self.use_grid_pos = use_grid_pos
        self.fusion_init_token=fusion_init_token
        self.merge_groups = merge_groups
        self.num_frames = num_frames
        self.group_concat = group_concat
        self.use_mlp_as_query = use_mlp_as_query
        self.src_as_value = src_as_value

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.channel_time:
            self.time_attn1 = MultiHeadedAttentionChannelTime([256,127,43,43,43], nhead)
        else:
            if self.share_head:
                self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            else:
                self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_mlp_fusion = MLP(input_dim = 256*(num_frames//merge_groups), hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.group_mlp_fusion = MLP(input_dim = 256*(num_frames//merge_groups), hidden_dim = 256, output_dim = 256, num_layers = 3)

            
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_cls_token = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ffn1 = FFN(d_model, dim_feedforward)
        # self.ffn2 = FFN(d_model, dim_feedforward)
  

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,src,tokens=None, pos=None):

        # src[65,512,128]
        # import pdb;pdb.set_trace()
        if self.fusion_init_token:
            if self.use_grid_pos:
                q = k = self.with_pos_embed(src, pos[:,:1].repeat(1,src.shape[1],1))

            src2 = self.self_attn(q, k, value=src)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            # group_length = self.num_frames//self.merge_groups
            tokens1 = src[0:1,:src.shape[1]//8]
            tokens2 = src[0:1,2*src.shape[1]//8:3*src.shape[1]//8]
            tokens3 = src[0:1,4*src.shape[1]//8:5*src.shape[1]//8]
            tokens4 = src[0:1,6*src.shape[1]//8:7*src.shape[1]//8]
            tokens = torch.cat([tokens1,tokens2,tokens3,tokens4],1)


            src_list = src[1:].chunk(4,1)
            src1,src2,src3,src4 = [src_list[i][:,:src_list[0].shape[1]//2] for i in range(4)]
            src5,src6,src7,src8 = [src_list[i][:,src_list[0].shape[1]//2:] for i in range(4)]
            src_list = [src_list[i].view(src_list[0].shape[0],src_list[0].shape[1]//2,512) for i in range(4)]
            src_merge_list = self.time_mlp_fusion(torch.cat(src_list,0)).chunk(4,0)
            if self.use_grid_pos:
                k1  = self.with_pos_embed(src_merge_list[0], pos[1:,:1].repeat(1,src1.shape[1],1))
                k2  = self.with_pos_embed(src_merge_list[1], pos[1:,:1].repeat(1,src1.shape[1],1))
                k3  = self.with_pos_embed(src_merge_list[2], pos[1:,:1].repeat(1,src1.shape[1],1))
                k4  = self.with_pos_embed(src_merge_list[3], pos[1:,:1].repeat(1,src1.shape[1],1))
                q1 = self.with_pos_embed(src1, pos[1:,:1].repeat(1,src1.shape[1],1))
                q2 = self.with_pos_embed(src2, pos[1:,:1].repeat(1,src1.shape[1],1))
                q3 = self.with_pos_embed(src3, pos[1:,:1].repeat(1,src1.shape[1],1))
                q4 = self.with_pos_embed(src4, pos[1:,:1].repeat(1,src1.shape[1],1))
                q5 = self.with_pos_embed(src5, pos[1:,:1].repeat(1,src1.shape[1],1))
                q6 = self.with_pos_embed(src6, pos[1:,:1].repeat(1,src1.shape[1],1))
                q7 = self.with_pos_embed(src7, pos[1:,:1].repeat(1,src1.shape[1],1))
                q8 = self.with_pos_embed(src8, pos[1:,:1].repeat(1,src1.shape[1],1))

            if self.share_head:
                if self.group_concat:
                    cross_src1 = self.time_attn(q1, k1, value=src_merge_list[0], attn_mask=None)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src5 = self.time_attn2(q5, k1, value=src_merge_list[0], attn_mask=None)[0]
                    src5 = self.ffn1(src5,cross_src5)
                    src1 = self.group_mlp_fusion(torch.cat([src1,src5],-1))


                    cross_src2 = self.time_attn(q2, k2, value=src_merge_list[1], attn_mask=None)[0]
                    src2 = self.ffn1(src2,cross_src2)
                    cross_src6 = self.time_attn2(q6, k2, value=src_merge_list[1], attn_mask=None)[0]
                    src6 = self.ffn1(src6,cross_src6)
                    src2 = self.group_mlp_fusion(torch.cat([src2,src6],-1))

                    cross_src3 = self.time_attn(q3, k3, value=src_merge_list[2], attn_mask=None)[0]
                    src3 = self.ffn1(src3,cross_src3)
                    cross_src7 = self.time_attn2(q7, k3, value=src_merge_list[2], attn_mask=None)[0]
                    src7 = self.ffn1(src7,cross_src7)
                    src3 = self.group_mlp_fusion(torch.cat([src3,src7],-1))

                    cross_src4 = self.time_attn(q4, k4, value=src_merge_list[3], attn_mask=None)[0]
                    src4 = self.ffn1(src4,cross_src4)
                    cross_src8 = self.time_attn2(q8, k4, value=src_merge_list[3], attn_mask=None)[0]
                    src8 = self.ffn1(src8,cross_src8)
                    src4 = self.group_mlp_fusion(torch.cat([src4,src8],-1))

                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([tokens,src1234],0)  # for multi-layer trans
                elif self.use_mlp_as_query:
                    #   TODO value should be change to src1,src2
                    if self.src_as_value:
                        cross_src1 = self.time_attn(k1, q1, value=src1, attn_mask=None)[0]
                        src1 = self.ffn1(src_merge_list[0],cross_src1)
                        cross_src1 = self.time_attn(k1, q5, value=src5, attn_mask=None)[0]
                        src1 = self.ffn1(src1,cross_src1)

                        cross_src2 = self.time_attn(k2, q2, value=src2, attn_mask=None)[0]
                        src2 = self.ffn1(src_merge_list[1],cross_src2)
                        cross_src2 = self.time_attn(k2, q6, value=src6, attn_mask=None)[0]
                        src2 = self.ffn1(src2,cross_src2)

                        cross_src3 = self.time_attn(k3, q3, value=src3, attn_mask=None)[0]
                        src3 = self.ffn1(src_merge_list[2],cross_src3)
                        cross_src3 = self.time_attn(k3, q7, value=src7, attn_mask=None)[0]
                        src3 = self.ffn1(src3,cross_src3)

                        cross_src4 = self.time_attn(k4, q4, value=src4, attn_mask=None)[0]
                        src4 = self.ffn1(src_merge_list[3],cross_src4)
                        cross_src4 = self.time_attn(k4, q8, value=src8, attn_mask=None)[0]
                        src4 = self.ffn1(src4,cross_src4)

                        src1234 = torch.cat([src1,src2,src3,src4],1)
                        src = torch.cat([tokens,src1234],0)  # for multi-layer trans

                    else:
                        cross_src1 = self.time_attn(k1, q1, value=src_merge_list[0], attn_mask=None)[0]
                        src1 = self.ffn1(src1,cross_src1)
                        cross_src1 = self.time_attn(k1, q5, value=src_merge_list[0], attn_mask=None)[0]
                        src1 = self.ffn1(src1,cross_src1)

                        cross_src2 = self.time_attn(k2, q2, value=src_merge_list[1], attn_mask=None)[0]
                        src2 = self.ffn1(src2,cross_src2)
                        cross_src2 = self.time_attn(k2, q6, value=src_merge_list[1], attn_mask=None)[0]
                        src2 = self.ffn1(src2,cross_src2)

                        cross_src3 = self.time_attn(k3, q3, value=src_merge_list[2], attn_mask=None)[0]
                        src3 = self.ffn1(src3,cross_src3)
                        cross_src3 = self.time_attn(k3, q7, value=src_merge_list[2], attn_mask=None)[0]
                        src3 = self.ffn1(src3,cross_src3)

                        cross_src4 = self.time_attn(k4, q4, value=src_merge_list[3], attn_mask=None)[0]
                        src4 = self.ffn1(src4,cross_src4)
                        cross_src4 = self.time_attn(k4, q8, value=src_merge_list[3], attn_mask=None)[0]
                        src4 = self.ffn1(src4,cross_src4)

                        src1234 = torch.cat([src1,src2,src3,src4],1)
                        src = torch.cat([tokens,src1234],0)  # for multi-layer trans

                else:
                    cross_src1 = self.time_attn(q1, k1, value=src_merge_list[0], attn_mask=None)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src2 = self.time_attn(q2, k2, value=src_merge_list[1], attn_mask=None)[0]
                    src2 = self.ffn1(src2,cross_src2)
                    cross_src3 = self.time_attn(q3, k3, value=src_merge_list[2], attn_mask=None)[0]
                    src3 = self.ffn1(src3,cross_src3)
                    cross_src4 = self.time_attn(q4, k4, value=src_merge_list[3], attn_mask=None)[0]
                    src4 = self.ffn1(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([tokens,src1234],0)  # for multi-layer trans
            else:
                cross_src1 = self.time_attn1(q1, k1, value=src_merge_list[0], attn_mask=None)[0]
                src1 = self.ffn1(src1,cross_src1)
                cross_src2 = self.time_attn2(q2, k2, value=src_merge_list[1], attn_mask=None)[0]
                src2 = self.ffn1(src2,cross_src2)
                cross_src3 = self.time_attn3(q3, k3, value=src_merge_list[2], attn_mask=None)[0]
                src3 = self.ffn1(src3,cross_src3)
                cross_src4 = self.time_attn4(q4, k4, value=src_merge_list[3], attn_mask=None)[0]
                src4 = self.ffn1(src4,cross_src4)
                src1234 = torch.cat([src1,src2,src3,src4],1)
                src = torch.cat([tokens,src1234],0)  # for multi-layer trans

        else:
            
            if self.use_grid_pos:
                q = k = self.with_pos_embed(src[1:], pos[1:,:1].repeat(1,src.shape[1],1))

            src2 = self.self_attn(q, k, value=src[1:])[0]
            src = src[1:] + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            group_length = self.num_frames//self.merge_groups
            tokens = src[0:1,:src.shape[1]//group_length]
            src_list = src.chunk(group_length,1)
            src1 = src_list[0]
            src_merge = torch.cat(src_list, 2)
            src_merge = self.time_mlp_fusion(src_merge)
            if self.use_grid_pos:
                q = self.with_pos_embed(src1, pos[1:,:1].repeat(1,src1.shape[1],1))
                k = self.with_pos_embed(src_merge, pos[1:,:1].repeat(1,src1.shape[1],1))
            cross_src1 = self.time_attn1(q, k, value=src_merge, attn_mask=None)[0]
            src1 = self.ffn1(src1,cross_src1)
            src = torch.cat([tokens,src1],0)
        # src2 = self.self_attn2(src1, src1, value=src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # src  = self.ffn2(src1,src2)
        # tokens = torch.cat([src[0:1],tokens[:,tokens.shape[1]//4:,:]],1)


        return src,tokens


    def forward(self, src=None,tokens=None,pos=None):
        return self.forward_post(src, tokens,pos)

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,split_time=False,use_channel_weight=True):
        super().__init__()

        self.split_time = split_time
        # if channeltime:
        #     encoder_layer1 = TransformerEncoderLayerTimeQKV(d_model, nhead, dim_feedforward,
        #                                             dropout, activation, normalize_before)
        #     encoder_layer2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                     dropout, activation, normalize_before)
        #     encoder_layer3 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                     dropout, activation, normalize_before)
        #     encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        #     encoder_layer = [encoder_layer1,encoder_layer2,encoder_layer3]
        #     self.encoder = TransformerEncoderTimeQKV(encoder_layer, num_encoder_layers, encoder_norm)
     
        # else:
        encoder_layer = [TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoderCT3D(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,use_channel_weight,split_time)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):

        BS, N, C = src.shape
        if self.split_time:
            src = src.view(BS*4,N//4,C)
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, BS, 1) #torch.Size([1, 128, 256]) num_query, bs, hideen_dim
        tgt = torch.zeros_like(query_embed) #torch.Size([1, 128, 256])
        memory,_ = self.encoder(src, src_key_padding_mask=None, pos=None,) # num_point,bs,feat torch.Size([128, 128, 256])
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,pos=None, query_pos=query_embed)
        if hs.shape[1] >1 :
            hs = hs.mean(1,True)
        return hs.transpose(1, 2), None

class TransformerTwins(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,channeltime=False,globalattn=False,usegloballayer=True):
        super().__init__()

        self.channeltime = channeltime
        if channeltime:
            encoder_layer1 = TransformerEncoderLayerTimeQKV(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_layer2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
            encoder_layer3 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            encoder_layer = [encoder_layer1,encoder_layer2,encoder_layer3]
            self.encoder = TransformerEncoderTimeQKV(encoder_layer, num_encoder_layers, encoder_norm)

        elif globalattn:
            encoder_layer = TransformerEncoderLayerWithGlobal(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,usegloballayer)

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

            
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):

        BS, N, C = src.shape
        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        src = src.contiguous().view(src.shape[0]//4,-1,src.shape[-1])
        pos_embed = pos_embed.contiguous().view(src.shape[0], -1,src.shape[-1])

        query_embed = query_embed.unsqueeze(1).repeat(1, BS, 1) #torch.Size([1, 128, 256]) num_query, bs, hideen_dim
        tgt = torch.zeros_like(query_embed) #torch.Size([1, 128, 256])
        
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed,) # num_point,bs,feat torch.Size([128, 128, 256])
        #memory =  memory.permute(1, 0, 2)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).contiguous().view(BS, C, N)

class TransformerDeit(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,
                 use_decoder=False,globalattn=False,usegloballayer=True,num_point=None,):
        super().__init__()

        self.use_decoder = use_decoder
        encoder_layer = [TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_point*4+1, d_model))

        if self.use_decoder:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        BS, N, C = src.shape
        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        cls_token = self.cls_token.repeat(BS,1,1)
        # pos_embed = self.pos_embed.repeat(BS,1,1)
        src = torch.cat([cls_token,src],dim=1)
        # src = src + pos_embed
        src = src.permute(1, 0, 2)
 
        memory = self.encoder(src, src_key_padding_mask=None, pos=None) # num_point,bs,feat torch.Size([128, 128, 256])
        # import pdb;pdb.set_trace()
        # show_feature_map(attn)
        if self.use_decoder:
            tgt = memory[0:1,:,:]
            memory = memory[1:]
            hs = self.decoder(tgt, memory, memory_key_padding_mask=None,pos=None, query_pos=None).squeeze()
            return hs, None
        else:
            memory = memory.permute(1, 0, 2)
            return memory[:,0,:], None

class TransformerDeitTime(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,use_learn_time_token=False,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,time_attn=False,
                 num_queries=None,use_channel_weight=False,uselearnpos=False,num_point=None,mlp_merge_query=False,
                 point128VS384=False,point128VS128x3=False,time_point_attn_pre=False,add_cls_token=False):
        super().__init__()


        self.mlp_merge_query = mlp_merge_query
        self.num_queries = num_queries
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.use_learn_time_token = use_learn_time_token

        if self.use_learn_time_token:
            self.time_token = MLP(input_dim = 1, hidden_dim = 256, output_dim = d_model, num_layers = 2)

        if self.mlp_merge_query:
            self.mlp_merge = MLP(input_dim = 4, hidden_dim = 0, output_dim = 1, num_layers = 1)
        encoder_layer = [TransformerEncoderLayerCrossAttn(d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before, num_point,use_channel_weight,
                        time_attn,point128VS384,point128VS128x3,use_learn_time_token) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.num_point = num_point
        self.add_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(num_queries, 1, d_model))
        self.time_index = torch.tensor([1,2,3,4]).view([4,1]).cuda()
        
        if uselearnpos:
            self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        BS, N, C = src.shape
        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        if self.num_queries > 1:
            cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
            cls_token2 = self.cls_token[1:2].repeat(BS,1,1)
            cls_token3 = self.cls_token[2:3].repeat(BS,1,1)
            cls_token4 = self.cls_token[3:4].repeat(BS,1,1)
        else:
            cls_token1 = cls_token2 = cls_token3 = cls_token4 = self.cls_token.repeat(BS,1,1)

        #time_token1,time_token2,time_token3,time_token4 = self.time_token.repeat(BS,1,1).chunk(4,1)
        import pdb;pdb.set_trace()
        if self.use_learn_time_token:
            time_token1,time_token2,time_token3,time_token4 = self.time_token(self.time_index).unsqueeze(1).repeat(1,self.num_point,1).chunk(4,0)
            src1 = torch.cat([reg_token1,time_token1+src[:,0:self.num_point]],dim=1)
            src2 = torch.cat([reg_token2,time_token2+src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,time_token3+src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,time_token4+src[:,3*self.num_point:]],dim=1)
        else:
            src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)

        # pos_embed2 = self.pos_embed[1:2].repeat(BS,1,1)
        # src2 = src2 + pos_embed2
        # pos_embed3 = self.pos_embed[2:3].repeat(BS,1,1)
        # src3 = src3 + pos_embed3
        # src4 = torch.cat([cls_token4,src[:,3*self.num_point:]],dim=1)
        # pos_embed4 = self.pos_embed[3:4].repeat(BS,1,1)
        # src4 = src4 + pos_embed4
        # pos_embed1 = self.pos_embed[0:1].repeat(BS,1,1)
        # src1 = src1 + pos_embed1

        src = torch.cat([src1,src2,src3,src4],dim=0)
        src = src.permute(1, 0, 2)
        # time_token = time_token.permute(1, 0, 2)
        # pos = torch.cat([pos_embed1, pos_embed2, pos_embed3, pos_embed3],dim=0)
        # pos = pos.permute(1, 0, 2)
        # import pdb;pdb.set_trace()
        memory = self.encoder(src, src_key_padding_mask=None, pos=None) # num_point,bs,feat torch.Size([128, 128, 256])

        if self.mlp_merge_query:
            memory = torch.cat(memory[0:1].chunk(4,dim=1),0).permute(2,1,0).contiguous()
            memory = self.mlp_merge(memory).permute(2,1,0).contiguous()
        else:
            # if self.weighted_sum:
            #     memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
            #     weight = F.softmax(memory,0)
            #     memory_weighted = (memory*weight).sum(0,True)
            # else:
            memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
            # memory = memory.mean(0,True)
        memory_weighted = memory.permute(1, 0, 2)
        return memory_weighted, None

class TransformerDeit128x384(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,use_learn_time_token=False,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,time_attn=False,
                 num_queries=None,use_channel_weight=False,uselearnpos=False,num_point=None,mlp_residual=False,
                 point128VS384=False,point128VS128x3=False,clstoken1VS384=False,time_attn_type=False,
                 use_decoder=False,use_t4_decoder=True,tgt_after_mean=True,tgt_before_mean=True,multi_decoder=False,weighted_sum=False,
                 channelwise_decoder=False,add_cls_token=False,share_head=True,p4conv_merge=False,masking_radius=None,num_frames=None,
                 fusion_type=None,fusion_mlp_norm =False,sequence_stride=None,channel_time=None,ms_pool=None,pyramid=False,use_grid_pos=False,
                 mlp_cross_grid_pos=False,merge_groups=False,fusion_init_token=False,use_box_pos=False,update_234=False,use_1_frame=False,
                 crossattn_last_layer=False, share_sa_layer=False):
        super().__init__()

        self.config = config
        self.mlp_residual = mlp_residual
        self.num_queries = num_queries
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.use_learn_time_token = use_learn_time_token
        self.use_decoder = use_decoder
        self.use_t4_decoder = use_t4_decoder
        self.multi_decoder = multi_decoder
        self.weighted_sum = weighted_sum
        self.add_cls_token = add_cls_token
        self.share_head = share_head
        self.masking_radius = masking_radius
        self.num_frames = num_frames
        self.nhead = nhead
        self.fusion_type = fusion_type
        self.fusion_mlp_norm = fusion_mlp_norm
        self.sequence_stride = sequence_stride
        self.channel_time = channel_time
        self.time_attn_type = time_attn_type
        self.merge_groups = merge_groups
        self.fusion_init_token = fusion_init_token
        self.use_1_frame  = use_1_frame
        self.crossattn_last_layer = crossattn_last_layer
        if self.config.use_fc_token.enabled:
            if self.config.use_fc_token.share:
                self.fc_token = MLP(input_dim = 64*256, hidden_dim = 256, output_dim = d_model, num_layers = 3)
            else:
                fc_layers = [MLP(input_dim = 64*256, hidden_dim = 256, output_dim = d_model, num_layers = 3) for i in range(3)]
                self.fc_token = nn.ModuleList(fc_layers)
        else:
            self.fc_token = None

        if self.channel_time:

            encoder_layer = [TransformerEncoderLayerChannelTime(d_model, nhead, dim_feedforward,
                            dropout, activation, normalize_before, num_point,use_channel_weight,
                            time_attn,time_attn_type,share_head = share_head) for i in range(num_encoder_layers)]
        else:

            encoder_layer = [TransformerEncoderLayerCrossAttn(self.config, d_model, d_model, nhead, dim_feedforward,
                            dropout, activation, normalize_before, num_point,use_channel_weight,time_attn,time_attn_type,share_head = share_head, use_box_pos=use_box_pos.enabled,
                            ms_pool=ms_pool,pyramid=pyramid,use_grid_pos=use_grid_pos,mlp_cross_grid_pos=mlp_cross_grid_pos,merge_groups=merge_groups,
                            update_234=update_234,crossattn_last_layer=crossattn_last_layer,share_sa_layer=share_sa_layer, add_extra_sa = self.config.add_extra_sa
                            ,fc_token=self.fc_token) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)
        self.p4conv_merge = p4conv_merge
        #self.encoder = nn.ModuleList(encoder_layer)

        # self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reg_token = nn.Parameter(torch.zeros(self.num_frames, 1, d_model))
        if self.use_learn_time_token:
            self.time_token = MLP(input_dim = 1, hidden_dim = 256, output_dim = d_model, num_layers = 2)

        self.time_index = torch.tensor([0,1,2,3]).view([4,1]).float().cuda()

        
        if self.p4conv_merge:

            if self.fusion_type == 'trans':
                # if self.num_frames > 4:
                #     self.mlp_merge = [CrossAttn(channel_time = self.channel_time).cuda() for i in range(4)]
                # if self.config.share_fusion_head:
                self.merge = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos,fusion_init_token=fusion_init_token,
                                                merge_groups=self.merge_groups,num_frames=self.num_frames,share_head=self.config.share_fusion_head,
                                                group_concat = self.config.group_concat,use_mlp_as_query=self.config.use_mlp_as_query,src_as_value=self.config.src_as_value)
                # else:
                #     self.merge_list = [CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos,fusion_init_token=fusion_init_token,
                #                                merge_groups=self.merge_groups,num_frames=self.num_frames).cuda() for i in range(self.merge_groups)]
                # self.mlp_merge2 = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos).cuda()
                # self.mlp_merge3 = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos).cuda()

            elif 'mlp_mixer' in self.fusion_type:
                self.mlp_mixer = SpatialMixerBlock(256)
                if self.merge_groups:
                    group = self.num_frames // self.merge_groups
                self.merge = MLP(input_dim = 256*group, hidden_dim = 256, output_dim = 256, num_layers = 4)

            else:
  
                if self.merge_groups:
                    group = self.num_frames // self.merge_groups
                self.merge = MLP(input_dim = self.config.hidden_dim*group, hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

                self.fusion_norm = FFN(d_model, dim_feedforward)
                self.layernorm = nn.LayerNorm(d_model)



        if uselearnpos:
            self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))

        if self.use_decoder.enabled:
            # if multi_decoder:
            #     decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
            #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            # elif channelwise_decoder.enabled:
            #     decoder_layer = TransformerDecoderLayerChannelwise(d_model, nhead, dim_feedforward,
            #                                 dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean,channelwise_decoder.weighted)
            if   self.use_decoder.name=='casc1':

                decoder_layer = TransformerDecoderLayerDeitCascaded1(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc2':

                decoder_layer = TransformerDecoderLayerDeitCascaded2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc3':

                decoder_layer = TransformerDecoderLayerDeitCascaded3(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)

            else:
                decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,self.use_decoder)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, src, src_mask=None,pos=None,num_frames=None,pos_pyramid=None,box_pos=None,center_token=None,empty_ball_mask=None,batch_dict=None):

        BS, N, C = src.shape
        self.num_point = N//num_frames
        src_merge = None
        group = self.merge_groups
        if not pos is None:
            pos = pos.permute(1, 0, 2)

        if pos_pyramid is not None:
            pos_pyramid = pos_pyramid.permute(1, 0, 2)

        if self.p4conv_merge and num_frames == 4:
            if self.fusion_type == 'mlp':
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                src = src.view(src.shape[0],src.shape[1]//group,-1)
                if self.mlp_residual:
                    src = src + self.merge(src)
                    # src = self.fusion_norm(src)
                else:
                    src = self.merge(src)
                src_merge = torch.max(src, 1, keepdim=True)[0]
                src = torch.cat([reg_token1,F.relu(src)],dim=1)
            else:
                if self.fusion_init_token:
                    reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                    reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                    reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                    reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)

                    if self.merge_groups==1:
                        src = torch.cat([src1,src2,src3,src4],dim=0)
                        src = src.permute(1, 0, 2)
                        src, tokens = self.merge(src,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        # src = src[:,:src.shape[1]//4]
                        src = src.permute(1, 0, 2)
                    elif self.merge_groups==2:
                        src1 = torch.cat([src1,src3],dim=0).permute(1, 0, 2)
                        src2 = torch.cat([src2,src4],dim=0).permute(1, 0, 2)
                        # src = src.permute(1, 0, 2)
                        src1, tokens = self.merge(src1,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        src2, tokens = self.merge(src2,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        src = torch.cat([src1,src2],1)
                        src = src.permute(1, 0, 2)

                    
                else:
                    reg_token1 = self.reg_token[0:1].repeat(BS,1,1).permute(1,0,2)
                    src = torch.cat(src.chunk(4,1),dim=0)
                    src = src.permute(1, 0, 2)
                    src, tokens = self.mlp_merge1(src,tokens=None,pos=pos)
                    src = torch.cat([reg_token1,src[:,0:src.shape[1]//4]],dim=0)
                    src = src.permute(1, 0, 2)

                # src = torch.cat([reg_token1,src_merge],dim=1)


        elif self.p4conv_merge and num_frames == 8:

            if self.fusion_type == 'mlp':
 
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                current_num = 4
                
                if self.fusion_mlp_norm=='ffn':
                    src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            elif self.fusion_type == 'mlp_res_cat':
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
                # src = self.mlp_mixer(src.permute(1,0,2)).permute(1,0,2).contiguous()
                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                
                current_num = 4

                if self.fusion_mlp_norm=='ffn':
                    src_1 = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                    src_2 = self.fusion_norm(src[:,current_num*64:],self.merge(src_merge))
                    src_merge = torch.cat([src_1,src_2],-1)
                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                src = self.merge(src_merge)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            elif self.fusion_type == 'mlp_mixer_cat':

                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
                src = src.view(-1,self.num_point,src.shape[-1]).permute(1,0,2) #[N,BS,C]
                src = self.mlp_mixer(src).permute(1,0,2).contiguous().view(BS,N,C)
                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                src = self.merge(src_merge)
                current_num = 4
                
                # if self.fusion_mlp_norm=='ffn':
                #     src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                # elif self.fusion_mlp_norm=='layernorm':
                #     src = src[:,:current_num*64] + self.merge(src_merge)
                #     src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            elif self.fusion_type == 'mlp_trans':

                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                current_num = 4
                
                if self.fusion_mlp_norm=='ffn':
                    src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)

                # src2,attn_weight = self.self_attn(src[0:1], src[0:1], value=src[1:], attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
                k = with_pos_embed(src[1:], pos[1:])
                src,_ = self.merge(src,tokens=None,pos=pos)
                src = src.permute(1, 0, 2)



            else:

                if not center_token is None:
                    reg_token_list = [ center_token[:,i:i+1].repeat(BS,1,1) for i in range(num_frames) ]
                else:
                    reg_token_list = [ self.reg_token[i:i+1].repeat(BS,1,1) for i in range(num_frames) ]
                # reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                # reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                # reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([zeros,     src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([zeros,     src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token3,src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([zeros,     src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([reg_token4,src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([zeros,     src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=0)
                    src = src.permute(1, 0, 2)
                    src_groups = src.chunk(4,dim=1)
                    # src_merge = torch.cat(src_groups,1)
                    # current_num = 8//group
                    src_merge1 = self.merge(src_groups[0])
                    src_merge2 = self.merge(src_groups[1])
                    src_merge3 = self.merge(src_groups[2])
                    src_merge4 = self.merge(src_groups[3])
                    src_merge = torch.cat([src_merge1,src_merge2,src_merge3,src_merge4],1)

                    # src = torch.cat([src[:,0:128],src[:,128*2:128*3],src[:,128*4:128*5],src[:,128*6:128*7]],1)
                    # src = src + src_merge

                elif self.sequence_stride > 1:

                    # zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token_list[0],src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([reg_token_list[1],src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token_list[2],src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token_list[3],src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token_list[4],src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([reg_token_list[5],src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([reg_token_list[6],src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([reg_token_list[7],src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src  = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=1)
                    # src  = src.permute(1, 0, 2)

                    length = num_frames//4
                    src_groups = []
                    group_index = [[0,4],[1,5],[2,6],[3,7]]
                    for i in range(4):
                        indexs = group_index[i]
                        groups = [src[:,idx*65:(idx+1)*65] for idx in indexs]
                        groups = torch.cat(groups,0)
                        src_groups.append(groups)

                    # import pdb;pdb.set_trace()
                    src = torch.cat(src_groups,0).view(-1, 65,src.shape[-1]).permute(1,0,2).contiguous()
                    src,_ = self.merge(src,tokens=None,pos=pos)
                    src = src.permute(1, 0, 2)

        elif self.p4conv_merge and num_frames == 12:
            # import pdb;pdb.set_trace()
            if self.fusion_type == 'mlp':
 
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                current_num = 4
                
                if self.fusion_mlp_norm=='ffn':
                    src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            else:

                if not center_token is None:
                    reg_token_list = [ center_token[:,i:i+1].repeat(BS,1,1) for i in range(num_frames) ]
                else:
                    reg_token_list = [ self.reg_token[i:i+1].repeat(BS,1,1) for i in range(num_frames) ]
                # reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                # reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                # reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([zeros,     src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([zeros,     src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token3,src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([zeros,     src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([reg_token4,src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([zeros,     src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=0)
                    src = src.permute(1, 0, 2)
                    src_groups = src.chunk(4,dim=1)
                    # src_merge = torch.cat(src_groups,1)
                    # current_num = 8//group
                    src_merge1 = self.merge(src_groups[0])
                    src_merge2 = self.merge(src_groups[1])
                    src_merge3 = self.merge(src_groups[2])
                    src_merge4 = self.merge(src_groups[3])
                    src_merge = torch.cat([src_merge1,src_merge2,src_merge3,src_merge4],1)

                    # src = torch.cat([src[:,0:128],src[:,128*2:128*3],src[:,128*4:128*5],src[:,128*6:128*7]],1)
                    # src = src + src_merge

                elif self.sequence_stride > 1:

                    # zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token_list[0],src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([reg_token_list[1],src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token_list[2],src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token_list[3],src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token_list[4],src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([reg_token_list[5],src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([reg_token_list[6],src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([reg_token_list[7],src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src  = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=1)
                    # src  = src.permute(1, 0, 2)

                    length = num_frames//4
                    src_groups = []
                    group_index = [[0,4],[1,5],[2,6],[3,7]]
                    for i in range(4):
                        indexs = group_index[i]
                        groups = [src[:,idx*65:(idx+1)*65] for idx in indexs]
                        groups = torch.cat(groups,0)
                        src_groups.append(groups)

                    # import pdb;pdb.set_trace()
                    src = torch.cat(src_groups,0).view(-1, 65,src.shape[-1]).permute(1,0,2).contiguous()
                    src,_ = self.merge(src,tokens=None,pos=pos)
                    # src_merge_list = []
                    # if self.config.share_fusion_head:
                    #     for i in range(self.merge_groups):
                    #         src_merge_list.append(self.merge(src_groups[i],tokens=None,pos=pos)[0])
                    #     # src_merge2,_ = self.merge(src_groups[1],tokens=None,pos=pos)
                    #     # src_merge3,_ = self.merge(src_groups[2],tokens=None,pos=pos)
                    #     # src_merge4,_ = self.merge(src_groups[3],tokens=None,pos=pos)
                    # else:

                    #     for i in range(self.merge_groups):
                    #         src_merge_list.append(self.merge_list[i](src_groups[i],tokens=None,pos=pos)[0])
                            # src_merge2,_ = self.merge2(src_groups[1],tokens=None,pos=pos)
                            # src_merge3,_ = self.merge3(src_groups[2],tokens=None,pos=pos)
                            # src_merge4,_ = self.merge4(src_groups[3],tokens=None,pos=pos)

                    # src = torch.cat(src_merge_list,1)
                    src = src.permute(1, 0, 2)

        elif self.p4conv_merge and num_frames == 16:

            if self.fusion_type == 'mlp':
 
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//group
                    src_groups = []
                    for idx, i in enumerate(range(group)):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]

                        # if batch_dict['use_future_frames'][0]:
                        #     if idx==0 or idx==1:
                        #         item = groups.pop(2)   
                        #         groups.insert(0, item)
                        #         # [8,4,0,-4]>[0,8,4,-4], [7,3,-1,-5] > [-1,7,3,-5]
                        #     else:
                        #         item = groups.pop(1)   
                        #         groups.insert(0, item)  
                        #          # [6,2,-2,-6]>[2,6,-2,-6], [5,1,-3,-7] > [1,5,3,7]

                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                current_num = 4
                

                if self.fusion_mlp_norm=='ffn':
                    # if batch_dict['use_future_frames'][0]:
                    #     src_near_t0 = torch.cat([src[:,(num_frames//2)*64 : (num_frames//2+1)*64],src[:,(num_frames//2 + 1)*64 : (num_frames//2 +2)*64],
                    #                              src[:,(num_frames//2-2)*64 : (num_frames//2-1)*64],src[:,(num_frames//2-1)*64 : (num_frames//2)*64],],1)
                    #     #[0, -1, 2, 1]
                    #     src = self.fusion_norm(src_near_t0,self.merge(src_merge))
                    # else:
                    src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            elif self.fusion_type == 'max_pool':

                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//group
                    src_groups = []
                    for idx, i in enumerate(range(group)):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64].unsqueeze(2) for j in range(length)]

                        # if batch_dict['use_future_frames'][0]:
                        #     if idx==0 or idx==1:
                        #         item = groups.pop(2)   
                        #         groups.insert(0, item)
                        #         # [8,4,0,-4]>[0,8,4,-4], [7,3,-1,-5] > [-1,7,3,-5]
                        #     else:
                        #         item = groups.pop(1)   
                        #         groups.insert(0, item)  
                        #          # [6,2,-2,-6]>[2,6,-2,-6], [5,1,-3,-7] > [1,5,3,7]

                        groups = torch.cat(groups,2).max(2)[0]
                        src_groups.append(groups)

                src_merge = torch.cat(src_groups,1)
                current_num = 4
                

                if self.fusion_mlp_norm=='ffn':
                    # if batch_dict['use_future_frames'][0]:
                    #     src_near_t0 = torch.cat([src[:,(num_frames//2)*64 : (num_frames//2+1)*64],src[:,(num_frames//2 + 1)*64 : (num_frames//2 +2)*64],
                    #                              src[:,(num_frames//2-2)*64 : (num_frames//2-1)*64],src[:,(num_frames//2-1)*64 : (num_frames//2)*64],],1)
                    #     #[0, -1, 2, 1]
                    #     src = self.fusion_norm(src_near_t0,src_merge)
                    # else:
                    src = self.fusion_norm(src[:,:current_num*64],src_merge)

                elif self.fusion_mlp_norm=='layernorm':
                    src = src[:,:current_num*64] + self.merge(src_merge)
                    src = self.layernorm(src)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)


        # elif self.p4conv_merge and num_frames == 32:

        #     if self.fusion_type == 'mlp':
 
        #         reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
        #         reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
        #         reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
        #         reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

        #         if self.sequence_stride ==1:
        #             src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
        #         elif self.sequence_stride > 1:
        #             length = num_frames//4
        #             src_groups = []
        #             for idx, i in enumerate(range(group)):
        #                 groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
        #                 if idx==0 or idx==1:
        #                     groups = groups[2,0,1,3] # [8,4,0,-4]>[0,8,4,-4], [7,3,-1,-5] > [-1,7,3,-5]
        #                 else:
        #                     groups = groups[1,0,2,3] # [6,2,-2,-6]>[2,6,-2,-6], [5,1,-3,-7] > [1,5,3,7]
        #                 groups = torch.cat(groups,-1)
        #                 src_groups.append(groups)
        #         if batch_dict['use_future_frames'][0]:
        #             src_merge = torch.cat(src_groups[2,3,0,1],1)
        #         else:
        #             src_merge = torch.cat(src_groups,1)
        #         current_num = group

        #         if self.fusion_mlp_norm=='ffn':
        #             if batch_dict['use_future_frames'][0]:
        #                 src_near_t0 = torch.cat([src[:,(current_num//2)*64 : (current_num//2+1)*64],src[:,(current_num//2 -1)*64 : (current_num//2)*64],
        #                                          src[:,(current_num//2+2)*64 : (current_num//2+3)*64],src[:,(current_num//2+1)*64 : (current_num//2+2)*64],],1)
        #                 #[0, -1, 2, 1]
        #                 src = self.fusion_norm(src_near_t0,self.merge(src_merge))
        #             else:
        #                 src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
        #         elif self.fusion_mlp_norm=='layernorm':
        #             src = src[:,:current_num*64] + self.merge(src_merge)
        #             src = self.layernorm(src)

        #         if current_num == 4:
        #             src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
        #             src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
        #             src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
        #             src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
        #             src = torch.cat([src1,src2,src3,src4],dim=0)
        #         else:
        #             src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

        else:

            if self.use_1_frame:
                reg_token1 =  self.reg_token[0:1].repeat(BS,1,1)
                src = torch.cat([reg_token1,src],dim=1)
            else:
                # import pdb;pdb.set_trace()
                if self.num_queries > 1:
                    if not center_token is None and not self.config.use_center_token_add:
                        if self.config.share_center_token:
                            reg_token1 = reg_token2 = reg_token3 = reg_token4 = center_token[:,0:1]
                        else:
                            reg_token1 = center_token[:,0:1]
                            reg_token2 = center_token[:,1:2]
                            reg_token3 = center_token[:,2:3]
                            reg_token4 = center_token[:,3:4]
                    else:
                        cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
                        reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                        reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                        reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                        reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
                else:
                    reg_token1 =  self.reg_token.repeat(BS,1,1)
                    reg_token2 = src[:,self.num_point:self.num_point+1]
                    reg_token3 = src[:,2*self.num_point:2*self.num_point+1]
                    reg_token4 = src[:,3*self.num_point:3*self.num_point+1]

                #time_token1,time_token2,time_token3,time_token4 = self.time_token.repeat(BS,1,1).chunk(4,1)
                
                if self.add_cls_token:
                    src1 = torch.cat([cls_token1,reg_token1,src[:,0:self.num_point-1]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                else:
                    if self.use_learn_time_token:
                        time_token1,time_token2,time_token3,time_token4 = self.time_token(self.time_index).unsqueeze(1).repeat(1,self.num_point,1).chunk(4,0)
                        src1 = torch.cat([reg_token1,time_token1+src[:,0:self.num_point]],dim=1)
                        src2 = torch.cat([reg_token2,time_token2+src[:,self.num_point:2*self.num_point]],dim=1)
                        src3 = torch.cat([reg_token3,time_token3+src[:,2*self.num_point:3*self.num_point]],dim=1)
                        src4 = torch.cat([reg_token4,time_token4+src[:,3*self.num_point:]],dim=1)
                    else:
                        src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                        src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                        src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                        src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)



                src = torch.cat([src1,src2,src3,src4],dim=0)


        src = src.permute(1, 0, 2)
        if self.config.use_fc_token.enabled:
            src = src[1:]
        # time_token = self.time_token.repeat(BS,1,1)
        # time_token = time_token.permute(1, 0, 2)
        # import pdb;pdb.set_trace()

        memory,tokens = self.encoder(src, mask = src_mask, num_frames=num_frames,pos_pyramid=pos_pyramid,pos=pos,
                                     box_pos=box_pos,center_token=center_token,empty_ball_mask=empty_ball_mask) # num_point,bs,feat torch.Size([128, 128, 256])

        if self.use_decoder.enabled:
            if self.num_queries > 1:
                # import pdb;pdb.set_trace()
                # tgt_cls = memory[0:1,:memory.shape[1]//4,:].contiguous()

                if self.use_decoder.local_decoder:
                    tgt = memory[0:1]
                    memory = memory[1:]

                else:
                    tgt = memory[0:1].view(-1, memory.shape[1]//4,memory.shape[-1])
                    memory = memory[1:].view(-1, memory.shape[1]//4,memory.shape[-1])


            else:
                tgt = memory[0:1,:,:]
                memory = memory[1:]
            hs = self.decoder(tgt, memory, memory_key_padding_mask=None,pos=pos, query_pos=None, grid_pos=None).squeeze()
            return hs, tokens, None
        else:

            if self.weighted_sum:
                memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                weight = F.softmax(memory,0)
                memory = (memory*weight).sum(0,True)
            else:
                if (self.p4conv_merge and num_frames==4 and self.merge_groups==1) or (self.time_attn_type in ['mlp_merge','trans_merge'] and self.merge_groups==1) \
                   or self.use_1_frame:
                    memory = memory[0:1]
                    return memory, tokens, src_merge
                elif self.merge_groups==2:
                    memory = torch.cat(memory[0:1].chunk(2,dim=1),0)
                    return memory, tokens, src_merge
                elif self.config.use_mlp_query_decoder:
                    return memory[0:1], tokens, src_merge
                else:
                    memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                    return memory, tokens, src_merge
            # else:
            #     memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
            #     memory= memory[0:1]
            memory = memory.permute(1, 0, 2)
            return memory, tokens, src_merge

class TransformerDeitClear(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,use_learn_time_token=False,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,time_attn=False,
                 num_queries=None,use_channel_weight=False,uselearnpos=False,num_point=None,mlp_residual=False,
                 point128VS384=False,point128VS128x3=False,clstoken1VS384=False,time_attn_type=False,
                 use_decoder=False,use_t4_decoder=True,tgt_after_mean=True,tgt_before_mean=True,multi_decoder=False,weighted_sum=False,
                 channelwise_decoder=False,add_cls_token=False,share_head=True,p4conv_merge=False,masking_radius=None,num_frames=None,
                 fusion_type=None,sequence_stride=None,channel_time=None,ms_pool=None,pyramid=False,use_grid_pos=False,
                 mlp_cross_grid_pos=False,merge_groups=False,fusion_init_token=False,use_box_pos=False,update_234=False,use_1_frame=False,
                 crossattn_last_layer=False, share_sa_layer=False):
        super().__init__()

        self.mlp_residual = mlp_residual
        self.num_queries = num_queries
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.use_learn_time_token = use_learn_time_token
        self.use_decoder = use_decoder
        self.use_t4_decoder = use_t4_decoder
        self.multi_decoder = multi_decoder
        self.weighted_sum = weighted_sum
        self.add_cls_token = add_cls_token
        self.share_head = share_head
        self.masking_radius = masking_radius
        self.num_frames = num_frames
        self.nhead = nhead
        self.fusion_type = fusion_type
        self.sequence_stride = sequence_stride
        self.channel_time = channel_time
        self.time_attn_type = time_attn_type
        self.merge_groups = merge_groups
        self.fusion_init_token = fusion_init_token
        self.use_1_frame  = use_1_frame
        self.crossattn_last_layer = crossattn_last_layer

        #     self.mlp_merge = MLP(input_dim = 4, hidden_dim = 0, output_dim = 1, num_layers = 1)

        if self.channel_time:

            encoder_layer = [TransformerEncoderLayerChannelTime(d_model, nhead, dim_feedforward,
                            dropout, activation, normalize_before, num_point,use_channel_weight,
                            time_attn,time_attn_type,share_head = share_head) for i in range(num_encoder_layers)]
        else:

            encoder_layer = [TransformerEncoderLayerCrossAttn(d_model, d_model, nhead, dim_feedforward,
                            dropout, activation, normalize_before, num_point,use_channel_weight,time_attn,time_attn_type,share_head = share_head, use_box_pos=use_box_pos.enabled,
                            ms_pool=ms_pool,pyramid=pyramid,use_grid_pos=use_grid_pos,mlp_cross_grid_pos=mlp_cross_grid_pos,merge_groups=merge_groups,
                            update_234=update_234,crossattn_last_layer=crossattn_last_layer,share_sa_layer=share_sa_layer) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.p4conv_merge = p4conv_merge

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reg_token = nn.Parameter(torch.zeros(self.num_frames, 1, d_model))
        if self.use_learn_time_token:
            self.time_token = MLP(input_dim = 1, hidden_dim = 256, output_dim = d_model, num_layers = 2)

        self.time_index = torch.tensor([0,1,2,3]).view([4,1]).float().cuda()

        self.fusion_norm = nn.LayerNorm(d_model)
        if self.p4conv_merge:

            if self.fusion_type == 'trans':

                self.merge = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos,fusion_init_token=fusion_init_token,
                                             merge_groups=self.merge_groups,num_frames=self.num_frames).cuda()
            else:

                if self.merge_groups:
                    group = 8 // self.merge_groups
                self.merge = MLP(input_dim = 256*group, hidden_dim = 256, output_dim = 256, num_layers = 4).cuda()

        if uselearnpos:
            self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))

        if self.use_decoder.enabled:

            if   self.use_decoder.name=='casc1':

                decoder_layer = TransformerDecoderLayerDeitCascaded1(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc2':

                decoder_layer = TransformerDecoderLayerDeitCascaded2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc3':

                decoder_layer = TransformerDecoderLayerDeitCascaded3(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            else:
                decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, src, src_mask=None,pos=None,num_frames=None,grid_pos=None,box_pos=None):

        BS, N, C = src.shape
        self.num_point = N//num_frames
        src_merge = None

        group = self.merge_groups

        if not pos is None:
            pos = pos.permute(1, 0, 2)

        # import pdb;pdb.set_trace()

        if self.p4conv_merge and num_frames == 4:
            if self.fusion_type == 'mlp':
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                src = src.view(src.shape[0],src.shape[1]//group,-1)
                if self.mlp_residual:
                    src = src + self.merge(src)
                    # src = self.fusion_norm(src)
                else:
                    src = self.merge(src)
                src_merge = torch.max(src, 1, keepdim=True)[0]
                src = torch.cat([reg_token1,F.relu(src)],dim=1)
            else:
                if self.fusion_init_token:
                    reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                    reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                    reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                    reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)

                    if self.merge_groups==1:
                        src = torch.cat([src1,src2,src3,src4],dim=0)
                        src = src.permute(1, 0, 2)
                        src, tokens = self.merge(src,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        # src = src[:,:src.shape[1]//4]
                        src = src.permute(1, 0, 2)
                    elif self.merge_groups==2:
                        src1 = torch.cat([src1,src3],dim=0).permute(1, 0, 2)
                        src2 = torch.cat([src2,src4],dim=0).permute(1, 0, 2)
                        # src = src.permute(1, 0, 2)
                        src1, tokens = self.merge(src1,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        src2, tokens = self.merge(src2,tokens=None,pos=pos) #src 65,512,256 ,token 1,512,256
                        src = torch.cat([src1,src2],1)
                        src = src.permute(1, 0, 2)

                    
                else:
                    reg_token1 = self.reg_token[0:1].repeat(BS,1,1).permute(1,0,2)
                    src = torch.cat(src.chunk(4,1),dim=0)
                    src = src.permute(1, 0, 2)
                    src, tokens = self.mlp_merge1(src,tokens=None,pos=pos)
                    src = torch.cat([reg_token1,src[:,0:src.shape[1]//4]],dim=0)
                    src = src.permute(1, 0, 2)

                # src = torch.cat([reg_token1,src_merge],dim=1)

        elif self.p4conv_merge and num_frames > 4:

            if self.fusion_type == 'mlp':

                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
                elif self.sequence_stride > 1:
                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*64:(i+j*4+1)*64] for j in range(length)]
                        groups = torch.cat(groups,-1)
                        src_groups.append(groups)
                src_merge = torch.cat(src_groups,1)
                current_num = 4
                src = src[:,:current_num*64] + self.merge(src_merge)

                if current_num == 4:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
                    src = torch.cat([src1,src2,src3,src4],dim=0)
                else:
                    src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            else:

                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

                if self.sequence_stride ==1:
                    zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([zeros,     src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token2,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([zeros,     src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([reg_token3,src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([zeros,     src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([reg_token4,src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([zeros,     src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=0)
                    src = src.permute(1, 0, 2)
                    src_groups = src.chunk(4,dim=1)

                    src_merge1 = self.merge(src_groups[0])
                    src_merge2 = self.merge(src_groups[1])
                    src_merge3 = self.merge(src_groups[2])
                    src_merge4 = self.merge(src_groups[3])
                    src_merge = torch.cat([src_merge1,src_merge2,src_merge3,src_merge4],1)

                elif self.sequence_stride > 1:

                    zeros = torch.zeros_like(reg_token1)
                    src0 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                    src1 = torch.cat([reg_token2,src[:,1*self.num_point:2*self.num_point]],dim=1)
                    src2 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                    src3 = torch.cat([reg_token4,src[:,3*self.num_point:4*self.num_point]],dim=1)
                    src4 = torch.cat([zeros,     src[:,4*self.num_point:5*self.num_point]],dim=1)
                    src5 = torch.cat([zeros,     src[:,5*self.num_point:6*self.num_point]],dim=1)
                    src6 = torch.cat([zeros,     src[:,6*self.num_point:7*self.num_point]],dim=1)
                    src7 = torch.cat([zeros,     src[:,7*self.num_point:8*self.num_point]],dim=1)
                    src  = torch.cat([src0,src1,src2,src3,src4,src5,src6,src7],dim=1)
                    # src  = src.permute(1, 0, 2)

                    length = num_frames//4
                    src_groups = []
                    for i in range(4):
                        groups = [src[:,(i+j*4)*65:(i+j*4+1)*65] for j in range(length)]
                        groups = torch.cat(groups,0).permute(1, 0, 2)
                        src_groups.append(groups)

                    src_merge1,_ = self.merge(src_groups[0],tokens=None,pos=pos)
                    src_merge2,_ = self.merge(src_groups[1],tokens=None,pos=pos)
                    src_merge3,_ = self.merge(src_groups[2],tokens=None,pos=pos)
                    src_merge4,_ = self.merge(src_groups[3],tokens=None,pos=pos)

                    src = torch.cat([src_merge1,src_merge2,src_merge3,src_merge4],1)
                    src = src.permute(1, 0, 2)
                    # current_num = 8//group
                    # src = src[:,:current_num*128] + src_merge


        else:

            reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
            reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
            reg_token4 = self.reg_token[3:4].repeat(BS,1,1)


            if self.use_learn_time_token:
                time_token1,time_token2,time_token3,time_token4 = self.time_token(self.time_index).unsqueeze(1).repeat(1,self.num_point,1).chunk(4,0)
                src1 = torch.cat([reg_token1,time_token1+src[:,0:self.num_point]],dim=1)
                src2 = torch.cat([reg_token2,time_token2+src[:,self.num_point:2*self.num_point]],dim=1)
                src3 = torch.cat([reg_token3,time_token3+src[:,2*self.num_point:3*self.num_point]],dim=1)
                src4 = torch.cat([reg_token4,time_token4+src[:,3*self.num_point:]],dim=1)
            else:
                src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)

            src = torch.cat([src1,src2,src3,src4],dim=0)


        src = src.permute(1, 0, 2)
        memory = self.encoder(src, mask = src_mask, num_frames=num_frames,grid_pos=grid_pos,pos=pos,box_pos=box_pos) # num_point,bs,feat torch.Size([128, 128, 256])
        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
        return memory, None, src_merge

class TransformerEncoderLayerCrossAttnClear(nn.Module):
    count = 0
    def __init__(self, d_model, dout, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,
                 use_channel_weight=False,time_attn=False,time_attn_type=False, use_motion_attn=False,share_head=True,
                 ms_pool=False,pyramid=False,use_grid_pos=False,use_box_pos=False,mlp_cross_grid_pos=False,merge_groups=None,
                 update_234=False,crossattn_last_layer=False,share_sa_layer=True):
        super().__init__()
        TransformerEncoderLayerCrossAttn.count += 1
        self.count = TransformerEncoderLayerCrossAttn.count

        if self.count == 3 and pyramid:
            d_model = 2*d_model
        self.ms_pool = ms_pool
        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.time_attn_type = time_attn_type
        self.use_motion_attn = use_motion_attn
        self.time_attn = time_attn
        self.pyramid = pyramid
        self.use_grid_pos = use_grid_pos
        self.mlp_cross_grid_pos = mlp_cross_grid_pos
        self.merge_groups = merge_groups
        self.update_234 = update_234
        self.crossattn_last_layer = crossattn_last_layer
        self.share_sa_layer = share_sa_layer

        if self.share_sa_layer.enabled:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            if self.share_sa_layer.share_ffn:
                self.sa_ffn = FFN(d_model, dim_feedforward)
            else:
                self.sa_ffn1 = FFN(d_model, dim_feedforward)
                self.sa_ffn2 = FFN(d_model, dim_feedforward)
                self.sa_ffn3 = FFN(d_model, dim_feedforward)
                self.sa_ffn4 = FFN(d_model, dim_feedforward)

        self.share_head = share_head
        self.use_box_pos = use_box_pos
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if d_model != dout:
            self.proj = nn.Linear(d_model, dout)

        if self.use_channel_weight=='ct3d':
            self.channel_attn = MultiHeadedAttention(d_model,nhead)
        elif self.use_channel_weight=='channelwise':
            self.channel_attn = MultiHeadedAttentionChannelwise(nhead, d_model)

        if self.time_attn:
            self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            if self.share_head:
                self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            else:
                self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            
            if use_motion_attn:
                self.time_point_attn = MultiHeadedAttentionMotion(nhead, d_model)
            else:
                self.time_point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)



            self.time_token_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_cls_token = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        if self.share_head:
            self.ffn1 = FFN(d_model, dim_feedforward)
        else:
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)
            self.ffn3 = FFN(d_model, dim_feedforward)
            self.ffn4 = FFN(d_model, dim_feedforward)

        if self.time_attn_type in ['time_mlp', 'crossattn_mlp', 'time_mlp_v2',]:
            self.time_mlp1 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp3 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp4 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            
            if self.count==3 and self.pyramid:
                self.time_mlp_fusion = MLP(input_dim = 512*4, hidden_dim = 512, output_dim = 512, num_layers = 4)
            else:
                self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)

            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

            if self.pyramid and self.count==2:
                self.ffn2 = FFNUp(d_model, dim_feedforward, dout=2*d_model)
                self.proj = nn.Linear(d_model, 2*d_model)
            else:
                self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'mlp_merge' and self.count==3:
            self.time_mlp1 = MLP(input_dim = 256*(merge_groups-1), hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'trans_merge' and self.count==3:
            self.time_mlp1 = MLP(input_dim = 256*merge_groups, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'trans_merge_cas':
            # self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_mlp_fusion1 = MLP(input_dim = 256*4, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion3 = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion4 = MLP(input_dim = 256*1, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.ffn1 = FFN(d_model, dim_feedforward)
            # self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'crossattn_trans':
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)

        if self.crossattn_last_layer:
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn_last = FFN(d_model, dim_feedforward)

        self.max2_pool = nn.MaxPool3d((2,2,2), (2,2,2), (0,0,0), ceil_mode=False)
        self.max4_pool = nn.MaxPool3d((4,4,4), (4,4,4), (0,0,0), ceil_mode=False)



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            if tensor.shape == pos.shape:
                return tensor + pos
            else:
                index = tensor.shape[0]
                pos = pos[:index]
                return tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     grid_pos = None,motion_feat= None,box_pos=None):

        src_ori = src
        
    
        src_rm_token = src[1:]
        q = k = self.with_pos_embed(src, pos)


        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        cls_token = None

        if self.time_attn:
            if self.time_attn_type == 'crossattn_mlp':

                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1_ori,src2_ori,src3_ori,src4_ori = src_512.chunk(4,0)

                if self.use_box_pos:
                    src = src + box_pos
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_fusion = torch.cat([src1,src2,src3,src4],-1)
                src_fusion = self.time_mlp_fusion(src_fusion)

                k = self.with_pos_embed(src_fusion, pos[1:])
                q1 = self.with_pos_embed(src1, pos[1:])
                q2 = self.with_pos_embed(src2, pos[1:])
                q3 = self.with_pos_embed(src3, pos[1:])
                q4 = self.with_pos_embed(src4, pos[1:])

        
                if self.share_head:
                    cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn2(src1,cross_src1)
                    if self.update_234:
                        cross_src2 = self.time_attn1(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src2 = self.ffn2(src2,cross_src2)
                        cross_src3 = self.time_attn1(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src3 = self.ffn2(src3,cross_src3)
                        cross_src4 = self.time_attn1(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src4 = self.ffn2(src4,cross_src4)
                        src1234 = torch.cat([src1,src2,src3,src4],1)
                    else:
                        src1234 = torch.cat([src1,src2_ori,src3_ori,src4_ori],1)
                    if self.pyramid and self.count==2:
                        src = torch.cat([self.proj(src[:1]),src1234],0)
                    else:
                        src = torch.cat([src[:1],src1234],0)
                else:
                    cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn2(src1,cross_src1)
                    cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn2(src2,cross_src2)
                    cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn2(src3,cross_src3)
                    cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn2(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    if self.pyramid and self.count==2:
                        src = torch.cat([self.proj(src[:1]),src1234],0)
                    else:
                        src = torch.cat([src[:1],src1234],0)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None,
                grid_pos = None, motion_feat=None,box_pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,num_frames,grid_pos,motion_feat,box_pos=box_pos)

class MaskedTransformerDeit128x384(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,use_learn_time_token=False,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,time_attn=False,
                 num_queries=None,use_channel_weight=False,uselearnpos=False,num_point=None,mlp_merge_query=False,
                 point128VS384=False,point128VS128x3=False,clstoken1VS384=False,time_attn_type=False,
                 use_decoder=False,use_128_decoder=True,tgt_after_mean=True,tgt_before_mean=True,multi_decoder=False,weighted_sum=False,
                 channelwise_decoder=False,add_cls_token=False,share_head=True,p4conv_merge=False,masking_radius=None):
        super().__init__()

        self.mlp_merge_query = mlp_merge_query
        self.num_queries = num_queries
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.use_learn_time_token = use_learn_time_token
        self.use_decoder = use_decoder
        self.use_128_decoder = use_128_decoder
        self.multi_decoder = multi_decoder
        self.weighted_sum = weighted_sum
        self.add_cls_token = add_cls_token
        self.share_head = share_head
        self.masking_radius = masking_radius
        self.nhead = nhead
        if self.mlp_merge_query:
            self.mlp_merge = MLP(input_dim = 4, hidden_dim = 0, output_dim = 1, num_layers = 1)
        encoder_layer = [TransformerEncoderLayerCrossAttn(d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before, num_point,use_channel_weight,
                        time_attn,time_attn_type,share_head = share_head) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.mlp_time_merge = p4conv_merge
        #self.encoder = nn.ModuleList(encoder_layer)

        self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reg_token = nn.Parameter(torch.zeros(num_queries, 1, d_model))
        self.time_token = nn.Parameter(torch.zeros(3, 1, d_model))
        if self.mlp_time_merge:
            self.mlp_merge = MLP(input_dim = 256*4, hidden_dim = 256, output_dim = 256, num_layers = 4)
        if uselearnpos:
            self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))
        if self.use_decoder.enabled:
            # if multi_decoder:
            #     decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
            #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            # elif channelwise_decoder.enabled:
            #     decoder_layer = TransformerDecoderLayerChannelwise(d_model, nhead, dim_feedforward,
            #                                 dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean,channelwise_decoder.weighted)
            if   self.use_decoder.name=='casc1':

                decoder_layer = TransformerDecoderLayerDeitCascaded1(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc2':

                decoder_layer = TransformerDecoderLayerDeitCascaded2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc3':

                decoder_layer = TransformerDecoderLayerDeitCascaded3(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            else:
                decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, src, src_mask=None, motion_feat=None):

        BS, N, C = src.shape
        self.num_point = N//4



        if self.mlp_time_merge:
            reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            src = src.view(src.shape[0],src.shape[1]//4 ,-1)
            src = self.mlp_merge(src)
            src = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

        else:

            if self.num_queries > 1:
                cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
                reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
                reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
                reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
                reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
            else:
                reg_token1 =  self.reg_token.repeat(BS,1,1)
                reg_token2 = src[:,self.num_point:self.num_point+1]
                reg_token3 = src[:,2*self.num_point:2*self.num_point+1]
                reg_token4 = src[:,3*self.num_point:3*self.num_point+1]

            #time_token1,time_token2,time_token3,time_token4 = self.time_token.repeat(BS,1,1).chunk(4,1)
            
            if self.add_cls_token:
                src1 = torch.cat([cls_token1,reg_token1,src[:,0:self.num_point-1]],dim=1)
                src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
            else:
                src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
                src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
                src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
                src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)


            src = torch.cat([src1,src2,src3,src4],dim=0)

        #  import pdb;pdb.set_trace()
        #  src = src.view(src.shape[0]*4,src.shape[1]//4,-1)
        src = src.permute(1, 0, 2)
        time_token = self.time_token.repeat(BS,1,1)
        time_token = time_token.permute(1, 0, 2)
        # pos = torch.cat([pos_embed1, pos_embed2, pos_embed3, pos_embed3],dim=0)
        # pos = pos.permute(1, 0, 2)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=None, pos=None,time_token=time_token,motion_feat=motion_feat) # num_point,bs,feat torch.Size([128, 128, 256])

        if self.use_decoder.enabled:
            if self.num_queries > 1:
                tgt_cls = memory[0:1,:memory.shape[1]//4,:].contiguous()
                tgt_reg1 = memory[1:2,:memory.shape[1]//4,:].contiguous()
                tgt_reg234 = memory[0:1,memory.shape[1]//4:,:].contiguous()
                tgt = torch.cat([tgt_reg1,tgt_reg234],1)

                memory1 = torch.cat([memory[2:3,:memory.shape[1]//4],memory[2:,:memory.shape[1]//4]],0).contiguous()
                memory234 = memory[1:,memory.shape[1]//4:].contiguous()
                memory = torch.cat([memory1,memory234],1).contiguous()
                # if self.use_128_decoder:
            #     memory = memory[1:,:memory.shape[1]//4,:] #current 128
            #     # else:
            #     #     memory = memory[1:].view(512,-1,memory.shape[-1]) #global 512
            else:
                tgt = memory[0:1,:,:]
                memory = memory[1:]
            hs = self.decoder(tgt, memory, memory_key_padding_mask=None,pos=None, query_pos=None).squeeze()
            return hs, tgt_cls
        else:

            if self.mlp_merge_query:
                memory = torch.cat(memory[0:1].chunk(4,dim=1),0).permute(2,1,0).contiguous()
                memory = self.mlp_merge(memory).permute(2,1,0).contiguous()
            else:
                if self.num_queries > 1:
                    if self.weighted_sum:
                        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                        weight = F.softmax(memory,0)
                        memory = (memory*weight).sum(0,True)
                    else:
                        if self.mlp_time_merge:
                            memory = memory[0:1]
                            return memory, None
                        else:
                            memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                            # memory = memory.mean(0,True)
                            return memory, None
                else:
                    memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                    memory= memory[0:1]
            memory = memory.permute(1, 0, 2)
            return memory, None

class TransformerDeitPerciverIO(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,use_learn_time_token=False,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,time_attn=False,
                 num_queries=None,use_channel_weight=False,uselearnpos=False,num_point=None,mlp_merge_query=False,
                 point128VS384=False,point128VS128x3=False,clstoken1VS384=False,time_point_attn_pre=False,
                 use_decoder=False,use_128_decoder=True,tgt_after_mean=True,tgt_before_mean=True,multi_decoder=False,weighted_sum=False,
                 channelwise_decoder=False,add_cls_token=False):
        super().__init__()

        self.mlp_merge_query = mlp_merge_query
        self.num_queries = num_queries
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.use_learn_time_token = use_learn_time_token
        self.use_decoder = use_decoder
        self.use_128_decoder = use_128_decoder
        self.multi_decoder = multi_decoder
        self.weighted_sum = weighted_sum
        self.add_cls_token = add_cls_token
        if self.mlp_merge_query:
            self.mlp_merge = MLP(input_dim = 4, hidden_dim = 0, output_dim = 1, num_layers = 1)
        encoder_layer = [TransformerEncoderLayerPerciverIO(d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before, num_point,use_channel_weight,
                        time_attn,point128VS384,point128VS128x3,clstoken1VS384,use_learn_time_token,time_point_attn_pre) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        #self.encoder = nn.ModuleList(encoder_layer)

        self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reg_token = nn.Parameter(torch.zeros(num_queries, 1, d_model))
        self.time_token = nn.Parameter(torch.zeros(3, 1, d_model))
        if uselearnpos:
            self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))
        if self.use_decoder.enabled:
            # if multi_decoder:
            #     decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
            #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            # elif channelwise_decoder.enabled:
            #     decoder_layer = TransformerDecoderLayerChannelwise(d_model, nhead, dim_feedforward,
            #                                 dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean,channelwise_decoder.weighted)
            if   self.use_decoder.name=='casc1':

                decoder_layer = TransformerDecoderLayerDeitCascaded1(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc2':

                decoder_layer = TransformerDecoderLayerDeitCascaded2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            elif self.use_decoder.name=='casc3':

                decoder_layer = TransformerDecoderLayerDeitCascaded3(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            else:
                decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, motion_feat=None):

        BS, N, C = src.shape
        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        if self.num_queries > 1:
            cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
            reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
            reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
            reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
        else:
            reg_token1 =  self.reg_token.repeat(BS,1,1)
            reg_token2 = src[:,self.num_point:self.num_point+1]
            reg_token3 = src[:,2*self.num_point:2*self.num_point+1]
            reg_token4 = src[:,3*self.num_point:3*self.num_point+1]

        #time_token1,time_token2,time_token3,time_token4 = self.time_token.repeat(BS,1,1).chunk(4,1)
        time_token = self.time_token.repeat(BS,1,1)
        if self.add_cls_token:
            src1 = torch.cat([cls_token1,reg_token1,src[:,0:self.num_point-1]],dim=1)
            src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
        else:
            src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)

        src = torch.cat([src1,src2,src3,src4],dim=0)
        src = src.permute(1, 0, 2)
        time_token = time_token.permute(1, 0, 2)
        # pos = torch.cat([pos_embed1, pos_embed2, pos_embed3, pos_embed3],dim=0)
        # pos = pos.permute(1, 0, 2)

        memory = self.encoder(src, src_key_padding_mask=None, pos=None,time_token=time_token,motion_feat=motion_feat) # num_point,bs,feat torch.Size([128, 128, 256])

        if self.use_decoder.enabled:
            if self.num_queries > 1:
                tgt_cls = memory[0:1,:memory.shape[1]//4,:].contiguous()
                tgt_reg1 = memory[1:2,:memory.shape[1]//4,:].contiguous()
                tgt_reg234 = memory[0:1,memory.shape[1]//4:,:].contiguous()
                tgt = torch.cat([tgt_reg1,tgt_reg234],1)

                memory1 = torch.cat([memory[2:3,:memory.shape[1]//4],memory[2:,:memory.shape[1]//4]],0).contiguous()
                memory234 = memory[1:,memory.shape[1]//4:].contiguous()
                memory = torch.cat([memory1,memory234],1).contiguous()
                # if self.use_128_decoder:
            #     memory = memory[1:,:memory.shape[1]//4,:] #current 128
            #     # else:
            #     #     memory = memory[1:].view(512,-1,memory.shape[-1]) #global 512
            else:
                tgt = memory[0:1,:,:]
                memory = memory[1:]
            hs = self.decoder(tgt, memory, memory_key_padding_mask=None,pos=None, query_pos=None).squeeze()
            return hs, tgt_cls
        else:

            if self.mlp_merge_query:
                memory = torch.cat(memory[0:1].chunk(4,dim=1),0).permute(2,1,0).contiguous()
                memory = self.mlp_merge(memory).permute(2,1,0).contiguous()
            else:
                if self.num_queries > 1:
                    if self.weighted_sum:
                        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                        weight = F.softmax(memory,0)
                        memory = (memory*weight).sum(0,True)
                    else:
                        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                        # memory = memory.mean(0,True)
                        return memory, None
                else:
                    memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
                    memory= memory[0:1]
            memory = memory.permute(1, 0, 2)
            return memory, None

class TransformerDeitTwins(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,
                 use_decoder=False,multi_decoder=False,tgt_after_mean=False,num_point=None,cross_attn=False,tgt_before_mean=False):
        super().__init__()

        self.cross_attn = cross_attn
        self.use_decoder = use_decoder

        # if cross_attn:
        encoder_layer = [TransformerEncoderLayerTwins(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before, num_point) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.num_point = num_point
        self.cls_token1 = nn.Parameter(torch.zeros(1, 8, d_model))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 8, d_model))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 8, d_model))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 8, d_model))

        if self.use_decoder.enabled:
            if multi_decoder:
                decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            else:
                decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, motion_feat=None):

        BS, N, C = src.shape

        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        # import pdb;pdb.set_trace()
        cls_token1 = self.cls_token1.repeat(BS,1,1)
        cls_token2 = self.cls_token2.repeat(BS,1,1)
        cls_token3 = self.cls_token3.repeat(BS,1,1)
        cls_token4 = self.cls_token4.repeat(BS,1,1)
        
        src1 = torch.cat([cls_token1,src[:,0:self.num_point]],dim=1)
        src2 = torch.cat([cls_token2,src[:,self.num_point:2*self.num_point]],dim=1)
        src3 = torch.cat([cls_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
        src4 = torch.cat([cls_token4,src[:,3*self.num_point:]],dim=1)

        src = torch.cat([src1,src2,src3,src4],dim=0)
        src = src.permute(1, 0, 2)

        memory = self.encoder(src, src_key_padding_mask=None, pos=None) # num_point,bs,feat torch.Size([128, 128, 256])
        #import pdb;pdb.set_trace()

        if self.use_decoder.enabled:
            q = memory[0:8,:memory.shape[1]//4,:].mean(0,True).contiguous()#.view(4,memory.shape[1]//4,-1)
            kv = memory[0:8,memory.shape[1]//4:,:].contiguous().view(8*3,memory.shape[1]//4,-1)
            hs = self.decoder(q, kv, memory_key_padding_mask=None,pos=None, query_pos=None).squeeze()
            return hs, None
        else:
            memory = torch.cat(memory[0:8].mean(0,keepdim=True).chunk(4,dim=1),0)
            # memory = memory.mean(0,keepdim=True)
            # memory = memory.permute(1, 0, 2)
            return memory, None

class TransformerDeitSwapToken(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,
                 use_decoder=False,multi_decoder=False,tgt_after_mean=False,num_point=None,cross_attn=False,tgt_before_mean=False):
        super().__init__()

        self.cross_attn = cross_attn
        self.use_decoder = use_decoder

        # if cross_attn:
        encoder_layer = [TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.num_point = num_point
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, d_model))

        if self.use_decoder.enabled:
            if multi_decoder:
                decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            else:
                decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, motion_feat=None):

        BS, N, C = src.shape

        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        # import pdb;pdb.set_trace()
        cls_token1 = self.cls_token1.repeat(BS,1,1)
        cls_token2 = self.cls_token2.repeat(BS,1,1)
        cls_token3 = self.cls_token3.repeat(BS,1,1)
        cls_token4 = self.cls_token4.repeat(BS,1,1)
        
        src1 = torch.cat([cls_token1,cls_token2,cls_token3,cls_token4,src[:,0:self.num_point]],dim=1)
        src2 = torch.cat([cls_token1,cls_token2,cls_token3,cls_token4,src[:,self.num_point:2*self.num_point]],dim=1)
        src3 = torch.cat([cls_token1,cls_token2,cls_token3,cls_token4,src[:,2*self.num_point:3*self.num_point]],dim=1)
        src4 = torch.cat([cls_token1,cls_token2,cls_token3,cls_token4,src[:,3*self.num_point:]],dim=1)

        src = torch.cat([src1,src2,src3,src4],dim=0)
        src = src.permute(1, 0, 2)

        memory = self.encoder(src, src_key_padding_mask=None, pos=None) # num_point,bs,feat torch.Size([128, 128, 256])
        #import pdb;pdb.set_trace()

        if self.use_decoder.enabled:
            q = memory[0:8,:memory.shape[1]//4,:].mean(0,True).contiguous()#.view(4,memory.shape[1]//4,-1)
            kv = memory[0:8,memory.shape[1]//4:,:].contiguous().view(8*3,memory.shape[1]//4,-1)
            hs = self.decoder(q, kv, memory_key_padding_mask=None,pos=None, query_pos=None).squeeze()
            return hs, None
        else:
            memory = memory[0:4,:memory.shape[1]//4]
            # memory = memory.mean(0,keepdim=True)
            # memory = memory.permute(1, 0, 2)
            return memory, None

class TransformerDeitTime5D(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_intermediate_dec=False,
                 channeltime=False,globalattn=False,usegloballayer=True,num_point=None,cross_attn=False):
        super().__init__()

        self.cross_attn = cross_attn


        encoder_layer = TransformerEncoderLayerCrossAttn5D(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before, num_point)


        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder5D(encoder_layer, num_encoder_layers, encoder_norm)

        self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(4, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))
        self.time_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.time_pos_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        BS, N, C = src.shape

        # if self.channeltime:
        #     C = C//4
        # src = src.view(src.shape[0]*4, -1, src.shape[-1])
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        #import pdb;pdb.set_trace()
        time_cls_token = self.time_cls_token.repeat(BS,1,1).permute(1, 0, 2)
        time_pos_token = self.time_pos_token.repeat(BS,1,1).permute(1, 0, 2)
        cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
        pos_embed1 = self.pos_embed[0:1].repeat(BS,1,1)
        cls_token2 = self.cls_token[1:2].repeat(BS,1,1)
        pos_embed2 = self.pos_embed[1:2].repeat(BS,1,1)
        cls_token3 = self.cls_token[2:3].repeat(BS,1,1)
        pos_embed3 = self.pos_embed[2:3].repeat(BS,1,1)
        cls_token4 = self.cls_token[3:4].repeat(BS,1,1)
        pos_embed4 = self.pos_embed[3:4].repeat(BS,1,1)
        src1 = torch.cat([cls_token1,src[:,0:self.num_point]],dim=1)
        src1 = src1 + pos_embed1
        src2 = torch.cat([cls_token2,src[:,self.num_point:2*self.num_point]],dim=1)
        src2 = src2 + pos_embed2
        src3 = torch.cat([cls_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
        src3 = src3 + pos_embed3
        src4 = torch.cat([cls_token4,src[:,3*self.num_point:]],dim=1)
        src4 = src4 + pos_embed4
        pos = torch.cat([pos_embed1, pos_embed2, pos_embed3, pos_embed3],dim=0)
        src = torch.cat([src1,src2,src3,src3],dim=0)
        src = src.permute(1, 0, 2)
        pos = pos.permute(1, 0, 2)

 
        memory = self.encoder(src, time_cls_token=time_cls_token, time_pos_token=time_pos_token,pos=pos) # num_point,bs,feat torch.Size([128, 128, 256])

        memory = memory.permute(1, 0, 2)
        return memory, None

class PerciverIOTime(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False,return_intermediate_dec=False,
                channeltime=False,globalattn=False,usegloballayer=True,num_point=None,cross_attn=False,self_attn=True):
        super().__init__()

        self.cross_attn = cross_attn
        self.self_attn = self_attn

        # if self.self_attn:
        #     self.self_attn_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_point)
        # else:
        #     encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                     dropout, activation, normalize_before)
        #     self.reduce_layer = TransformerEncoderLayerDeitTime(d_model, nhead, dim_feedforward,
        #                                 dropout, activation, normalize_before)

        self.cross_attn_layer1 = CrossAttention(heads=4)
        self.cross_attn_layer2 = CrossAttention(heads=4)
        self.cross_attn_layer3 = CrossAttention(heads=4)
    
        self.self_attn_layer = Attention(heads=4)
        # self.encoder1 = TransformerEncoder(encoder_layer, num_encoder_layers//2, None)
        # self.encoder2 = TransformerEncoder(encoder_layer, num_encoder_layers//2, None)

        self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(1, 4, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, 1, d_model))
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        

        # if self.channeltime:
        #     C = C//4
        src = src.view(src.shape[0], 4, -1, src.shape[-1])
        BS, T, N, C = src.shape
        #if self.self_attn:
        # src = src.permute(1, 0, 2)
        # pos_embed = pos_embed.permute(1, 0, 2)
        #import pdb;pdb.set_trace()
        # cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
        # pos_embed1 = self.pos_embed[0:1].repeat(BS,1,1)
        # cls_token2 = self.cls_token[1:2].repeat(BS,1,1)
        # pos_embed2 = self.pos_embed[1:2].repeat(BS,1,1)
        # cls_token3 = self.cls_token[2:3].repeat(BS,1,1)
        # pos_embed3 = self.pos_embed[2:3].repeat(BS,1,1)
        # cls_token4 = self.cls_token[3:4].repeat(BS,1,1)
        # pos_embed4 = self.pos_embed[3:4].repeat(BS,1,1)
        # src1 = torch.cat([cls_token1,src[:,0:self.num_point]],dim=1)
        # src1 = src1 + pos_embed1
        # src2 = torch.cat([cls_token2,src[:,self.num_point:2*self.num_point]],dim=1)
        # src2 = src2 + pos_embed2
        # src3 = torch.cat([cls_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
        # src3 = src3 + pos_embed3
        # src4 = torch.cat([cls_token4,src[:,3*self.num_point:]],dim=1)
        # src4 = src4 + pos_embed4
        # pos = torch.cat([pos_embed1, pos_embed2, pos_embed3, pos_embed3],dim=0)
        # src = torch.cat([src1,src2,src3,src3],dim=0)
        # src = src.permute(1, 0, 2)
        # pos = pos.permute(1, 0, 2)
        cls_token = self.cls_token.repeat(BS, 1, 1, 1)
        #pos_embed = self.pos_embed.repeat(BS, 1, self.num_point, 1)

        #src = src + pos_embed 

        cls_token = self.cross_attn_layer1(cls_token,src)
        memory = self.encoder1(cls_token.squeeze(2), src_key_padding_mask=None,) # num_point,bs,feat torch.Size([128, 128, 256])
        cls_token = self.cross_attn_layer2(cls_token,src)
        memory = self.encoder2(cls_token.squeeze(2), src_key_padding_mask=None,) # num_point,bs,feat torch.Size([128, 128, 256])

        memory = memory.mean(1,keepdim=True)

        return memory, None

class PerciverIO(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False,return_intermediate_dec=False,
                use_channel_weight=False,time_attn=False,num_queries = None,num_point=None,split_time=False,self_attn=True):
        super().__init__()

        self.split_time= split_time
        self.self_attn = self_attn

        # if self.self_attn:
        encoder_layer = [TransformerEncoderLayerPerciverIO(d_model, nhead, dim_feedforward, dropout, activation, \
                            normalize_before, num_point,use_channel_weight,time_attn) for i in range(num_encoder_layers)]
        # else:
        #     encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                     dropout, activation, normalize_before)
        #     self.reduce_layer = TransformerEncoderLayerDeitTime(d_model, nhead, dim_feedforward,
        #                                 dropout, activation, normalize_before)

        # self.cross_attn_layer1 = CrossAttention(heads=4)
        # self.cross_attn_layer2 = CrossAttention(heads=4)
        # self.cross_attn_layer3 = CrossAttention(heads=4)
    
        # self.self_attn_layer = Attention(heads=4)
        self.encoder = TransformerEncoderPerciver(encoder_layer, num_encoder_layers, None)
        # self.encoder2 = TransformerEncoder(encoder_layer, num_encoder_layers//2, None)

        self.num_point = num_point
        self.cls_token = nn.Parameter(torch.zeros(1, num_queries, d_model))
        #self.pos_embed = nn.Parameter(torch.zeros(1, 4, 1, d_model))
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        

        # if self.channeltime:
        #     C = C//4
        BS, N, C = src.shape
        if self.split_time:
            src = src.view(BS*4,N//4,C)

        #if self.self_attn:
        src = src.permute(1, 0, 2)

        cls_token = self.cls_token.repeat(BS, 1, 1).permute(1, 0, 2)

        #import pdb;pdb.set_trace()
        memory,cls_token = self.encoder(src, cls_token=cls_token) # num_point,bs,feat torch.Size([128, 128, 256])
        # cls_token = self.cross_attn_layer2(cls_token,src)
        # memory = self.encoder2(cls_token.squeeze(2), src_key_padding_mask=None,) # num_point,bs,feat torch.Size([128, 128, 256])
        if cls_token.shape[0] > 1:
            cls_token = cls_token.mean(0,keepdim=True)

        return cls_token, None

class TransformerEnc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,channeltime=False,boxencoding=False):
        super().__init__()

        self.channeltime = channeltime
        if channeltime:
            encoder_layer1 = TransformerEncoderLayerTimeQKV(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_layer2 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
            encoder_layer3 = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            encoder_layer = [encoder_layer1,encoder_layer2,encoder_layer3]
            self.encoder = TransformerEncoderTimeQKV(encoder_layer, num_encoder_layers, encoder_norm)

        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)

            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        #import pdb;pdb.set_trace()
        bs, n, c = src.shape
        if self.channeltime:
            c = c//4
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed) # num_point,bs,feat torch.Size([128, 128, 256])
        memory = torch.max(memory, 0, keepdim=True)[0]

        return memory.permute(1, 0, 2).view(bs, c)

class Transformer_crossbox(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,spacetime=False,boxencoding=False):
        super().__init__()


        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                        dropout, activation, normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        # self.encoder_cross_box = TransformerEncoderLayerBoxEncoding(d_model, nhead, dim_feedforward,
        #                                 dropout, activation, normalize_before)
        #self.boxmlp = MLP_v2([8,64,128,256])
        self.down_mlp = MLP(input_dim = 512, hidden_dim = 0, output_dim = 256, num_layers = 1)
        #     nn.Conv1d(8, 64, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(64),
        #                                 nn.ReLU(),
        # )
        self.boxencoding = nn.Sequential(nn.Conv1d(8, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(512) 
                                        )
        self.box_reduce = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos,box_seq=None):
        #import pdb;pdb.set_trace()
        bs, num, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos.permute(1, 0, 2)#[:,:,:256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs//4, 1) #torch.Size([1, 128, 256]) num_query, bs, hideen_dim
        tgt = torch.zeros_like(query_embed) #torch.Size([1, 128, 256])
        memory = self.encoder(src, src_key_padding_mask=None, pos=pos_embed) # num_point,bs,feat torch.Size([128, 128, 256])
        seq_encoding = torch.max(self.boxencoding(box_seq.permute(0,2,1,3).contiguous().view(bs, 8, 4)), -1, keepdim=False)[0]
        seq_encoding = self.box_reduce(seq_encoding.unsqueeze(-1)).squeeze(-1)[None,:,:].repeat(memory.shape[0],1,1)
        # boxencoding = F.softplus(self.boxmlp(box_seq[:,:,None])).permute(2,0,1)
        memory = self.down_mlp(torch.cat([memory,seq_encoding],-1))
        # memory =  self.encoder_cross_box(memory,pos=pos_embed)
        import pdb;pdb.set_trace()
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos.permute(1, 0, 2), query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, num), seq_encoding

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        #self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None, box_pos=None,
                pos_pyramid=None,motion_feat=None,center_token=None,empty_ball_mask=None):
        token_list = []
        output = src
        for layer in self.layers:
            output,tokens = layer(output, src_mask=mask,src_key_padding_mask=src_key_padding_mask, box_pos=box_pos,
                        pos=pos, num_frames=num_frames,)
            if self.config.use_center_token_add:
                token_list.append(tokens+center_token.repeat(4,1,1))
            else:
                token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,torch.cat(token_list,0)

class TransformerEncoderCT3D(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        #self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None, box_pos=None,
                pos_pyramid=None,motion_feat=None,center_token=None,empty_ball_mask=None):
        token_list = []
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,src_key_padding_mask=src_key_padding_mask, box_pos=box_pos,
                        pos=pos, num_frames=num_frames,)
        if self.norm is not None:
            output = self.norm(output)

        return output, None

class TransformerEncoderPerciver(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                cls_token,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        output = src
        for layer in self.layers:
            output,cls_token = layer(output, cls_token=cls_token,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output,cls_token

class TransformerEncoder5D(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                time_cls_token,
                time_pos_token,
                pos: Optional[Tensor] = None):

        output = src
        for layer in self.layers:
            output,time_cls_token = layer(output, time_cls_token=time_cls_token,
                           time_pos_token=time_pos_token, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return time_cls_token

class TransformerEncoderTimeQKV(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
   
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


        # row_num = np.ceil(np.sqrt(feature_map_num))
        # plt.figure()
        # for index in range(1, feature_map_num+1):
        #     plt.subplot(row_num, row_num, index)
        #     plt.imshow(feature_map[index-1], cmap='gray')
        #     plt.axis('off')
        #     cv2.imwrite(str(index)+".png", feature_map[index-1])
        # plt.show()

    # def forward_post(self,
    #                  src,
    #                  src_mask: Optional[Tensor] = None,
    #                  src_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None,motion_feat=None):
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     grid_pos = None,motion_feat= None,box_pos=None):

        q = k = self.with_pos_embed(src, pos)

        src2,attn = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src 

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None,
                grid_pos = None, motion_feat=None,box_pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerCrossAttn(nn.Module):
    count = 0
    def __init__(self, config, d_model, dout, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,
                 use_channel_weight=False,time_attn=False,time_attn_type=False, use_motion_attn=False,share_head=True,
                 ms_pool=False,pyramid=False,use_grid_pos=False,use_box_pos=False,mlp_cross_grid_pos=False,merge_groups=None,
                 update_234=False,crossattn_last_layer=False,share_sa_layer=True,add_extra_sa=False,fc_token=None):
        super().__init__()
        TransformerEncoderLayerCrossAttn.count += 1
        self.count = TransformerEncoderLayerCrossAttn.count
        # self.pool = False
        # if self.count == 2:
        #     self.pool = True
        # if self.count >= 2 and pyramid:
        #     d_model = 2*d_model
        self.config = config
        self.ms_pool = ms_pool
        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.time_attn_type = time_attn_type
        self.use_motion_attn = use_motion_attn
        self.time_attn = time_attn
        self.pyramid = pyramid
        self.use_grid_pos = use_grid_pos
        self.mlp_cross_grid_pos = mlp_cross_grid_pos
        self.merge_groups = merge_groups
        self.update_234 = update_234
        self.crossattn_last_layer = crossattn_last_layer
        self.share_sa_layer = share_sa_layer
        self.add_extra_sa = add_extra_sa
        self.use_mlp_query_decoder  = self.config.use_mlp_query_decoder
        self.fc_token = fc_token
        if self.add_extra_sa:
            self.self_attn_extra = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn_extra = FFN(d_model, dim_feedforward)

        if self.share_sa_layer.enabled:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            if self.share_sa_layer.share_ffn:
                self.sa_ffn = FFN(d_model, dim_feedforward)
            else:
                self.sa_ffn1 = FFN(d_model, dim_feedforward)
                self.sa_ffn2 = FFN(d_model, dim_feedforward)
                self.sa_ffn3 = FFN(d_model, dim_feedforward)
                self.sa_ffn4 = FFN(d_model, dim_feedforward)

        self.share_head = share_head
        self.use_box_pos = use_box_pos
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if d_model != dout:
            self.proj = nn.Linear(d_model, dout)

        # if self.use_channel_weight=='ct3d':
        #     self.channel_attn = MultiHeadedAttention(d_model,nhead)
        if self.use_mlp_query_decoder and self.count==3:
            self.decoder_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.decoder_ffn = FFN(d_model, dim_feedforward)

        if self.time_attn:
            self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            if self.share_head:
                self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            else:
                self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            
            if use_motion_attn:
                self.time_point_attn = MultiHeadedAttentionMotion(nhead, d_model)
            else:
                self.time_point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)



            self.time_token_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_cls_token = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        if self.share_head:
            self.ffn1 = FFN(d_model, dim_feedforward)
        else:
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)
            self.ffn3 = FFN(d_model, dim_feedforward)
            self.ffn4 = FFN(d_model, dim_feedforward)

        if self.time_attn_type in ['time_mlp', 'crossattn_mlp', 'time_mlp_v2', 'mlp_mixer_v2']:
            self.time_mlp1 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp3 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp4 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            
            # if self.count>=2 and self.pyramid:
            #     self.time_mlp_fusion = MLP(input_dim = 512*4, hidden_dim = 512, output_dim = 512, num_layers = 4)
            # else:
            # import pdb;pdb.set_trace()
            if self.config.get('use_semlp',None):
                self.time_mlp_fusion = SEMLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model)
            else:
                self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)

            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

        if self.config.get('only_use_ca_for_ab', None):
            self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)

            self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            # if self.pyramid and self.count==2:
            #     #     self.ffn_up = FFNUp(d_model, dim_feedforward, dout=2*d_model)
            #     self.proj = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256*2, num_layers = 3)
            # else:
            #     self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'mlp_merge' and self.count==3:
            self.time_mlp1 = MLP(input_dim = 256*(merge_groups-1), hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'trans_merge' and self.count==3:
            self.time_mlp1 = MLP(input_dim = 256*merge_groups, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'trans_merge_cas':
            # self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_mlp_fusion1 = MLP(input_dim = 256*4, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion3 = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.time_mlp_fusion4 = MLP(input_dim = 256*1, hidden_dim = 256, output_dim = 256, num_layers = 4)
            self.ffn1 = FFN(d_model, dim_feedforward)
            # self.ffn2 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'crossattn_trans':
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn1 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'mlp_mixer_v2':
            self.mlp_merge = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 3)

        if self.crossattn_last_layer:
            self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn_last = FFN(d_model, dim_feedforward)

        if self.pyramid and self.count==2:
            self.proj = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 3)
            # self.proj_token = MLP(input_dim = 256, hidden_dim = 256*2, output_dim = 256*2, num_layers = 3)

        # self.pool_q = nn.Conv3d(
        #             head_dim,
        #             head_dim,
        #             (1,1,1),
        #             stride=(1,1,1),
        #             padding=(0,0,0),
        #             groups=head_dim,
        #             bias=False,
        #         )
        # stride = (2,2,2)
        # kernel_skip = [s + 1 if s > 1 else s for s in stride]
        # stride_skip = stride_q
        # padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.max2_pool = nn.MaxPool3d((2,2,2), (2,2,2), (0,0,0), ceil_mode=False)
        self.max4_pool = nn.MaxPool3d((4,4,4), (4,4,4), (0,0,0), ceil_mode=False)

        if self.config.use_mlp_mixer.enabled:
            if self.config.use_mlp_mixer.use_v2:
                self.mixer = SpatialMixerBlockV2(self.config.use_mlp_mixer.hidden_dim)
            else:
                # v1 use 16
                self.mixer = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,self.config.use_mlp_mixer.get('grid_size', 4),self.config.hidden_dim, self.config.use_mlp_mixer)
            self.cross_mixer = TimeMixerBlock()




    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     pos_pyramid = None,motion_feat= None,box_pos=None,empty_ball_mask=None):

        src_ori = src
        src_rm_token = src[1:]


        if self.count==2 and self.pyramid:
            # src_points = self.proj(src[1:])
            # src_token = self.proj_token(src[:1])
            # src_3d = src_points.permute(1,2,0,).contiguous().view(src_points.shape[1],src_points.shape[2],4,4,4)
            # src_3d = self.max2_pool(src_3d).view(src_3d.shape[0],src_3d.shape[1],-1).permute(2,0,1)
            # src = torch.cat([src_token,src_3d],0)
            src_reducez = src_rm_token.view(32,2,src.shape[1],src.shape[-1]).permute(0,2,1,3).contiguous().view(32,src.shape[1],2*src.shape[-1])
            src_reducez = self.proj(src_reducez)
            src = torch.cat([src[0:1],src_reducez],0)
            src_rm_token = src[1:]


        if self.pyramid and self.count >= 2:
            pos_index =[0] + [i for i in range(1,65,2)]
            q = k = self.with_pos_embed(src, pos[pos_index])
        else:
            if self.config.use_fc_token.enabled:
                q = k = self.with_pos_embed(src, pos[1:])
            else:
                q = k = self.with_pos_embed(src, pos)

        if self.add_extra_sa and self.count == 1:
            src_extra = self.self_attn_extra(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
            src = self.ffn_extra(src,src_extra)
            q = k = self.with_pos_embed(src, pos)

        if self.share_sa_layer.enabled:
            if self.config.use_mlp_mixer.enabled:

                if not self.config.get('only_use_ca_for_ab', None):

                    token = src[:1]
                    src_mixer = self.mixer(src[1:])

                    if not pos is None:
                        k = self.with_pos_embed(src_mixer, pos[1:])
                    else:
                        k = src_mixer

                    src2,attn_weight = self.self_attn(token, k, value=src_mixer, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
                    token = token + self.dropout1(src2)
                    token = self.norm1(token)
                    src2 = self.linear2(self.dropout(self.activation(self.linear1(token))))
                    token = token + self.dropout2(src2)
                    token = self.norm2(token)
                    if self.config.use_mlp_mixer.use_attn_reweight:
                        src_mixer = src_mixer * attn_weight.permute(2,0,1).clone().detach()
                    src = torch.cat([token,src_mixer],0)
                    cls_token = None
                else:

                    src_512 = src[1:].contiguous().view((src.shape[0]-1)*4,-1,src.shape[-1])
                    src1,src2,src3,src4 = src_512.chunk(4,0)
                    src_fusion = torch.cat([src1,src2,src3,src4],-1)
                    src_fusion = self.time_mlp_fusion(src_fusion)
                    k = self.with_pos_embed(src_fusion, pos[1:])
                    q1 = self.with_pos_embed(src1, pos[1:])
                    q2 = self.with_pos_embed(src2, pos[1:])
                    q3 = self.with_pos_embed(src3, pos[1:])
                    q4 = self.with_pos_embed(src4, pos[1:])
                    cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn2(src1,cross_src1)
                    cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn2(src2,cross_src2)
                    cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn2(src3,cross_src3)
                    cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn2(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)

            else:
                src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)[0]
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                cls_token = None
                if self.config.use_fc_token.enabled:
                    if self.config.use_fc_token.share:
                        tokens = self.fc_token(src.permute(1,0,2).contiguous().view(src.shape[1],-1)).view(4,-1,src.shape[-1])
                    else:
                        tokens = self.fc_token[self.count-1](src.permute(1,0,2).contiguous().view(src.shape[1],-1)).view(4,-1,src.shape[-1])


        else:
            q1,q2,q3,q4 = q.chunk(4,1)
            k1,k2,k3,k4 = k.chunk(4,1)
            src1,src2,src3,src4 = src.chunk(4,1)
            src_sa1 = self.self_attn1(q1, k1, value=src1, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
            src_sa2 = self.self_attn2(q2, k2, value=src2, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
            src_sa3 = self.self_attn3(q3, k3, value=src3, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
            src_sa4 = self.self_attn4(q4, k4, value=src4, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

            if self.share_sa_layer.share_ffn:
                src1 = self.sa_ffn(src1,src_sa1)
                src2 = self.sa_ffn(src2,src_sa2)
                src3 = self.sa_ffn(src3,src_sa3)
                src4 = self.sa_ffn(src4,src_sa4)
            else:
                src1 = self.sa_ffn1(src1,src_sa1)
                src2 = self.sa_ffn2(src2,src_sa2)
                src3 = self.sa_ffn3(src3,src_sa3)
                src4 = self.sa_ffn4(src4,src_sa4)
            
            src = torch.cat([src1,src2,src3,src4],1)



        if self.time_attn:
            if self.time_attn_type == '128x384':
                src_512 = torch.cat([src[2:3],src[2:]],0).view(512,-1,src.shape[-1])
                src1 = src_512[0:128]
                if not motion_feat is None:
                    src2 = src_512[128:2*128]  + motion_feat[:,1:2]
                    src3 = src_512[2*128:3*128]+ motion_feat[:,2:3]
                    src4 = src_512[3*128:4*128]+ motion_feat[:,3:4]
                    src234 = torch.cat([src2,src3,src4],0)
                    #import pdb;pdb.set_trace()
                    src234 = src_512[128:] #[N,BS,C]
                    src2 = self.time_point_attn(src1.permute(1,2,0), src234.permute(1,2,0), value=src234.permute(1,2,0), motion_feat = motion_feat)[0]
                else:
                    src234 = src_512[128:]
                    src2 = self.time_point_attn(src1, src234, value=src234, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                src1 = src1 + self.dropout3(src2)
                src1 = self.norm3(src1)
                src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
                src1 = src1 + self.dropout5(src2)
                src_point = torch.cat([self.norm4(src1),src_512[128:]],0).view(128,-1,src.shape[-1])
                src = torch.cat([src[0:1],src_point],0)

            elif self.time_attn_type == '128x384x3':
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                # src_512 = src.view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                if src1.shape[0]==64:
                    src1 = self.with_pos_embed(src1, pos[1:])
                    src2 = self.with_pos_embed(src2, pos[1:])
                    src3 = self.with_pos_embed(src3, pos[1:])
                    src4 = self.with_pos_embed(src4, pos[1:])

                src_pre1 = torch.cat([src2,src3,src4],0)
                src_pre2 = torch.cat([src1,src3,src4],0)
                src_pre3 = torch.cat([src1,src2,src4],0)
                src_pre4 = torch.cat([src1,src2,src3],0)

                if self.share_head:
                    cross_src1 = self.time_attn1(src1, src_pre1, value=src_pre1, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src2 = self.time_attn2(src2, src_pre2, value=src_pre2, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn1(src2,cross_src2)
                    cross_src3 = self.time_attn3(src3, src_pre3, value=src_pre3, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn1(src3,cross_src3)
                    cross_src4 = self.time_attn4(src4, src_pre4, value=src_pre4, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn1(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([src[:1],src1234],0)
                    # src = torch.cat([src[:1],src1234],0)
                    # src = src1234 #torch.cat([src[:1],src1234],0)
                    # src = torch.cat([src[:1],src1234],0)

                else:
                    
                    cross_src1 = self.time_attn1(src1, src_pre1, value=src_pre1, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src2 = self.time_attn2(src2, src_pre2, value=src_pre2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn2(src2,cross_src2)
                    cross_src3 = self.time_attn3(src3, src_pre3, value=src_pre3, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn3(src3,cross_src3)
                    cross_src4 = self.time_attn4(src4, src_pre4, value=src_pre4, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn4(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([src[:1],src1234],0)

            elif self.time_attn_type == 'time_mlp':
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_pre1 = torch.cat([src2,src3,src4],-1)
                src_pre2 = torch.cat([src1,src3,src4],-1)
                src_pre3 = torch.cat([src1,src2,src4],-1)
                src_pre4 = torch.cat([src1,src2,src3],-1)
                src1 = self.ffn1(src1,self.time_mlp1(src_pre1))
                src2 = self.ffn1(src2,self.time_mlp2(src_pre2))
                src3 = self.ffn1(src3,self.time_mlp3(src_pre3))
                src4 = self.ffn1(src4,self.time_mlp4(src_pre4))

                src1234 = torch.cat([src1,src2,src3,src4],1)
                src = torch.cat([src[:1],src1234],0)
                
            elif self.time_attn_type == 'time_mlp_v2':
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_fusion = torch.cat([src1,src2,src3,src4],-1)
                src_fusion = self.time_mlp_fusion(src_fusion)
                src1 = self.ffn1(src1,src_fusion)
                src2 = self.ffn1(src2,src_fusion)
                src3 = self.ffn1(src3,src_fusion)
                src4 = self.ffn1(src4,src_fusion)

                src1234 = torch.cat([src1,src2,src3,src4],1)
                src = torch.cat([src[:1],src1234],0)

            elif self.time_attn_type == 'mlp_merge':
                
                if self.count ==3:
                    token = src[:1,:src.shape[1]//self.merge_groups]
                    src_512 = src[1:].view(src_rm_token.shape[0]*self.merge_groups,-1,src.shape[-1])
                    src_merge = torch.cat(src_512.chunk(self.merge_groups,0)[1:],-1)
                    src_cur = src_512.chunk(self.merge_groups,0)[0]

                    src1 = self.ffn1(src_cur,self.time_mlp1(src_merge))
                    src1 = torch.cat([token,src1],0)
                    q = k = self.with_pos_embed(src1, pos)
                    src2 = self.self_attn2(q, k, value=src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                    src = self.ffn2(src1,src2)
                    
            elif self.time_attn_type == 'trans_merge_cas': 

                if self.count==1:

                    src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                    src1,src2,src3,src4 = src_512.chunk(4,0)
                    src_fusion = torch.cat([src1,src2,src3,src4],-1)
                    src_fusion = self.time_mlp_fusion1(src_fusion)
                    ##
                    if self.mlp_cross_grid_pos:
                        k = self.with_pos_embed(src_fusion, pos[1:])
                        q1 = self.with_pos_embed(src1, pos[1:])
                        q2 = self.with_pos_embed(src2, pos[1:])
                        q3 = self.with_pos_embed(src3, pos[1:])
                        q4 = self.with_pos_embed(src4, pos[1:])
                        # src_fusion = src_fusion + pos[1:,:src_fusion.shape[1]]
                            ##
                        if self.ms_pool:
                            src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                            src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                            

                        cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src1 = self.ffn1(src1,cross_src1)
                        cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src2 = self.ffn1(src2,cross_src2)
                        cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src3 = self.ffn1(src3,cross_src3)
                        cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src4 = self.ffn1(src4,cross_src4)
                        src1234 = torch.cat([src1,src2,src3,src4],1)
                        if self.pyramid and self.count==2:
                            src = torch.cat([self.proj(src[:1]),src1234],0)
                        else:
                            src = torch.cat([src[:1],src1234],0)

                elif self.count ==2:

                    src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                    src1,src2,src3,src4 = src_512.chunk(4,0)
                    src_fusion = torch.cat([src1,src2,src3],-1)
                    src_fusion = self.time_mlp_fusion2(src_fusion)
                    ##
                    if self.mlp_cross_grid_pos:
                        k = self.with_pos_embed(src_fusion, pos[1:])
                        q1 = self.with_pos_embed(src1, pos[1:])
                        q2 = self.with_pos_embed(src2, pos[1:])
                        q3 = self.with_pos_embed(src3, pos[1:])
                        # q4 = self.with_pos_embed(src4, pos[1:])
                        # src_fusion = src_fusion + pos[1:,:src_fusion.shape[1]]
                            ##
                        if self.ms_pool:
                            src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                            src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                            

                        cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src1 = self.ffn1(src1,cross_src1)
                        cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src2 = self.ffn1(src2,cross_src2)
                        cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src3 = self.ffn1(src3,cross_src3)
                        # cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        # src4 = self.ffn2(src4,cross_src4)
                        src1234 = torch.cat([src1,src2,src3,src4],1)
                        if self.pyramid and self.count==2:
                            src = torch.cat([self.proj(src[:1]),src1234],0)
                        else:
                            src = torch.cat([src[:1],src1234],0)

                elif self.count ==3:

                    src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                    src1,src2,src3,src4 = src_512.chunk(4,0)
                    src_fusion = torch.cat([src1,src2],-1)
                    src_fusion = self.time_mlp_fusion3(src_fusion)
                    ##
                    if self.mlp_cross_grid_pos:
                        k = self.with_pos_embed(src_fusion, pos[1:])
                        q1 = self.with_pos_embed(src1, pos[1:])
                        q2 = self.with_pos_embed(src2, pos[1:])
                        # q3 = self.with_pos_embed(src3, pos[1:])
                        # q4 = self.with_pos_embed(src4, pos[1:])
                        # src_fusion = src_fusion + pos[1:,:src_fusion.shape[1]]
                            ##
                        if self.ms_pool:
                            src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                            src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                            src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                            

                        cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src1 = self.ffn1(src1,cross_src1)
                        cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        src2 = self.ffn1(src2,cross_src2)
                        # cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        # src3 = self.ffn2(src3,cross_src3)
                        # cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                        # src4 = self.ffn2(src4,cross_src4)
                        src1234 = torch.cat([src1,src2,src3,src4],1)
                        if self.pyramid and self.count==2:
                            src = torch.cat([self.proj(src[:1]),src1234],0)
                        else:
                            src = torch.cat([src[:1],src1234],0)

                elif self.count == 4:
                    pass

                    # token = src[:1,:src.shape[1]//self.merge_groups]
                    # src_512 = src[1:].view(src_rm_token.shape[0]*self.merge_groups,-1,src.shape[-1])
                    # src_list = src_512.chunk(self.merge_groups,0)
                    # src_merge = torch.cat(src_list,-1)
                    # # src_before = torch.cat(src_list[1:],-1)
                    # src_cur = src_512.chunk(self.merge_groups,0)[0]
                    # src_merge = self.time_mlp1(src_merge)
                    # src2 = self.cross_attn2(src_cur, src_merge, value=src_merge, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                    # src1 = self.ffn1(src_cur,src2)
                    # src1 = torch.cat([token,src1],0)
                    # q = k = self.with_pos_embed(src1, pos)
                    # src2 = self.self_attn2(q, k, value=src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                    # src_cur = self.ffn2(src1,src2)
                    # src_before = src[:,src.shape[1]//self.merge_groups:,]
                    # src = torch.cat([src_cur,src_before],1)

            elif self.time_attn_type == 'trans_merge':

                if self.count==3:
                    
                    token = src[:1,:src.shape[1]//self.merge_groups]
                    src_512 = src[1:].view(src_rm_token.shape[0]*self.merge_groups,-1,src.shape[-1])
                    src_list = src_512.chunk(self.merge_groups,0)
                    src_merge = torch.cat(src_list,-1)
                    # src_before = torch.cat(src_list[1:],-1)
                    src_cur = src_512.chunk(self.merge_groups,0)[0]
                    src_merge = self.time_mlp1(src_merge)
                    src2 = self.cross_attn2(src_cur, src_merge, value=src_merge, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn1(src_cur,src2)
                    src1 = torch.cat([token,src1],0)
                    q = k = self.with_pos_embed(src1, pos)
                    src2 = self.self_attn2(q, k, value=src1, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                    src_cur = self.ffn2(src1,src2)
                    src_before = src[:,src.shape[1]//self.merge_groups:,]
                    src = torch.cat([src_cur,src_before],1)  
        
            elif self.time_attn_type == 'crossattn_mlp':

                if self.count <= self.config.enc_layers-1:
                    if self.crossattn_last_layer:

                        if self.count==3:
                            src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                            src1_ori,src2_ori,src3_ori,src4_ori = src_512.chunk(4,0)

                            if self.use_box_pos:
                                src = src + box_pos
                            src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                            src1,src2,src3,src4 = src_512.chunk(4,0)
                            src_fusion = torch.cat([src1,src2,src3,src4],-1)
                            src_fusion = self.time_mlp_fusion(src_fusion)
                            ##
                            if self.mlp_cross_grid_pos:

                                k = self.with_pos_embed(src_fusion, pos[1:])
                                q1 = self.with_pos_embed(src1, pos[1:])
                                q2 = self.with_pos_embed(src2, pos[1:])
                                q3 = self.with_pos_embed(src3, pos[1:])
                                q4 = self.with_pos_embed(src4, pos[1:])
                                # src_fusion = src_fusion + pos[1:,:src_fusion.shape[1]]
                                    ##
                                if self.ms_pool:
                                    src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                                    src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                                    src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                                    src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                                    
                                if self.share_head:
                                    cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src1 = self.ffn2(src1,cross_src1)
                                    if self.update_234:
                                        cross_src2 = self.time_attn1(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                        src2 = self.ffn2(src2,cross_src2)
                                        cross_src3 = self.time_attn1(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                        src3 = self.ffn2(src3,cross_src3)
                                        cross_src4 = self.time_attn1(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                        src4 = self.ffn2(src4,cross_src4)
                                        src1234 = torch.cat([src1,src2,src3,src4],1)
                                    else:
                                        src1234 = torch.cat([src1,src2_ori,src3_ori,src4_ori],1)
                                    if self.pyramid and self.count==2:
                                        src = torch.cat([self.proj(src[:1]),src1234],0)
                                    else:
                                        src = torch.cat([src[:1],src1234],0)
                                else:
                                    cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src1 = self.ffn2(src1,cross_src1)
                                    cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src2 = self.ffn2(src2,cross_src2)
                                    cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src3 = self.ffn2(src3,cross_src3)
                                    cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src4 = self.ffn2(src4,cross_src4)
                                    src1234 = torch.cat([src1,src2,src3,src4],1)
                                    if self.pyramid and self.count==2:
                                        src = torch.cat([self.proj(src[:1]),src1234],0)
                                    else:
                                        src = torch.cat([src[:1],src1234],0)


                                q = k = self.with_pos_embed(src, pos)

                                src2 = self.self_attn2(q, k, value=src, attn_mask=src_mask,
                                                        key_padding_mask=src_key_padding_mask)[0]

                                src = self.ffn_last(src,src2)

                                    

                    else:
                        # src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                        # src1_ori,src2_ori,src3_ori,src4_ori = src_512.chunk(4,0)
                        if self.config.use_fc_token.enabled:
                            src_512 = src.view(src.shape[0]*4,-1,src.shape[-1])
                            src1,src2,src3,src4 = src_512.chunk(4,0) 
                        else:           
                            src_512 = src[1:].view((src.shape[0]-1)*4,-1,src.shape[-1])
                            src1,src2,src3,src4 = src_512.chunk(4,0)

                        src_fusion = torch.cat([src1,src2,src3,src4],-1)
                        src_fusion = self.time_mlp_fusion(src_fusion)
                        ##
                        if self.mlp_cross_grid_pos:

                            if self.pyramid and self.count>=2:
                                # k = self.with_pos_embed(src_fusion, pos_pyramid[1:])
                                # q1 = self.with_pos_embed(src1, pos_pyramid[1:])
                                # q2 = self.with_pos_embed(src2, pos_pyramid[1:])
                                # q3 = self.with_pos_embed(src3, pos_pyramid[1:])
                                # q4 = self.with_pos_embed(src4, pos_pyramid[1:])
                                pos_index = [i for i in range(0,64,2)]
                                k = self.with_pos_embed(src_fusion, pos[1:][pos_index])
                                q1 = self.with_pos_embed(src1, pos[1:][pos_index])
                                q2 = self.with_pos_embed(src2, pos[1:][pos_index])
                                q3 = self.with_pos_embed(src3, pos[1:][pos_index])
                                q4 = self.with_pos_embed(src4, pos[1:][pos_index])

                            else:
                                # import pdb;pdb.set_trace()
                                if self.config.use_mlp_cross_mixer:
                                    src_fusion = self.cross_mixer(src_fusion)

                                k = self.with_pos_embed(src_fusion, pos[1:])
                                q1 = self.with_pos_embed(src1, pos[1:])
                                q2 = self.with_pos_embed(src2, pos[1:])
                                q3 = self.with_pos_embed(src3, pos[1:])
                                q4 = self.with_pos_embed(src4, pos[1:])


                            if self.ms_pool:
                                src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                                src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                                src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                                src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                                
                            if self.share_head:
                                cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                src1 = self.ffn2(src1,cross_src1)
                                if self.update_234:
                                    cross_src2 = self.time_attn1(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src2 = self.ffn2(src2,cross_src2)
                                    cross_src3 = self.time_attn1(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src3 = self.ffn2(src3,cross_src3)
                                    cross_src4 = self.time_attn1(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                    src4 = self.ffn2(src4,cross_src4)
                                    src1234 = torch.cat([src1,src2,src3,src4],1)
                                else:
                                    src1234 = torch.cat([src1,src2_ori,src3_ori,src4_ori],1)
                                if self.pyramid and self.count==2:
                                    src = torch.cat([self.proj(src[:1]),src1234],0)
                                else:
                                    src = torch.cat([src[:1],src1234],0)
                            else:
                                cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                src1 = self.ffn2(src1,cross_src1)
                                cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                src2 = self.ffn2(src2,cross_src2)
                                cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                src3 = self.ffn2(src3,cross_src3)
                                cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                                src4 = self.ffn2(src4,cross_src4)
                                src1234 = torch.cat([src1,src2,src3,src4],1)
                            # if self.pyramid and self.count==2:
                                #     src = torch.cat([self.proj(src[:1]),src1234],0)
                                if self.config.use_fc_token.enabled:
                                    src = src1234
                                else:
                                    src = torch.cat([src[:1],src1234],0)
                        else:

                            cross_src1 = self.time_attn1(src1, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                            src1 = self.ffn2(src1,cross_src1)
                            cross_src2 = self.time_attn2(src2, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                            src2 = self.ffn2(src2,cross_src2)
                            cross_src3 = self.time_attn3(src3, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                            src3 = self.ffn2(src3,cross_src3)
                            cross_src4 = self.time_attn4(src4, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                            src4 = self.ffn2(src4,cross_src4)
                            src1234 = torch.cat([src1,src2,src3,src4],1)
                            if self.pyramid and self.count==2:
                                src = torch.cat([self.proj(src[:1]),src1234],0)
                            else:
                                src = torch.cat([src[:1],src1234],0)

            elif self.time_attn_type == 'crossattn_trans':
                
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                srcs = [i.view(1,-1,i.shape[-1]) for i in src_512.chunk(4,0)]
                # src_space = torch.cat(src_512.chunk(4,0),-1)
                srcs = torch.cat(srcs,0)
                src2 = self.cross_attn(srcs, srcs, value=srcs, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
                src2 = src2.view(-1,src.shape[1],src.shape[-1])
                src_point = self.ffn1(src[1:],src2)
                src = torch.cat([src[:1],src_point],0)
                # src_fusion = self.time_mlp_fusion(src_fusion)
                # ##
                # if self.mlp_cross_grid_pos:
                #     src_fusion = src_fusion + pos[1:,:src_fusion.shape[1]]
                # ##
                # if self.ms_pool:
                #     src_fusion_3d = src_fusion.permute(1,2,0,).contiguous().view(src_fusion.shape[1],src_fusion.shape[2],4,4,4)
                #     src_pool2 = self.max2_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                #     src_pool4 = self.max4_pool(src_fusion_3d).view(src_fusion_3d.shape[0],src_fusion_3d.shape[1],-1)
                #     src_fusion = torch.cat([src_fusion.permute(1,2,0),src_pool2,src_pool4],-1).permute(2,0,1).contiguous()
                    

                # cross_src1 = self.time_attn1(src1, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                # src1 = self.ffn2(src1,cross_src1)
                # cross_src2 = self.time_attn2(src2, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                # src2 = self.ffn2(src2,cross_src2)
                # cross_src3 = self.time_attn3(src3, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                # src3 = self.ffn2(src3,cross_src3)
                # cross_src4 = self.time_attn4(src4, src_fusion, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                # src4 = self.ffn2(src4,cross_src4)
                # src1234 = torch.cat([src1,src2,src3,src4],1)
                # if self.pyramid and self.count==2:
                #     src = torch.cat([self.proj(src[:1]),src1234],0)
                # else:
                #     src = torch.cat([src[:1],src1234],0)

            elif self.time_attn_type == 'mlp_mixer':
                src_512 = src[1:].view((src.shape[0]-1)*4,-1,src.shape[-1],1)
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_fusion = torch.cat([src1,src2,src3,src4],-1)
                src_fusion = self.cross_mixer(src_fusion)
                src = torch.cat([src[:1],src_fusion],0)

            elif self.time_attn_type == 'mlp_mixer_v2':
                if self.count < 3:
                    src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                    src1,src2,src3,src4 = src_512.chunk(4,0)
                    src_fusion = torch.cat([src1,src2,src3,src4],-1)
                    src_fusion = self.time_mlp_fusion(src_fusion)
                    cross_src1 = self.mlp_merge(torch.cat([src1, src_fusion],-1))
                    cross_src2 = self.mlp_merge(torch.cat([src2, src_fusion],-1))
                    cross_src3 = self.mlp_merge(torch.cat([src3, src_fusion],-1))
                    cross_src4 = self.mlp_merge(torch.cat([src4, src_fusion],-1))
                    src1234 = torch.cat([cross_src1,cross_src2,cross_src3,cross_src4],1)
                    src = torch.cat([src[:1],src1234],0)

            else:  
                raise NotImplementedError

        if self.use_mlp_query_decoder and self.count==3:
            # import pdb;pdb.set_trace()
            # import pdb;pdb.set_trace()

            src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
            src1,src2,src3,src4 = src_512.chunk(4,0)
            src_fusion = torch.cat([src1,src2,src3,src4],-1)
            src_fusion = self.time_mlp_fusion(src_fusion)
            ##

            k = self.with_pos_embed(src_fusion, pos[1:])
            q1 = self.with_pos_embed(src1, pos[1:])
            q2 = self.with_pos_embed(src2, pos[1:])
            q3 = self.with_pos_embed(src3, pos[1:])
            q4 = self.with_pos_embed(src4, pos[1:])

            src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
            src1,src2,src3,src4 = src_512.chunk(4,0)
            src_fusion = torch.cat([src1,src2,src3,src4],-1)
            src_fusion = self.time_mlp_fusion(src_fusion)

            cross_src1 = self.time_attn1(k, q1, value=src1, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
            src_fusion = self.ffn2(src_fusion,cross_src1)
            cross_src2 = self.time_attn2(k, q2, value=src2, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
            src_fusion = self.ffn2(src_fusion,cross_src2)
            cross_src3 = self.time_attn3(k, q3, value=src3, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
            src_fusion = self.ffn2(src_fusion,cross_src3)
            cross_src4 = self.time_attn4(k, q4, value=src4, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
            src_fusion = self.ffn2(src_fusion,cross_src4)

            src = torch.cat([src[:1,:src.shape[1]//4],src_fusion],0)

            q = k = self.with_pos_embed(src, pos)
            src_decoder = self.decoder_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
            src = self.decoder_ffn(src,src_decoder)


        if self.use_channel_weight == 'ct3d' and self.count>=3 and self.point128VS384:
            # import pdb;pdb.set_trace()
            # cls_token = self.norm_cls_token(src[0:1,:src.shape[1]//4])
            # q = cls_token
            # q = q.permute(1,2,0)   #should be [BS, C, N]
            # k = v = src[1:,:src.shape[1]//4].permute(1,2,0) 
            # cls_token = self.norm_cls_token(src[0:1])
            cls_token =  src[0:1]
            q = cls_token.permute(1,2,0)   #should be [BS, C, N]
            k = v = src[1:,].permute(1,2,0) 
            # q = k = cls_token # num_query, bs, hideen_dim [1,128,128]
            # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
            #                     key_padding_mask=tgt_key_padding_mask)
            # tgt = tgt + self.dropout1(tgt2)
            #q = self.norm_cls_token(q) #torch.Size([num_query, bs, C])
            src2 = self.channel_attn(q, k, value=v).permute(2,0,1)
            cls_token = cls_token + self.dropout3(src2)
            cls_token = self.norm3(cls_token)
            src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
            cls_token = cls_token + self.dropout5(src2)
            cls_token = self.norm4(cls_token)
            src = torch.cat([cls_token,src[1:]],0)

            ## this version only decoder current frame point. performance drop
            # cls_token = cls_token.view(-1,src.shape[1]//4,src.shape[2])
            # src_frame1 = torch.cat([cls_token,src[1:,:src.shape[1]//4]],0)
            # src = torch.cat([src_frame1,src[:,src.shape[1]//4:]],1)
        if self.config.use_fc_token.enabled:
            return src, tokens
        else:
            return src, torch.cat(src[:1].chunk(4,1),0)

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None,
                pos_pyramid = None, motion_feat=None,box_pos=None,empty_ball_mask=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,num_frames,pos_pyramid,motion_feat,box_pos=box_pos,empty_ball_mask=empty_ball_mask)

class TransformerEncoderLayerChannelTime(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,
                 use_channel_weight=False,time_attn=False,time_attn_type=False, use_motion_attn=False,share_head=True):
        super().__init__()
 
        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.time_attn_type = time_attn_type
        self.use_motion_attn = use_motion_attn
        self.time_attn = time_attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.share_head = share_head
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.use_channel_weight=='ct3d':
            self.channel_attn = MultiHeadedAttention(nhead, d_model)
        elif self.use_channel_weight=='channelwise':
            self.channel_attn = MultiHeadedAttentionChannelwise(nhead, d_model)

        if self.time_attn:
            self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            # if self.share_head:
            #     self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # else:
            # d_model = [256,127,43,43,43]
            self.time_attn1 = MultiHeadedAttentionChannelTime([256,127,43,43,43], nhead)
            self.time_attn2 = MultiHeadedAttentionChannelTime([256,127,43,43,43], nhead)
            self.time_attn3 = MultiHeadedAttentionChannelTime([256,127,43,43,43], nhead)
            self.time_attn4 = MultiHeadedAttentionChannelTime([256,127,43,43,43], nhead)
            
            if use_motion_attn:
                self.time_point_attn = MultiHeadedAttentionMotion(nhead, d_model)
            else:
                self.time_point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            self.time_token_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_cls_token = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        if self.share_head:
            self.ffn1 = FFN(d_model, dim_feedforward)
        else:
            self.ffn1 = FFN(d_model, dim_feedforward)
            self.ffn2 = FFN(d_model, dim_feedforward)
            self.ffn3 = FFN(d_model, dim_feedforward)
            self.ffn4 = FFN(d_model, dim_feedforward)

        if self.time_attn_type == 'time_mlp':
            self.time_mlp1 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp3 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            self.time_mlp4 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     time_token = None,motion_feat= None):
        # src = src[1:]
        src_rm_token = src[1:]
        q = k = self.with_pos_embed(src, pos)

        # if self.time_point_attn_pre and self.point128VS384:
        #     src_512 = src[1:].contiguous().view(512,-1,src.shape[-1])
        #     src1 = src_512[0:128]
        #     src234 = src_512[128:]
        #     src2 = self.time_point_attn(src1, src234, value=src234, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        #     src1 = src1 + self.dropout3(src2)
        #     src1 = self.norm3(src1)
        #     src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
        #     src1 = src1 + self.dropout5(src2)
        #     src_point = torch.cat([self.norm4(src1),src_512[128:]],0).view(128,-1,src.shape[-1])
        #     src = torch.cat([src[0:1],src_point],0)
        # import pdb;pdb.set_trace()

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        cls_token = None

        if self.time_attn:
            if self.time_attn_type == '128x384':
                src_512 = torch.cat([src[2:3],src[2:]],0).view(512,-1,src.shape[-1])
                src1 = src_512[0:128]
                if not motion_feat is None:
                    src2 = src_512[128:2*128]  + motion_feat[:,1:2]
                    src3 = src_512[2*128:3*128]+ motion_feat[:,2:3]
                    src4 = src_512[3*128:4*128]+ motion_feat[:,3:4]
                    src234 = torch.cat([src2,src3,src4],0)
                    #import pdb;pdb.set_trace()
                    src234 = src_512[128:] #[N,BS,C]
                    src2 = self.time_point_attn(src1.permute(1,2,0), src234.permute(1,2,0), value=src234.permute(1,2,0), motion_feat = motion_feat)[0]
                else:
                    src234 = src_512[128:]
                    src2 = self.time_point_attn(src1, src234, value=src234, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                src1 = src1 + self.dropout3(src2)
                src1 = self.norm3(src1)
                src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
                src1 = src1 + self.dropout5(src2)
                src_point = torch.cat([self.norm4(src1),src_512[128:]],0).view(128,-1,src.shape[-1])
                src = torch.cat([src[0:1],src_point],0)

            # if self.clstoken1VS384:
            #     # import pdb;pdb.set_trace()
            #     src_512 = src[1:].view(512,-1,src.shape[-1])
            #     src1 = src[0:1,:128]
            #     src234 = src_512[128:]
            #     src2 = self.time_point_attn(src1, src234, value=src234, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
            #     src1 = src1 + self.dropout3(src2)
            #     src1 = self.norm3(src1)
            #     src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
            #     src1 = src1 + self.dropout5(src2)
            #     src1 = self.norm4(src1)
            #     src_frame1 = torch.cat([src1,src_rm_token[:,:128]],0)
            #     src = torch.cat([src_frame1,src[:,128:]],1)

            # if self.point128VS128x3:
            #     src_512 = src[1:].view(512,-1,src.shape[-1])
            #     src_cur = src_512[0:128]
            #     src_cur_rep = src_cur.repeat(1,3,1)
            #     src_pre = torch.cat(src_512[128:].chunk(3,0),1)
            #     #import pdb;pdb.set_trace()
            #     if self.use_learn_time_token:
            #         #time_token = self.time_token_attn(time_token, src_pre, value=src_pre, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
            #         src_pre = src_pre + time_token
            #     src2 = self.time_point_attn(src_cur_rep, src_pre, value=src_pre, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
            #     src2 = src2.view(3,128,-1,src2.shape[-1]).sum(0)
            #     src1 = src_cur + self.dropout3(src2)
            #     src1 = self.norm3(src1)
            #     src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
            #     src1 = src1 + self.dropout5(src2)
            #     src1 = self.norm4(src1)#[:,:src1.shape[1]//3]
            #     src_point = torch.cat([self.norm2(src1),src_512[128:]],0).view(128,-1,src.shape[-1])
            #     src = torch.cat([src[:1],src_point],0)

            elif self.time_attn_type == '128x384x3':
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                # src_512 = src.view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_pre1 = torch.cat([src2,src3,src4],0)
                src_pre2 = torch.cat([src1,src3,src4],0)
                src_pre3 = torch.cat([src1,src2,src4],0)
                src_pre4 = torch.cat([src1,src2,src3],0)

                if self.share_head:
                    cross_src1 = self.time_attn1(src1, src_pre1, value=src_pre1, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src2 = self.time_attn2(src2, src_pre2, value=src_pre2, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn1(src2,cross_src2)
                    cross_src3 = self.time_attn3(src3, src_pre3, value=src_pre3, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn1(src3,cross_src3)
                    cross_src4 = self.time_attn4(src4, src_pre4, value=src_pre4, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn1(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([src[:1],src1234],0)
                    # src = torch.cat([src[:1],src1234],0)
                    # src = src1234 #torch.cat([src[:1],src1234],0)
                    # src = torch.cat([src[:1],src1234],0)

                else:
                    
                    cross_src1 = self.time_attn1(src1, src_pre1, value=src_pre1, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src1 = self.ffn1(src1,cross_src1)
                    cross_src2 = self.time_attn2(src2, src_pre2, value=src_pre2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src2 = self.ffn2(src2,cross_src2)
                    cross_src3 = self.time_attn3(src3, src_pre3, value=src_pre3, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src3 = self.ffn3(src3,cross_src3)
                    cross_src4 = self.time_attn4(src4, src_pre4, value=src_pre4, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
                    src4 = self.ffn4(src4,cross_src4)
                    src1234 = torch.cat([src1,src2,src3,src4],1)
                    src = torch.cat([src[:1],src1234],0)

            elif self.time_attn_type == 'time_mlp':
                src_512 = src[1:].view(src_rm_token.shape[0]*4,-1,src.shape[-1])
                src1,src2,src3,src4 = src_512.chunk(4,0)
                src_pre1 = torch.cat([src2,src3,src4],-1)
                src_pre2 = torch.cat([src1,src3,src4],-1)
                src_pre3 = torch.cat([src1,src2,src4],-1)
                src_pre4 = torch.cat([src1,src2,src3],-1)
                src1 = src1 + self.time_mlp1(src_pre1)
                src2 = src2 + self.time_mlp2(src_pre2)
                src3 = src3 + self.time_mlp3(src_pre3)
                src4 = src4 + self.time_mlp4(src_pre4)

                src1234 = torch.cat([src1,src2,src3,src4],1)
                src = torch.cat([src[:1],src1234],0)

            else:  
                raise NotImplementedError



     
        if self.use_channel_weight == 'channelwise' and self.count>=3 and self.point128VS384:
            # import pdb;pdb.set_trace()
            # cls_token = self.norm_cls_token(src[0:1,:src.shape[1]//4])
            # q = cls_token
            # q = q.permute(1,2,0)   #should be [BS, C, N]
            # k = v = src[1:,:src.shape[1]//4].permute(1,2,0) 
            # cls_token = self.norm_cls_token(src[0:1])
            cls_token =  src[0:1]
            q = cls_token.permute(1,2,0)   #should be [BS, C, N]
            k = v = src[1:,].permute(1,2,0) 
            # q = k = cls_token # num_query, bs, hideen_dim [1,128,128]
            # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
            #                     key_padding_mask=tgt_key_padding_mask)
            # tgt = tgt + self.dropout1(tgt2)
            #q = self.norm_cls_token(q) #torch.Size([num_query, bs, C])
            src2 = self.channel_attn(q, k, value=v).permute(2,0,1)
            cls_token = cls_token + self.dropout3(src2)
            cls_token = self.norm3(cls_token)
            src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
            cls_token = cls_token + self.dropout5(src2)
            cls_token = self.norm4(cls_token)
            src = torch.cat([src[:1],cls_token],0)
            # src = torch.cat([src_frame1,src[:,src.shape[1]//4:]],1)

        if self.use_channel_weight == 'ct3d' and self.count>=3 and self.point128VS384:
            # import pdb;pdb.set_trace()
            # cls_token = self.norm_cls_token(src[0:1,:src.shape[1]//4])
            # q = cls_token
            # q = q.permute(1,2,0)   #should be [BS, C, N]
            # k = v = src[1:,:src.shape[1]//4].permute(1,2,0) 
            # cls_token = self.norm_cls_token(src[0:1])
            cls_token =  src[0:1]
            q = cls_token.permute(1,2,0)   #should be [BS, C, N]
            k = v = src[1:,].permute(1,2,0) 
            # q = k = cls_token # num_query, bs, hideen_dim [1,128,128]
            # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
            #                     key_padding_mask=tgt_key_padding_mask)
            # tgt = tgt + self.dropout1(tgt2)
            #q = self.norm_cls_token(q) #torch.Size([num_query, bs, C])
            src2 = self.channel_attn(q, k, value=v).permute(2,0,1)
            cls_token = cls_token + self.dropout3(src2)
            cls_token = self.norm3(cls_token)
            src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
            cls_token = cls_token + self.dropout5(src2)
            cls_token = self.norm4(cls_token)
            src = torch.cat([cls_token,src[1:]],0)

            ## this version only decoder current frame point. performance drop
            # cls_token = cls_token.view(-1,src.shape[1]//4,src.shape[2])
            # src_frame1 = torch.cat([cls_token,src[1:,:src.shape[1]//4]],0)
            # src = torch.cat([src_frame1,src[:,src.shape[1]//4:]],1)
    
        # if self.time_attn:
        #     cls_token = cls_token if not cls_token is None else torch.cat(src[0:1].chunk(4,dim=1),0)
        #     q = k = self.with_pos_embed(cls_token,time_token) #[N, BS, C]
        #     src2 = self.time_attn(q, k, value=cls_token, attn_mask=src_mask,
        #                         key_padding_mask=src_key_padding_mask)[0]
        #     cls_token = cls_token + self.dropout3(src2)
        #     cls_token = self.norm3(cls_token)
        #     src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
        #     cls_token = cls_token + self.dropout5(src2)
        #     cls_token = self.norm4(cls_token)
        #     cls_token = cls_token.view(-1,src.shape[1],src.shape[2])

        # if not cls_token is None:
        #     src = torch.cat([cls_token,src_token],0)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, num_frames=None,
                time_token = None, motion_feat=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,num_frames,time_token,motion_feat)

class TransformerEncoderLayerPerciverIO(nn.Module):
    count = 0
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,
                 use_channel_weight=False,time_attn=False,point128VS384=False,point128VS128x3=False,
                 clstoken1VS384=False, use_learn_time_token=False,use_motion_attn=False):
        super().__init__()
        TransformerEncoderLayerCrossAttn.count += 1
        self.count = TransformerEncoderLayerCrossAttn.count
        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.point128VS384 = point128VS384
        self.point128VS128x3 = point128VS128x3
        self.clstoken1VS384 = clstoken1VS384
        self.use_motion_attn = use_motion_attn
        self.use_learn_time_token = use_learn_time_token
        self.time_attn = time_attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.count == 3:
            self.self_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.self_attn5 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.ffn7 = FFN(d_model, dim_feedforward)
            self.ffn8 = FFN(d_model, dim_feedforward)


        if self.use_channel_weight=='ct3d':
            self.channel_attn = MultiHeadedAttention(nhead, d_model)
        elif self.use_channel_weight=='channelwise':
            self.channel_attn = MultiHeadedAttentionChannelwise(nhead, d_model)

        if self.time_attn:
            self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            
            if use_motion_attn:
                self.time_point_attn = MultiHeadedAttentionMotion(nhead, d_model)
            else:
                self.time_point_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)



            self.time_token_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_cls_token = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)
        self.ffn5 = FFN(d_model, dim_feedforward)
        self.ffn6 = FFN(d_model, dim_feedforward)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     time_token = None,motion_feat= None):
        src_rm_token = src[1:]
        q = k = self.with_pos_embed(src, pos)

        # if self.time_point_attn_pre and self.point128VS384:
        #     src_512 = src[1:].contiguous().view(512,-1,src.shape[-1])
        #     src1 = src_512[0:128]
        #     src234 = src_512[128:]
        #     src2 = self.time_point_attn(src1, src234, value=src234, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        #     src1 = src1 + self.dropout3(src2)
        #     src1 = self.norm3(src1)
        #     src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
        #     src1 = src1 + self.dropout5(src2)
        #     src_point = torch.cat([self.norm4(src1),src_512[128:]],0).view(128,-1,src.shape[-1])
        #     src = torch.cat([src[0:1],src_point],0)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
 

        src_512 = src[1:].view(512,-1,src.shape[-1])
        src1,src2,src3,src4 = src_512.chunk(4,0)
        # src_cur = src_512[0:128]
        # src_cur_rep = src_cur.repeat(1,3,1)
        src_pre1 = torch.cat([src2,src3,src4],0)
        src_pre2 = torch.cat([src1,src3,src4],0)
        src_pre3 = torch.cat([src1,src2,src4],0)
        src_pre4 = torch.cat([src1,src2,src3],0)

        cross_src1 = self.time_attn1(src1, src_pre1, value=src_pre1, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src1 = self.ffn1(src1,cross_src1)
        q = k = v = src1  
        src1_refine = self.self_attn2(q,k,v)[0]
        src1_refine = self.ffn5(src1,src1_refine)
        q = k = v = src1_refine
        src1_refine2 = self.self_attn3(q,k,v)[0]
        src1 = self.ffn6(src1_refine,src1_refine2)
        cross_src2 = self.time_attn2(src2, src_pre2, value=src_pre2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src2 = self.ffn2(src2,cross_src2)
        cross_src3 = self.time_attn3(src3, src_pre3, value=src_pre3, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src3 = self.ffn3(src3,cross_src3)
        cross_src4 = self.time_attn4(src4, src_pre4, value=src_pre4, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src4 = self.ffn4(src4,cross_src4)
        src1234 = torch.cat([src1,src2,src3,src4],1)
        src = torch.cat([src[:1],src1234],0)

        if self.count == 3:
            q = k = v = src[:,:src.shape[1]//4]
            src1_refine2 = self.self_attn4(q,k,v)[0]
            src1_refine2 = self.ffn7(q,src1_refine2)
            q = k = v = src1_refine2
            src1_refine2 = self.self_attn5(q,k,v)[0]
            src1_refine2 = self.ffn8(q,src1_refine2)
            src = torch.cat([src1_refine2,src[:,src.shape[1]//4:]],1)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                time_token = None, motion_feat=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,time_token,motion_feat)

class TransformerEncoderLayerPerciver(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,use_channel_weight=False,time_attn=False):
        super().__init__()
        self.num_point = num_points
        self.use_channel_weight = use_channel_weight
        self.time_attn = time_attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if self.use_channel_weight:
            self.channel_attn = MultiHeadedAttention(nhead, d_model)
        if self.time_attn:
            self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     cls_token: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=None,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if self.use_channel_weight:
            q = cls_token.permute(1,2,0)   #should be [BS, C, N]
            k = v = src.permute(1,2,0) 
            src2 = self.channel_attn(q, k, value=v).permute(2,0,1)
            cls_token = cls_token + self.dropout3(src2)
            cls_token = self.norm3(cls_token)
            src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
            cls_token = cls_token + self.dropout5(src2)
            cls_token = self.norm4(cls_token)
            #cls_token = cls_token.view(-1,src.shape[1],src.shape[2])
    
        if self.time_attn:
            #cls_token = cls_token if not cls_token is None else torch.cat(src[0:1].chunk(4,dim=1),0)
            q = k = cls_token #[N, BS, C]
            src2 = self.time_attn(q, k, value=cls_token, attn_mask=None,
                                key_padding_mask=src_key_padding_mask)[0]
            cls_token = cls_token + self.dropout3(src2)
            cls_token = self.norm3(cls_token)
            src2 = self.linear4(self.dropout4(self.activation(self.linear3(cls_token))))
            cls_token = cls_token + self.dropout5(src2)
            cls_token = self.norm4(cls_token)
            #cls_token = cls_token.view(-1,src.shape[1],src.shape[2])

        # if not cls_token is None:
        #     src = torch.cat([cls_token,src_token],0)

        return src,cls_token

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                cls_token: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, cls_token, src_key_padding_mask, pos)
        return self.forward_post(src, cls_token, src_key_padding_mask, pos)

class TransformerEncoderLayerTwins(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None):
        super().__init__()
        self.num_point = num_points
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     time_token=None,motion_feat=None):

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        #index = [0,self.num_point,2*self.num_point,3*self.num_point]
        src1,src2,src3,src4 = src[8:].chunk(4,1)
        cls_token1,cls_token2,cls_token3,cls_token4 = src[0:8].chunk(4,1)
        twins_token1 = torch.cat([cls_token2,cls_token3,cls_token4],0)
        twins_token2 = torch.cat([cls_token1,cls_token3,cls_token4],0)
        twins_token3 = torch.cat([cls_token1,cls_token2,cls_token4],0)
        twins_token4 = torch.cat([cls_token1,cls_token2,cls_token3],0)
        # src1   = torch.cat(src[1:].chunk(4,dim=1),0)
        # src1 = src[1:,:128]
        # q = src_512[:128]

        cross_src1 = self.cross_attn1(src1, twins_token1, value=twins_token1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        cross_src1 = self.ffn1(src1,cross_src1)

        cross_src2 = self.cross_attn2(src2, twins_token2, value=twins_token2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        cross_src2 = self.ffn2(src2,cross_src2)

        cross_src3 = self.cross_attn3(src3, twins_token3, value=twins_token3, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        cross_src3 = self.ffn3(src3,cross_src3)

        cross_src4 = self.cross_attn4(src4, twins_token4, value=twins_token4, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        cross_src4 = self.ffn4(src4,cross_src4)


        src_points = torch.cat([cross_src1,cross_src2,cross_src3,cross_src4],1)
        src_tokens = torch.cat([cls_token1,cls_token2,cls_token3,cls_token4],1)
        src = torch.cat([src_tokens,src_points],0)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,motion_feat=None,
                time_token=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerCrossAttn5D(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None):
        super().__init__()
        self.num_point = num_points
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     time_cls_token,
                     time_pos_token,
                     pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        #index = [0,self.num_point,2*self.num_point,3*self.num_point]
        cls_token = torch.cat([time_cls_token,*src[0:1].chunk(4,dim=1)],0)
        cls_pos   = torch.cat([time_pos_token,*pos[0:1].chunk(4,dim=1)],0)
        q = k = self.with_pos_embed(cls_token, cls_pos)

        src2 = self.cross_attn(q, k, value=cls_token, attn_mask=None,
                              key_padding_mask=None)[0]
        src1 = cls_token + self.dropout3(src2)
        src1 = self.norm3(src1)
        src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
        src1 = cls_token + self.dropout5(src2)
        src1 = self.norm4(src1)
        time_cls_token = src1[0:1].view(1,src.shape[1]//4,src.shape[2])
        src1 = src1[1:].view(-1,src.shape[1],src.shape[2])
        src[0:1] = src1

        return src,time_cls_token

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                    time_cls_token,
                    time_pos_token,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, None, None, pos)
        return self.forward_post(src, time_cls_token, time_pos_token, pos)

class TransformerEncoderLayerEmbedAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None):
        super().__init__()
        self.num_point = num_points
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        #import pdb;pdb.set_trace()
        #index = [0,self.num_point,2*self.num_point,3*self.num_point]
        cls_token = torch.cat(src[0:1].chunk(4,dim=1),0)
        cls_pos   = torch.cat(pos[0:1].chunk(4,dim=1),0)
        q = k = self.with_pos_embed(cls_token, cls_pos)

        src2 = self.cross_attn(q, k, value=cls_token, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src1 = cls_token + self.dropout3(src2)
        src1 = self.norm3(src1)
        src2 = self.linear4(self.dropout4(self.activation(self.linear3(src1))))
        src1 = cls_token + self.dropout5(src2)
        src1 = self.norm4(src1)
        src1 = src1.view(-1,src.shape[1],src.shape[2])
        src[0:1] = src1

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerDeitTime(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        index = [0,128,128*2,128*3]
        src = src[index]
        q = k = self.with_pos_embed(src, pos[index])

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src).max(0,keepdim=True)[0]
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerWithGlobal(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,use_global_layer=False):
        super().__init__()
        self.use_global_layer = use_global_layer
        #self.local_attn = LocalAttention(d_model, nhead, attn_drop=dropout)
        self.local_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # if self.use_global_layer:
        #     self.global_attn = GlobalAttention(d_model, nhead,attn_drop=dropout)
        #     self.linear3 = nn.Linear(d_model, dim_feedforward)
        #     self.dropout3 = nn.Dropout(dropout)
        #     self.linear4 = nn.Linear(dim_feedforward, d_model)

        #     self.norm3 = nn.LayerNorm(d_model)
        #     self.norm4 = nn.LayerNorm(d_model)
        #     self.dropout4 = nn.Dropout(dropout)
        #     self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(src, pos)
        src2 = self.local_attn(q, k, value=src, attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # if self.use_global_layer:
            
            
        #     src2 = self.global_attn(src)
        #     src = src + self.dropout3(src2)
        #     src = self.norm3(src)
        #     src2 = self.linear4(self.dropout5(self.activation(self.linear3(src))))
        #     src = src + self.dropout4(src2)
        #     src = self.norm4(src)
        #     src = src.permute(1,0,2)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerTimeQKV(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.reduce_k = nn.Linear(d_model, d_model//4)
        self.reduce_v = nn.Linear(d_model, d_model//4)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        #src [N,bs,C]
        #src_reduce = self.reduce(src.permute(1,0,2).contiguous().view(src.shape[0],-1, src.shape[-1]//4))
        src_reduce_k = self.reduce_k(src.contiguous().view(-1,src.shape[1], src.shape[-1]//4))
        src_reduce_v = self.reduce_v(src.contiguous().view(-1,src.shape[1], src.shape[-1]//4))
        # point_split = src.shape[0]//4
        # c_s = src.shape[-1]//6
        # h_c = src.shape[-1]//2
        q = src[:,:,:src.shape[-1]//4] #current
        k = src_reduce_k.view(src.shape[0],src.shape[1],src.shape[-1]//4)
        v = src_reduce_v.view(src.shape[0],src.shape[1],src.shape[-1]//4)
        # k = torch.cat([src[:point_split,:,:128],src[point_split:point_split*2,:,h_c:h_c+c_s],\
        #                src[point_split*2:point_split*3,:,(h_c+c_s):h_c+2*c_s],src[point_split*3:,:,h_c+2*c_s:]],dim=-1)
        # v = torch.cat([src[:point_split,:,:128],src[point_split:point_split*2,:,h_c:h_c+c_s],\
        #                src[point_split*2:point_split*3,:,(h_c+c_s):h_c+2*c_s],src[point_split*3:,:,h_c+2*c_s:]],dim=-1)
        q = self.with_pos_embed(q, pos) #n,bs,c
        k = self.with_pos_embed(k, pos) #n,bs,c
        src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src[:,:,:src.shape[-1]//4] + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerBoxEncoding(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.boxmlp = MLP_v2([8,64,128,256])

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, box_seq=None,):
        current_idx = src.shape[1]//4
        src_current = src[:,:current_idx,:]
        src_pre = src[:,current_idx:,:]
        q = self.with_pos_embed(src_current, pos[:,:current_idx,:])
        k = self.with_pos_embed(src_pre, pos[:,current_idx:,:])
        v = self.with_pos_embed(src_pre, pos[:,current_idx:,:])
        src2 = self.self_attn(q, k, v)[0]
        src = src_current + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoderLayerSpaceTime(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.spactial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     srcs,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        src_list = [srcs[i*128:(i+1)*128] for i in range(4)]
        pos = torch.zeros_like(srcs[:128])
        for  i in range(4):
            src = src_list[i]
            q = k = self.with_pos_embed(src_list[i], pos)
            src2 = self.spactial_attn(q, k, value=src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src= self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        for j in range(3):
            q = self.with_pos_embed(src_list[0], pos)
            k = self.with_pos_embed(src_list[j+1], pos)
            src2 = self.time_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        srcs = torch.cat([src,*src_list[1:]],0)

        return srcs

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class CrossAttention(nn.Module):
    def __init__(self, query_dim=256, context_dim = 256, heads = 1,
                 dim_feedforward=512,dropout=0.1,activation="relu", normalize_before=False):
        super().__init__()
        dim_head = query_dim//heads
        context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, query_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, query_dim * 2, bias = False)
        self.to_out = nn.Linear(query_dim, query_dim)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(query_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src, context = None, mask = None):

        h = self.heads
        q = self.to_q(src)
        #context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b t h n d', h = h), (q, k, v))

        sim = einsum('b t h i d, b t h d j -> b t h i j', q, k.transpose(-2,-1)) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b t h i j, b t h j d -> b t h i d', attn, v)
        out = rearrange(out, 'b t h n d -> b t n (h d)', h = h)
        out = self.to_out(out)

        src = src + self.dropout1(out)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class Attention(nn.Module):
    def __init__(self, query_dim=256, context_dim = 256, heads = 8,
                  dim_feedforward=512,dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        dim_head = query_dim//heads
        context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, query_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, query_dim * 2, bias = False)
        self.to_out = nn.Linear(query_dim, query_dim)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(query_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src, src_mask = None, src_key_padding_mask= None, pos = None):
        h = self.heads

        q = self.to_q(src)
        #context = default(context, x)
        k, v = self.to_kv(src).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h d j -> b h i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        src = src + self.dropout1(out)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class GlobalAttention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., split=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim*4, dim * 2, bias=qkv_bias)
        self.reduce_channel = nn.Linear(dim, dim//split, bias=qkv_bias)
        #self.reduce_attn = nn.Linear(dim*4, dim, bias=qkv_bias)
        self.recuce_scale =  1/(dim//split)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):

        B, N, C = x.shape

        q = self.q(x).reshape(B,  N, self.num_heads, C // self.num_heads).permute(0,2,1,3) #[*,*,512,64]

        x = x.view(B,4,N//4,C)
        x1 = self.reduce_channel(x)
        x2 = x1.permute(0,1, 3, 2)
        attn = (x2@x1).reshape(B,4,-1) #[B,4,Cout]

        kv = self.kv(attn).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1] ##[*,*,4,C]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [*,*,512,64] @ [*,* 64, 4]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) #[*,* 512, 4]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LocalAttention(nn.Module):

    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., split=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        #self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.reduce_channel = nn.Linear(dim, dim//split, bias=qkv_bias)
        #self.reduce_attn = nn.Linear(dim*4, dim, bias=qkv_bias)
        self.recuce_scale =  1/(dim//split)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):

        B, N, C = x.shape
        x = x.contiguous().view(B,4,N//4,C)
        qkv = self.qkv(x).reshape(B, 4, N//4, 3, self.num_heads, C // self.num_heads).permute(3,0,1,4,2,5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2] ##[B,4,4,128,C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def attention_boxcoding(query, key,  value, boxencoding):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
    """
    key = key.permute(0,2,3,1)
    value = value.permute(0,2,3,1)
    query = query.permute(0,2,3,1)
    BS,H,N,C = query.size()
    
    #import pdb;pdb.set_trace()
    #energy =  torch.bmm(query,key.permute(0,2,1))/ C**-0.5 # transpose check
    energy =  torch.einsum('bdhm,bdmn->bdhn', query,key.permute(0,1,3,2))/ C**-0.5
    energy = energy + boxencoding[:,:,None,:].repeat_interleave(128,3)
    attention = F.softmax(energy) # BX (C) X (C) 
    # proj_value = value # B X C X N

    # out = torch.bmm(attention, value )
    out = torch.einsum('bdhm,bdmn->bdhn', attention, value)
    out = out.permute(2,0,1,3).contiguous().view(N, BS,H*C)

    return out,attention

def attention(query, key,  value):

    #[bs,num_chanel,head,num_query] query: torch.Size([128, 64, 4, 1]) key: torch.Size([bs, num_channel, head, point])
    dim = query.shape[1]
    # add channel and get weight for point shape =point [bs,num_chanel,head,num_query],[bs, num_channel, head, point])
    scores_1 = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5  # torch.Size([bs, num_head, num_query, points])
    # use N weight reweight memory(key) [bs, num_channel, head, point], [bs, num_head, num_query, points] >>> [bs_roi, feat, head, num_point]
    scores_2 = torch.einsum('abcd, aced->abcd', key, scores_1)        # [bs_roi, feat, head, num_point]
    # use channel-wise reweighted point throgh softmax get final N weight
    prob = torch.nn.functional.softmax(scores_2, dim=-1) #softmax on points [bs_roi, feat, head, num_point]
    # [bs, num_channel, head, num_point] * [bs, num_channel, head, point] >>> [bs,num_channel,head,num_channel]
    output = torch.einsum('bnhm,bdhm->bdhn', prob, value) #[bs,num_channel,head,num_channel] 
    return output, prob

def attention_channelwise(query, key,  value):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
    """
    # import pdb;pdb.set_trace()
    key = key.permute(0,2,1,3)
    value = value.permute(0,2,1,3)
    BS,H,C,N = key.size()
    query = query.permute(0,2,1,3).repeat(1,1,1,N)

    #energy =  torch.bmm(query,key.permute(0,2,1))/ C**-0.5 # transpose check
    energy =  torch.einsum('bdhm,bdmn->bdhn', query,key.permute(0,1,3,2))/ C**-0.5
    attention = F.softmax(energy) # BX (C) X (C) 
    # proj_value = value # B X C X N

    # out = torch.bmm(attention, value )
    out = torch.einsum('bdhm,bdmn->bdhn', attention, value)
    out = out.view(BS,H*C, N)

    return out,attention

def attention_channelwise(query, key,  value):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
    """
    # import pdb;pdb.set_trace()
    key = key.permute(0,2,1,3)
    value = value.permute(0,2,1,3)
    BS,H,C,N = key.size()
    query = query.permute(0,2,1,3).repeat(1,1,1,N)

    #energy =  torch.bmm(query,key.permute(0,2,1))/ C**-0.5 # transpose check
    energy =  torch.einsum('bdhm,bdmn->bdhn', query,key.permute(0,1,3,2))/ C**-0.5
    attention = F.softmax(energy) # BX (C) X (C) 
    # proj_value = value # B X C X N

    # out = torch.bmm(attention, value )
    out = torch.einsum('bdhm,bdmn->bdhn', attention, value)
    out = out.view(BS,H*C, N)

    return out,attention

def attention_channelTime(query, key,  value):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
    """
    # import pdb;pdb.set_trace()
    # key = key.permute(0,2,1,3)
    # value = value.permute(0,2,1,3)
    BS,H,N,C = key.size()
    # query = query.permute(0,2,1,3).repeat(1,1,1,N)

    #energy =  torch.bmm(query,key.permute(0,2,1))/ C**-0.5 # transpose check
    energy =  torch.einsum('bdhm,bdmn->bdhn', query,key.permute(0,1,3,2))/ C**-0.5
    attention = F.softmax(energy) # BX (C) X (C) 
    # proj_value = value # B X C X N

    # out = torch.bmm(attention, value )
    out = torch.einsum('bdhm,bdmn->bdhn', attention, value)
    out = out.permute(2,0,1,3).contiguous().view(N,BS,H*C)

    return out,attention

def attention_gridpos(query, key,  value,pos):
    """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
    """
    # key = key.permute(0,2,1,3)
    # value = value.permute(0,2,1,3)
    BS,H,N,C = key.size()
    # query = query.permute(0,2,1,3).repeat(1,1,1,N)

    #energy =  torch.bmm(query,key.permute(0,2,1))/ C**-0.5 # transpose check
    energy =  (torch.einsum('bdhm,bdmn->bdhn', query,key.permute(0,1,3,2)) + pos )/ C**-0.5
    attention = F.softmax(energy) # BX (C) X (C) 
    # proj_value = value # B X C X N

    # out = torch.bmm(attention, value )
    out = torch.einsum('bdhm,bdmn->bdhn', attention, value)
    out = out.permute(2,0,1,3).contiguous().view(N,BS,H*C)

    return out,attention

def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, query, key, value):

        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        
        x, prob = attention(query, key, value) #[bs,num_channel,head,num_channel]
                                                                                                  
        x = self.down_mlp(x) #torch.Size([128, 64, 4, 1]) merge the last num_channel weight to 1
        return x.contiguous().view(batch_dim, self.dim*self.num_heads, -1) #torch.Size([bs, C, 1])

class MultiHeadedAttentionGridpos(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, query, key, value,grid_pos):

        query = query.permute(1,0,2).contiguous()
        key = key.permute(1,0,2).contiguous()
        value = value.permute(1,0,2).contiguous()
        batch_dim,num_point = query.size(0),query.size(1)
        query, key, value = [l(x).view(batch_dim, num_point,self.num_heads,self.dim).permute(0,2,1,3).contiguous()
                             for l, x in zip(self.proj, (query, key, value))]
        
        x, prob = attention_gridpos(query, key, value,grid_pos) #[bs,num_channel,head,num_channel]
                                                                                                  
        return x #torch.Size([bs, C, 1])

class MultiHeadedAttentionChannelwise(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, d_model: int, num_heads: int ):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention_channelwise(query, key, value )
        # x = self.down_mlp(x) #torch.Size([128, 64, 4, 1])
        return x.contiguous().view(batch_dim, self.dim*self.num_heads, -1) #torch.Size([128, 256, 1])

class MultiHeadedAttentionChannelTime(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model[0] % num_heads == 0
        self.dim = d_model[0] // num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model[0], d_model[0])
        self.num_frames = len(d_model) -1
        k_list = []
        v_list = []
        for i in range(1,len(d_model)):
            k_list.append(nn.Linear(d_model[0], d_model[i]))
            v_list.append(nn.Linear(d_model[0], d_model[i]))
        # self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.multi_frames_k = nn.ModuleList(k_list)
        self.multi_frames_v = nn.ModuleList(v_list)
        # self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, current_q, previous_k, value, attn_mask=None, key_padding_mask=None):


        current_q = current_q.permute(1,0,2)
        previous_k = previous_k.permute(1,0,2)
        value = value.permute(1,0,2)
        batch_dim,num,c = current_q.shape
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]

        query = self.q(current_q).view(batch_dim, self.num_heads, self.dim, -1)
        key_list = []
        value_list = []
        previous_k = previous_k.chunk(3,1)
        previous_v = value.chunk(3,1)

        for i in range(self.num_frames):
            if i == 0:
                key_list.append(self.multi_frames_k[i](current_q))
                value_list.append(self.multi_frames_v[i](current_q))
            else:
                key_list.append(self.multi_frames_k[i](previous_k[i-1]))
                value_list.append(self.multi_frames_v[i](previous_v[i-1]))

        key   = torch.cat(key_list,-1).view(batch_dim, self.num_heads, num, self.dim )
        value = torch.cat(value_list,-1).view(batch_dim, self.num_heads, num, self.dim)

        x, prob = attention_channelTime(query, key, value )
        # x = self.down_mlp(x) #torch.Size([128, 64, 4, 1])
        return x, prob #torch.Size([128, 256, 1])

class MultiHeadedAttentionMotion(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.down_mlp = MLP(input_dim = d_model, hidden_dim = 32, output_dim = 4, num_layers = 1)


    def forward(self, query, key, value,motion_feat):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention_boxcoding(query, key, value, self.down_mlp(motion_feat).permute(0,2,1)[:,:,1:])
        # x = self.down_mlp(x) #torch.Size([128, 64, 4, 1])
        return x #torch.Size([128, 256, 1])

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, grid_pos=None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, grid_pos=grid_pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_channel_weight=True,split_time=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.use_channel_weight = use_channel_weight
        self.split_time = split_time
        if self.use_channel_weight:
            self.multihead_attn = MultiHeadedAttention(nhead, d_model)
        else:
            if self.split_time:
                self.norm_attn = CrossAttention(heads=4)
            else:
                self.norm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, grid_pos=None,
                     query_pos: Optional[Tensor] = None):


        q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        #memory [N,Bs,C]
        if self.use_channel_weight:
            # if (self.split_time and tgt.shape[0]==1):
            #     memory = memory.view(memory.shape[0]*4,memory.shape[1]//4,-1) # line44 CT3D.3epoch.SNT4attn-encode.1query.channeldecoder

            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).permute(1,2,0),#[Bs,C,N]
                                    key=self.with_pos_embed(memory, None).permute(1,2,0),
                                    value=memory.permute(1,2,0))  
            # tgt2 torch.Size([bs, C, num_query( 1 point)])
            tgt2 = tgt2.permute(2,0,1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        else:
            tgt2 = self.norm_attn(query=self.with_pos_embed(tgt, query_pos),#[Bs,C,N]
                        key=self.with_pos_embed(memory, pos),
                        value=memory)[0]
            # tgt2 torch.Size([bs, C, num_query( 1 point)])
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, grid_pos =None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, grid_pos, query_pos)

class TransformerDecoderLayerDeit(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,use_decoder=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.use_decoder = use_decoder
        if self.use_decoder.name == 'casc-channel':
            self.multihead_attn = MultiHeadedAttention(nhead, d_model)
        elif self.use_decoder.name == 'casc-norm':
            self.norm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif self.use_decoder.name == 'casc-norm256head':
            self.norm_attn = nn.MultiheadAttention(d_model, 256, dropout=dropout)
        else:
            print('wrong decoder name')
            exit()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, grid_pos = None,
                     query_pos: Optional[Tensor] = None):

        # if not self.use_decoder.local_decoder:
        #     q = k = self.with_pos_embed(tgt, query_pos) 
        #     tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)
        #     tgt = tgt + self.dropout1(tgt2)
        #     tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        # import pdb;pdb.set_trace()

        query = self.with_pos_embed(tgt, query_pos)

        if self.use_decoder.local_decoder:
            key = self.with_pos_embed(memory, pos[1:])
        else:
            key = self.with_pos_embed(memory, pos[1:].repeat(4,1,1))

        if self.use_decoder.name == 'casc-channel':
            tgt2 = self.multihead_attn(query=query.permute(1,2,0),#[Bs,C,N]
                                    key=key.permute(1,2,0),
                                    value=memory.permute(1,2,0))  
            # tgt2 torch.Size([bs, C, num_query( 1 point)])
            tgt2 = tgt2.permute(2,0,1)

            # elif not self.tgt_before_mean and tgt.shape[0] > 1:
            #     tgt = tgt.sum(0,True)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            tgt = tgt.view(4,-1,tgt.shape[-1])

        else:
            tgt2 = self.norm_attn(query=query,#[Bs,C,N]
                        key=key,
                        value=memory)[0]
            # tgt2 torch.Size([bs, C, num_query( 1 point)])
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            tgt = tgt.view(4,-1,tgt.shape[-1])

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,grid_pos=None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, grid_pos, query_pos)

class TransformerDecoderLayerMLPQuery(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,use_decoder=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.use_decoder = use_decoder
        if self.use_decoder.name == 'casc-channel':
            self.multihead_attn = MultiHeadedAttention(nhead, d_model)
        elif self.use_decoder.name == 'casc-norm':
            self.norm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        elif self.use_decoder.name == 'casc-norm256head':
            self.norm_attn = nn.MultiheadAttention(d_model, 256, dropout=dropout)
        else:
            print('wrong decoder name')
            exit()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, grid_pos = None,
                     query_pos: Optional[Tensor] = None):

        # if not self.use_decoder.local_decoder:
        #     q = k = self.with_pos_embed(tgt, query_pos) 
        #     tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)
        #     tgt = tgt + self.dropout1(tgt2)
        #     tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        # import pdb;pdb.set_trace()

        query = self.with_pos_embed(tgt, query_pos)

        if self.use_decoder.local_decoder:
            key = self.with_pos_embed(memory, pos[1:])
        else:
            key = self.with_pos_embed(memory, pos[1:].repeat(4,1,1))

        tgt2 = self.norm_attn(query=query,#[Bs,C,N]
                    key=key,
                    value=memory)[0]
        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        tgt = tgt.view(4,-1,tgt.shape[-1])

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,grid_pos=None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, grid_pos, query_pos)


class TransformerDecoderLayerDeitCascaded1(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,tgt_before_mean=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.tgt_before_mean = tgt_before_mean
        # if self.use_channel_weight:
        self.multihead_attn1 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn2 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn3 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn4 = MultiHeadedAttention(nhead, d_model)
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)
        self.ffn_norm1 = nn.LayerNorm(d_model)
        self.ffn_norm2 = nn.LayerNorm(d_model)
        self.ffn_norm3 = nn.LayerNorm(d_model)
        self.ffn_norm4 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, motion_feat = None,
                     query_pos: Optional[Tensor] = None):

        # import pdb;pdb.set_trace()
        # q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                     key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        num_query,_,_ = tgt.shape
        # if self.tgt_before_mean:
        #     tgt = tgt.mean(0,True)
        # else:
        #     tgt = tgt.view(1,-1,tgt.shape[-1])
        #     if not memory.shape[1]==tgt.shape[1]:
        #         memory = memory.view(memory.shape[0]//4,-1 ,memory.shape[-1])
        tgt1,tgt2,tgt3,tgt4 = tgt.chunk(4,1)
        memory1,memory2,memory3,memory4 = memory.chunk(4,1)
        tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        # memory [N,Bs,C]


        tgt_frame4 = self.multihead_attn4(query=self.with_pos_embed(tgt4, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory4, pos).permute(1,2,0),
                                value=memory4.permute(1,2,0)).permute(2,0,1)
        tgt4 = self.ffn4(tgt4,tgt_frame4)
        tgt3 =  torch.cat([tgt4.detach().clone(),tgt3],0).mean(0,True)

        tgt_frame3 = self.multihead_attn3(query=self.with_pos_embed(tgt3, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory3, pos).permute(1,2,0),
                                value=memory3.permute(1,2,0)).permute(2,0,1)
        tgt3 = self.ffn3(tgt3,tgt_frame3).detach().clone()
        tgt2 =  torch.cat([tgt3.detach().clone(),tgt2],0).mean(0,True)

        tgt_frame2 = self.multihead_attn3(query=self.with_pos_embed(tgt2, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory2, pos).permute(1,2,0),
                                value=memory2.permute(1,2,0)).permute(2,0,1)
        tgt2 = self.ffn2(tgt2,tgt_frame2).detach().clone()
        tgt1 =  torch.cat([tgt2.detach().clone(),tgt1],0).mean(0,True)

        tgt_frame1 = self.multihead_attn1(query=self.with_pos_embed(tgt1, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory1, pos).permute(1,2,0),
                                value=memory1.permute(1,2,0)).permute(2,0,1)
        tgt = self.ffn1(tgt1.detach().clone(),tgt_frame1)

        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        # tgt2 = tgt2.permute(2,0,1)

        # elif not self.tgt_before_mean and tgt.shape[0] > 1:
        #     tgt = tgt.sum(0,True)
        # tgt = tgt1 + self.dropout2(tgt_frame1)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)

        #import pdb;pdb.set_trace()
        tgt = torch.cat([tgt,tgt2,tgt3,tgt4],0)#.view(-1,tgt1.shape[1],tgt.shape[-1])
        # if self.tgt_after_mean:
            # tgt = tgt.view(4,-1,tgt.shape[-1])
            # tgt = tgt.mean(0,True)



        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayerDeitCascaded2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,tgt_before_mean=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.tgt_before_mean = tgt_before_mean
        # if self.use_channel_weight:
        self.multihead_attn1 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn2 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn3 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn4 = MultiHeadedAttention(nhead, d_model)
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)
        self.ffn_norm1 = nn.LayerNorm(d_model)
        self.ffn_norm2 = nn.LayerNorm(d_model)
        self.ffn_norm3 = nn.LayerNorm(d_model)
        self.ffn_norm4 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, motion_feat = None,
                     query_pos: Optional[Tensor] = None):

        # import pdb;pdb.set_trace()
        # q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                     key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        N,BS,C = tgt.shape
        # if self.tgt_before_mean:
        #     tgt = tgt.mean(0,True)
        # else:
        #     tgt = tgt.view(1,-1,tgt.shape[-1])
        #     if not memory.shape[1]==tgt.shape[1]:
        #         memory = memory.view(memory.shape[0]//4,-1 ,memory.shape[-1])
        # tgt1,tgt2,tgt3,tgt4 = tgt.chunk(4,1)
        tgt4 = self.cls_token.repeat(1,BS//4,1)
        memory1,memory2,memory3,memory4 = memory.chunk(4,1)
        tgt4 = self.norm1(tgt4) #torch.Size([num_query, bs, C])
        # memory [N,Bs,C]


        tgt_frame4 = self.multihead_attn4(query=self.with_pos_embed(tgt4, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory4, pos).permute(1,2,0),
                                value=memory4.permute(1,2,0)).permute(2,0,1)
        tgt4 = self.ffn4(tgt4,tgt_frame4)
        # tgt3 =  torch.cat([tgt4.detach().clone(),tgt3],0).mean(0,True)

        tgt_frame3 = self.multihead_attn3(query=self.with_pos_embed(tgt4, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory3, pos).permute(1,2,0),
                                value=memory3.permute(1,2,0)).permute(2,0,1)
        tgt3 = self.ffn3(tgt4,tgt_frame3).detach().clone()
        # tgt2 =  torch.cat([tgt3.detach().clone(),tgt2],0).mean(0,True)

        tgt_frame2 = self.multihead_attn3(query=self.with_pos_embed(tgt3, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory2, pos).permute(1,2,0),
                                value=memory2.permute(1,2,0)).permute(2,0,1)
        tgt2 = self.ffn2(tgt3,tgt_frame2).detach().clone()
        # tgt1 =  torch.cat([tgt2.detach().clone(),tgt1],0).mean(0,True)

        tgt_frame1 = self.multihead_attn1(query=self.with_pos_embed(tgt2, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory1, pos).permute(1,2,0),
                                value=memory1.permute(1,2,0)).permute(2,0,1)
        tgt1 = self.ffn1(tgt2.detach().clone(),tgt_frame1)

        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        # tgt2 = tgt2.permute(2,0,1)

        # elif not self.tgt_before_mean and tgt.shape[0] > 1:
        #     tgt = tgt.sum(0,True)
        # tgt = tgt1 + self.dropout2(tgt_frame1)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)

        tgt = torch.cat([tgt1,tgt2,tgt3,tgt4],0)#.view(-1,tgt1.shape[1],tgt.shape[-1])
        # if self.tgt_after_mean:
            # tgt = tgt.view(4,-1,tgt.shape[-1])
            # tgt = tgt.mean(0,True)



        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayerDeitCascaded3(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,tgt_before_mean=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.tgt_before_mean = tgt_before_mean
        # if self.use_channel_weight:
        self.multihead_attn1 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn2 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn3 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn4 = MultiHeadedAttention(nhead, d_model)
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)
        self.ffn_norm1 = nn.LayerNorm(d_model)
        self.ffn_norm2 = nn.LayerNorm(d_model)
        self.ffn_norm3 = nn.LayerNorm(d_model)
        self.ffn_norm4 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, motion_feat = None,
                     query_pos: Optional[Tensor] = None):

        # import pdb;pdb.set_trace()
        # q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                     key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        N,BS,C = tgt.shape
        # if self.tgt_before_mean:
        #     tgt = tgt.mean(0,True)
        # else:
        #     tgt = tgt.view(1,-1,tgt.shape[-1])
        #     if not memory.shape[1]==tgt.shape[1]:
        #         memory = memory.view(memory.shape[0]//4,-1 ,memory.shape[-1])
        tgt1,tgt2,tgt3,tgt4 = tgt.chunk(4,1)
        # tgt4 = self.cls_token.repeat(1,BS,1)
        memory1,memory2,memory3,memory4 = memory.chunk(4,1)
        tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        # memory [N,Bs,C]


        tgt_frame4 = self.multihead_attn4(query=self.with_pos_embed(tgt4, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory4, pos).permute(1,2,0),
                                value=memory4.permute(1,2,0)).permute(2,0,1)
        tgt4 = self.ffn4(tgt4,tgt_frame4)
        tgt3 =  torch.cat([tgt4,tgt3],0).mean(0,True)

        tgt_frame3 = self.multihead_attn3(query=self.with_pos_embed(tgt3, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory3, pos).permute(1,2,0),
                                value=memory3.permute(1,2,0)).permute(2,0,1)
        tgt3 = self.ffn3(tgt3,tgt_frame3).detach().clone()
        tgt2 =  torch.cat([tgt3,tgt2],0).mean(0,True)

        tgt_frame2 = self.multihead_attn3(query=self.with_pos_embed(tgt2, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory2, pos).permute(1,2,0),
                                value=memory2.permute(1,2,0)).permute(2,0,1)
        tgt2 = self.ffn2(tgt2,tgt_frame2).detach().clone()
        tgt1 =  torch.cat([tgt2,tgt1],0).mean(0,True)

        tgt_frame1 = self.multihead_attn1(query=self.with_pos_embed(tgt1, query_pos).permute(1,2,0),#[Bs,C,N]
                                key=self.with_pos_embed(memory1, pos).permute(1,2,0),
                                value=memory1.permute(1,2,0)).permute(2,0,1)
        tgt1 = self.ffn1(tgt1,tgt_frame1)

        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        # tgt2 = tgt2.permute(2,0,1)

        # elif not self.tgt_before_mean and tgt.shape[0] > 1:
        #     tgt = tgt.sum(0,True)
        # tgt = tgt1 + self.dropout2(tgt_frame1)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)

        #import pdb;pdb.set_trace()
        tgt = torch.cat([tgt1,tgt2,tgt3,tgt4],0)#.view(-1,tgt1.shape[1],tgt.shape[-1])
        # if self.tgt_after_mean:
            # tgt = tgt.view(4,-1,tgt.shape[-1])
            # tgt = tgt.mean(0,True)



        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayerChannelwise(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, tgt_after_mean=False,tgt_before_mean=False,weighted=True):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_after_mean = tgt_after_mean
        self.tgt_before_mean = tgt_before_mean
        # if self.use_channel_weight:
        self.multihead_attn = MultiHeadedAttentionChannelwise(nhead, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.weighted = weighted 


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # import pdb;pdb.set_trace()
        q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                     key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        num_query,_,_ = tgt.shape
        if self.tgt_before_mean:
            tgt = tgt.view(4,-1,tgt.shape[-1]).mean(0,True)
        # else:
        #     tgt = tgt.view(1,-1,tgt.shape[-1])
        #     if not memory.shape[1]==tgt.shape[1]:
        #         memory = memory.contiguous().view(memory.shape[0]//4,-1 ,memory.shape[-1])
        # tgt = self.norm1(tgt) #torch.Size([num_query, bs, C])
        # memory [N,Bs,C]


        # tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).permute(1,2,0),#[Bs,C,N]
        #                         key=self.with_pos_embed(memory, pos).permute(1,2,0),
        #                         value=memory.permute(1,2,0))  
        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        # import pdb;pdb.set_trace()
        query = tgt.permute(1,2,0)
        key =  memory.permute(1,2,0) #  BS,C, N
        # if self.weighted:
        #     dim = q.shape[1]
        #     # add channel and get weight for point shape =point [bs,num_chanel,head,num_query],[bs, num_channel, head, point])
        #     scores_1 = torch.einsum('bhn,bhm->bnm', query, key) / dim**.5  # torch.Size([bs, num_head, num_query, points])
        #     # use N weight reweight memory(key) [bs, num_channel, head, point], [bs, num_head, num_query, points] >>> [bs_roi, feat, head, num_point]
        #     key = torch.einsum('abd, aed->abd', key, scores_1)        # [bs_roi, feat, head, num_point]

        # query = query.permute(1,0,2)
        # key = key.permute(1,0,2) #  BS,C,N
        v = memory.permute(1,2,0)
        tgt2  = self.multihead_attn(query, key, value=v,)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if self.tgt_after_mean and num_query > 1:
            tgt = tgt.view(4,-1,tgt.shape[-1])
            tgt = tgt.mean(0,True)



        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayerDeitMultiHead(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,tgt_after_mean=False,tgt_before_mean=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn1 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn2 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn3 = MultiHeadedAttention(nhead, d_model)
        self.multihead_attn4 = MultiHeadedAttention(nhead, d_model)

        # Implementation of Feedforward model
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):


        # q = k = self.with_pos_embed(tgt, query_pos) # num_query, bs, hideen_dim [1,128,128]
        # tgt2,attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                     key_padding_mask=tgt_key_padding_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # if self.tgt_before_mean:
        #     tgt = tgt.mean(0,True)
        tgt1,tgt2,tgt3,tgt4 = self.norm1(tgt).chunk(4,0) #torch.Size([num_query, bs, C])
        #memory1,memory2,memory3,memory4, = memory.chunk(4,0) # use 128 to decode will drop
        memory1 = memory2 = memory3 = memory4 = memory # use 128 to decode will drop
        # memory [N,Bs,C]
        # import pdb;pdb.set_trace()

        tgt_decoder1 = self.multihead_attn1(query=self.with_pos_embed(tgt1, query_pos).permute(1,2,0),#[Bs,C,N]
                                     key=self.with_pos_embed(memory1, pos).permute(1,2,0),
                                   value=memory1.permute(1,2,0)).permute(2,0,1)

        tgt_decoder2 = self.multihead_attn2(query=self.with_pos_embed(tgt2, query_pos).permute(1,2,0),#[Bs,C,N]
                                     key=self.with_pos_embed(memory2, pos).permute(1,2,0),
                                   value=memory2.permute(1,2,0)).permute(2,0,1)

        tgt_decoder3 = self.multihead_attn3(query=self.with_pos_embed(tgt3, query_pos).permute(1,2,0),#[Bs,C,N]
                                     key=self.with_pos_embed(memory3, pos).permute(1,2,0),
                                   value=memory3.permute(1,2,0)).permute(2,0,1)

        tgt_decoder4 = self.multihead_attn4(query=self.with_pos_embed(tgt4, query_pos).permute(1,2,0),#[Bs,C,N]
                                     key=self.with_pos_embed(memory4, pos).permute(1,2,0),
                                   value=memory4.permute(1,2,0)).permute(2,0,1)
        
        tgt1 = self.ffn1(tgt1,tgt_decoder1)
        tgt2 = self.ffn2(tgt2,tgt_decoder2)
        tgt3 = self.ffn3(tgt3,tgt_decoder3)
        tgt4 = self.ffn4(tgt4,tgt_decoder4)
        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        # tgt_decoder = tgt_decoder.permute(2,0,1)
        # if self.tgt_after_mean:
        tgt = torch.cat([tgt1,tgt2,tgt3,tgt4],0).mean(0,True)
        # elif not self.tgt_before_mean:
        #     tgt = tgt.sum(0,True)
        # tgt = tgt + self.dropout2(tgt_decoder)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)



        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,dout=None,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt,tgt_decoder,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):



        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        #tgt_decoder = tgt_decoder.permute(2,0,1)
        tgt = tgt + self.dropout2(tgt_decoder)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class FFNUp(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,dout=None,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.din = d_model
        self.dout = dout
        if dout is None:
            self.dout = d_model
        if dout != d_model:
            self.proj = nn.Linear(self.din, self.dout)
        self.linear2 = nn.Linear(dim_feedforward, dout)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(dout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt,tgt_decoder,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):


        # import pdb;pdb.set_trace()
        # tgt2 torch.Size([bs, C, num_query( 1 point)])
        #tgt_decoder = tgt_decoder.permute(2,0,1)
        tgt = tgt + self.dropout2(tgt_decoder)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        if self.dout != self.din:
            tgt = self.proj(tgt) + self.dropout3(tgt2)
        else:
            tgt = tgt + self.dropout3(tgt2)

        tgt = self.norm3(tgt)

        return tgt

def build_transformer(args):

    if args.name=='deit':
        return TransformerDeit(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            use_decoder=args.use_decoder,
            #use_channel_weight = args.deit_channel_weight,

        )
    elif args.name=='deittime':
        return TransformerDeitTime(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            mlp_merge_query=args.mlp_merge_query,
            num_queries = args.num_queries,
            use_channel_weight = args.deit_channel_weight,
            time_attn = args.time_attn,
            point128VS384=args.point128VS384,
            point128VS128x3=args.point128VS128x3,
            use_learn_time_token = args.use_learn_time_token

        )

    elif args.name=='deit128x384':
        return TransformerDeit128x384(
            config = args,
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            mlp_residual=args.mlp_residual,
            num_queries = args.num_queries,
            use_channel_weight = args.deit_channel_weight,
            time_attn = args.time_attn,
            time_attn_type=args.time_attn_type,
            use_learn_time_token = args.use_learn_time_token,
            use_decoder=args.use_decoder,
            tgt_before_mean = args.tgt_before_mean,
            tgt_after_mean = args.tgt_after_mean,
            multi_decoder = args.multi_decoder,
            add_cls_token= args.add_cls_token,
            share_head = args.share_head,
            p4conv_merge= args.p4conv_merge,
            num_frames = args.num_frames,
            fusion_type = args.fusion_type,
            fusion_mlp_norm  = args.fusion_mlp_norm ,
            sequence_stride = args.sequence_stride,
            channel_time = args.channel_time,
            ms_pool=args.ms_pool,
            pyramid=args.pyramid,
            use_grid_pos = args.use_grid_pos,
            mlp_cross_grid_pos=args.mlp_cross_grid_pos,
            merge_groups=args.merge_groups,
            fusion_init_token = args.fusion_init_token,
            use_box_pos = args.use_box_pos,
            update_234 = args.update_234,
            use_1_frame=args.use_1_frame,
            crossattn_last_layer = args.crossattn_last_layer,
            share_sa_layer = args.share_sa_layer
        )

    elif args.name=='deitclear':
        return TransformerDeitClear(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            mlp_residual=args.mlp_residual,
            num_queries = args.num_queries,
            use_channel_weight = args.deit_channel_weight,
            time_attn = args.time_attn,
            time_attn_type=args.time_attn_type,
            use_learn_time_token = args.use_learn_time_token,
            use_decoder=args.use_decoder,
            tgt_before_mean = args.tgt_before_mean,
            tgt_after_mean = args.tgt_after_mean,
            multi_decoder = args.multi_decoder,
            add_cls_token= args.add_cls_token,
            share_head = args.share_head,
            p4conv_merge= args.p4conv_merge,
            num_frames = args.num_frames,
            fusion_type = args.fusion_type,
            sequence_stride = args.sequence_stride,
            channel_time = args.channel_time,
            ms_pool=args.ms_pool,
            pyramid=args.pyramid,
            use_grid_pos = args.use_grid_pos,
            mlp_cross_grid_pos=args.mlp_cross_grid_pos,
            merge_groups=args.merge_groups,
            fusion_init_token = args.fusion_init_token,
            use_box_pos = args.use_box_pos,
            update_234 = args.update_234,
            use_1_frame=args.use_1_frame,
            crossattn_last_layer = args.crossattn_last_layer,
            share_sa_layer = args.share_sa_layer
        )

    elif args.name=='maskencoder':
        return MaskedTransformerDeit128x384(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            mlp_merge_query=args.mlp_merge_query,
            num_queries = args.num_queries,
            use_channel_weight = args.deit_channel_weight,
            time_attn = args.time_attn,
            time_attn_type=args.time_attn_type,
            use_learn_time_token = args.use_learn_time_token,
            use_decoder=args.use_decoder,
            use_128_decoder=args.use_128_decoder,
            tgt_before_mean = args.tgt_before_mean,
            tgt_after_mean = args.tgt_after_mean,
            multi_decoder = args.multi_decoder,
            add_cls_token= args.add_cls_token,
            share_head = args.share_head,
            p4conv_merge= args.p4conv_merge,
            masking_radius= args.masking_radius
        )


    elif args.name=='perc':
        return TransformerDeitPerciverIO(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            mlp_merge_query=args.mlp_merge_query,
            num_queries = args.num_queries,
            use_channel_weight = args.deit_channel_weight,
            time_attn = args.time_attn,
            point128VS384=args.point128VS384,
            point128VS128x3=args.point128VS128x3,
            clstoken1VS384 = args.clstoken1VS384,
            use_learn_time_token = args.use_learn_time_token,
            time_point_attn_pre = args.use_motion_attn,
            use_decoder=args.use_decoder,
            use_128_decoder=args.use_128_decoder,
            tgt_before_mean = args.tgt_before_mean,
            tgt_after_mean = args.tgt_after_mean,
            multi_decoder = args.multi_decoder,
            add_cls_token= args.add_cls_token
        )

    elif args.name=='deittime5D':
        return TransformerDeitTime5D(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            cross_attn= args.cross_attn
        )

    elif args.name=='twins':
        return TransformerDeitTwins(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            use_decoder=args.use_decoder,
            tgt_after_mean = args.tgt_after_mean,
            tgt_before_mean = args.tgt_before_mean,
            multi_decoder = args.multi_decoder,
        )

    elif args.name=='swaptoken':
        return TransformerDeitSwapToken(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            num_point = args.num_points,
            use_decoder=args.use_decoder,
            tgt_after_mean = args.tgt_after_mean,
            tgt_before_mean = args.tgt_before_mean,
            multi_decoder = args.multi_decoder,
        )

    else:
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            # split_time=args.split_time
        )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
