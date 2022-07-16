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

# class TimeMixerBlock(nn.Module):

#     def __init__(self):
#         super().__init__()


#         self.mixer_x = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
#         self.mixer_y = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
#         self.mixer_z = MLP(input_dim = 4, hidden_dim = 16, output_dim = 4, num_layers = 3)
#         self.norm_x = nn.LayerNorm(256)
#         self.norm_y = nn.LayerNorm(256)
#         self.norm_z = nn.LayerNorm(256)
#         self.norm_channel = nn.LayerNorm(256)
#         self.ffn = FeedForward(256,512)

#     def forward(self, src):
#         # import pdb;pdb.set_trace()
#         mixed_x = self.mixer_x(src) #[0,1,2,3,4]
#         # mixed_x = src_3d.permute(0,2,3,4,1) + self.ffn(mixed_x)
#         mixed_x = src + mixed_x #torch.Size([64, 128, 256, 4])
#         mixed_x = self.norm_x(mixed_x.permute(0,1,3,2))

#         src_mixer = (mixed_x + self.ffn(mixed_x)).permute(0,2,1,3).contiguous().view(src.shape[0],-1,src.shape[2])
#         src_mixer = self.norm_channel(src_mixer)

#         return src_mixer

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

class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                 num_queries=None,num_point=None,share_head=True,merge_groups=None,
                 sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        # self.mlp_residual = mlp_residual
        self.num_queries = num_queries
        # self.use_channel_weight = use_channel_weight
        # self.time_attn = time_attn
        # self.use_learn_time_token = use_learn_time_token
        # self.use_decoder = use_decoder
        # self.use_t4_decoder = use_t4_decoder
        # self.multi_decoder = multi_decoder
        # self.weighted_sum = weighted_sum
        # self.add_cls_token = add_cls_token
        self.share_head = share_head
        # self.masking_radius = masking_radius
        self.num_frames = num_frames
        self.nhead = nhead
        # self.fusion_type = fusion_type
        # self.fusion_mlp_norm = fusion_mlp_norm
        self.sequence_stride = sequence_stride
        # self.channel_time = channel_time
        # self.time_attn_type = time_attn_type
        self.merge_groups = merge_groups
        # self.fusion_init_token = fusion_init_token
        # self.use_1_frame  = use_1_frame
        # self.crossattn_last_layer = crossattn_last_layer
 
        # self.fc_token = None


        encoder_layer = [TransformerEncoderLayerCrossAttn(self.config, d_model, d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before, num_point,merge_groups=merge_groups) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)
        # self.p4conv_merge = p4conv_merge

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reg_token = nn.Parameter(torch.zeros(self.num_frames, 1, d_model))
        # if self.use_learn_time_token:
        #     self.time_token = MLP(input_dim = 1, hidden_dim = 256, output_dim = d_model, num_layers = 2)

        # self.time_index = torch.tensor([0,1,2,3]).view([4,1]).float().cuda()

        
        if self.num_frames >4 :

            # if self.fusion_type == 'trans':
            #     # if self.num_frames > 4:
            #     #     self.mlp_merge = [CrossAttn(channel_time = self.channel_time).cuda() for i in range(4)]
            #     # if self.config.share_fusion_head:
            #     self.merge = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos,fusion_init_token=fusion_init_token,
            #                                     merge_groups=self.merge_groups,num_frames=self.num_frames,share_head=self.config.share_fusion_head,
            #                                     group_concat = self.config.group_concat,use_mlp_as_query=self.config.use_mlp_as_query,src_as_value=self.config.src_as_value)
            #     # else:
            #     #     self.merge_list = [CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos,fusion_init_token=fusion_init_token,
            #     #                                merge_groups=self.merge_groups,num_frames=self.num_frames).cuda() for i in range(self.merge_groups)]
            #     # self.mlp_merge2 = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos).cuda()
            #     # self.mlp_merge3 = CrossAttn(channel_time = self.channel_time,use_grid_pos=use_grid_pos).cuda()

            # elif 'mlp_mixer' in self.fusion_type:
            #     self.mlp_mixer = SpatialMixerBlock(256)
            #     if self.merge_groups:
            #         group = self.num_frames // self.merge_groups
            #     self.merge = MLP(input_dim = 256*group, hidden_dim = 256, output_dim = 256, num_layers = 4)

            # else:
  
            if self.merge_groups:
                group = self.num_frames // self.merge_groups
            self.merge = MLP(input_dim = self.config.hidden_dim*group, hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)
            self.layernorm = nn.LayerNorm(d_model)



        # if uselearnpos:
        #     self.pos_embed = nn.Parameter(torch.zeros(4, num_point+1, d_model))

        # if self.use_decoder.enabled:
        #     # if multi_decoder:
        #     #     decoder_layer = TransformerDecoderLayerDeitMultiHead(d_model, nhead, dim_feedforward,
        #     #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
        #     # elif channelwise_decoder.enabled:
        #     #     decoder_layer = TransformerDecoderLayerChannelwise(d_model, nhead, dim_feedforward,
        #     #                                 dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean,channelwise_decoder.weighted)
        #     if   self.use_decoder.name=='casc1':

        #         decoder_layer = TransformerDecoderLayerDeitCascaded1(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
        #     elif self.use_decoder.name=='casc2':

        #         decoder_layer = TransformerDecoderLayerDeitCascaded2(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)
        #     elif self.use_decoder.name=='casc3':

        #         decoder_layer = TransformerDecoderLayerDeitCascaded3(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before,tgt_after_mean,tgt_before_mean)

        #     else:
        #         decoder_layer = TransformerDecoderLayerDeit(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before,tgt_after_mean,self.use_decoder)
        #     decoder_norm = nn.LayerNorm(d_model)
        #     self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                     return_intermediate=return_intermediate_dec)

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

    def forward(self, src, src_mask=None,pos=None,num_frames=None):

        BS, N, C = src.shape
        self.num_point = N//num_frames
        src_merge = None
        group = self.merge_groups
        # import pdb;pdb.set_trace()
        if not pos is None:
            pos = pos.permute(1, 0, 2)

        # if pos_pyramid is not None:
        #     pos_pyramid = pos_pyramid.permute(1, 0, 2)
        """
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
        """
        if self.num_frames == 16:
            # import pdb;pdb.set_trace()
            # if self.fusion_type == 'mlp':
 
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
                    #             # [6,2,-2,-6]>[2,6,-2,-6], [5,1,-3,-7] > [1,5,3,7]

                    groups = torch.cat(groups,-1)
                    src_groups.append(groups)
            src_merge = torch.cat(src_groups,1)
            current_num = 4
            

            # if self.fusion_mlp_norm=='ffn':
                # if batch_dict['use_future_frames'][0]:
                #     src_near_t0 = torch.cat([src[:,(num_frames//2)*64 : (num_frames//2+1)*64],src[:,(num_frames//2 + 1)*64 : (num_frames//2 +2)*64],
                #                                 src[:,(num_frames//2-2)*64 : (num_frames//2-1)*64],src[:,(num_frames//2-1)*64 : (num_frames//2)*64],],1)
                #     #[0, -1, 2, 1]
                #     src = self.fusion_norm(src_near_t0,self.merge(src_merge))
                # else:
            src = self.fusion_norm(src[:,:current_num*64],self.merge(src_merge))
            # elif self.fusion_mlp_norm=='layernorm':
            #     src = src[:,:current_num*64] + self.merge(src_merge)
            #     src = self.layernorm(src)

            # if current_num == 4:
            src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
            src = torch.cat([src1,src2,src3,src4],dim=0)
            # else:
            #     src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

            #  elif self.fusion_type == 'max_pool':

            #     reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            #     reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
            #     reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
            #     reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

            #     if self.sequence_stride ==1:
            #         src_groups = src.view(src.shape[0],src.shape[1]//group ,-1).chunk(4,dim=1)
            #     elif self.sequence_stride > 1:
            #         length = num_frames//group
            #         src_groups = []
            #         for idx, i in enumerate(range(group)):
            #             groups = [src[:,(i+j*4)*64:(i+j*4+1)*64].unsqueeze(2) for j in range(length)]

            #             if batch_dict['use_future_frames'][0]:
            #                 if idx==0 or idx==1:
            #                     item = groups.pop(2)   
            #                     groups.insert(0, item)
            #                     # [8,4,0,-4]>[0,8,4,-4], [7,3,-1,-5] > [-1,7,3,-5]
            #                 else:
            #                     item = groups.pop(1)   
            #                     groups.insert(0, item)  
            #                      # [6,2,-2,-6]>[2,6,-2,-6], [5,1,-3,-7] > [1,5,3,7]

            #             groups = torch.cat(groups,2).max(2)[0]
            #             src_groups.append(groups)

            #     src_merge = torch.cat(src_groups,1)
            #     current_num = 4
                

            #     if self.fusion_mlp_norm=='ffn':
            #         if batch_dict['use_future_frames'][0]:
            #             src_near_t0 = torch.cat([src[:,(num_frames//2)*64 : (num_frames//2+1)*64],src[:,(num_frames//2 + 1)*64 : (num_frames//2 +2)*64],
            #                                      src[:,(num_frames//2-2)*64 : (num_frames//2-1)*64],src[:,(num_frames//2-1)*64 : (num_frames//2)*64],],1)
            #             #[0, -1, 2, 1]
            #             src = self.fusion_norm(src_near_t0,src_merge)
            #         else:
            #             src = self.fusion_norm(src[:,:current_num*64],src_merge)

            #     elif self.fusion_mlp_norm=='layernorm':
            #         src = src[:,:current_num*64] + self.merge(src_merge)
            #         src = self.layernorm(src)

            #     if current_num == 4:
            #         src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            #         src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            #         src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            #         src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
            #         src = torch.cat([src1,src2,src3,src4],dim=0)
            #     else:
            #         src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)

        else:

            # if self.use_1_frame:
            #     reg_token1 =  self.reg_token[0:1].repeat(BS,1,1)
            #     src = torch.cat([reg_token1,src],dim=1)
            # else:
                # import pdb;pdb.set_trace()
            # if self.num_queries > 1:
            #     if not center_token is None and not self.config.use_center_token_add:
            #         if self.config.share_center_token:
            #             reg_token1 = reg_token2 = reg_token3 = reg_token4 = center_token[:,0:1]
            #         else:
            #             reg_token1 = center_token[:,0:1]
            #             reg_token2 = center_token[:,1:2]
            #             reg_token3 = center_token[:,2:3]
            #             reg_token4 = center_token[:,3:4]
            #     else:
            # cls_token1 = self.cls_token[0:1].repeat(BS,1,1)
            reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
            reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
            reg_token4 = self.reg_token[3:4].repeat(BS,1,1)
            # else:
            #     reg_token1 =  self.reg_token.repeat(BS,1,1)
            #     reg_token2 = src[:,self.num_point:self.num_point+1]
            #     reg_token3 = src[:,2*self.num_point:2*self.num_point+1]
            #     reg_token4 = src[:,3*self.num_point:3*self.num_point+1]

            #time_token1,time_token2,time_token3,time_token4 = self.time_token.repeat(BS,1,1).chunk(4,1)
            
            # if self.add_cls_token:
            #     src1 = torch.cat([cls_token1,reg_token1,src[:,0:self.num_point-1]],dim=1)
            #     src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            #     src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            #     src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
            # else:
            #     if self.use_learn_time_token:
            #         time_token1,time_token2,time_token3,time_token4 = self.time_token(self.time_index).unsqueeze(1).repeat(1,self.num_point,1).chunk(4,0)
            #         src1 = torch.cat([reg_token1,time_token1+src[:,0:self.num_point]],dim=1)
            #         src2 = torch.cat([reg_token2,time_token2+src[:,self.num_point:2*self.num_point]],dim=1)
            #         src3 = torch.cat([reg_token3,time_token3+src[:,2*self.num_point:3*self.num_point]],dim=1)
            #         src4 = torch.cat([reg_token4,time_token4+src[:,3*self.num_point:]],dim=1)
            #     else:
            group_1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            group_2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            group_3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            group_4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)



            src = torch.cat([group_1,group_2,group_3,group_4],dim=0)


        src = src.permute(1, 0, 2)
        # if self.config.use_fc_token.enabled:
        #     src = src[1:]
        # time_token = self.time_token.repeat(BS,1,1)
        # time_token = time_token.permute(1, 0, 2)
        # import pdb;pdb.set_trace()

        memory,tokens = self.encoder(src, mask = src_mask, num_frames=num_frames,pos=pos) # num_point,bs,feat torch.Size([128, 128, 256])


        # if (self.p4conv_merge and num_frames==4 and self.merge_groups==1) or (self.time_attn_type in ['mlp_merge','trans_merge'] and self.merge_groups==1) \
        #     or self.use_1_frame:
        #     memory = memory[0:1]
        #     return memory, tokens, src_merge
        # elif self.merge_groups==2:
        #     memory = torch.cat(memory[0:1].chunk(2,dim=1),0)
        #     return memory, tokens, src_merge
        # elif self.config.use_mlp_query_decoder:
        #     return memory[0:1], tokens, src_merge
        # else:
        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
        return memory, tokens, src_merge

        # memory = memory.permute(1, 0, 2)
        # return memory, tokens, src_merge

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
            # if self.config.use_center_token_add:
            #     token_list.append(tokens+center_token.repeat(4,1,1))
            # else:
            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,torch.cat(token_list,0)

# class TransformerEncoderLayer(nn.Module):

#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before

#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos


#     def forward_post(self,
#                      src,
#                      src_mask: Optional[Tensor] = None,
#                      src_key_padding_mask: Optional[Tensor] = None,
#                      pos: Optional[Tensor] = None, num_frames=None,
#                      grid_pos = None,motion_feat= None,box_pos=None):

#         q = k = self.with_pos_embed(src, pos)

#         src2,attn = self.self_attn(q, k, value=src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)

#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src 

#     def forward_pre(self, src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#         src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         return src

#     def forward(self, src,
#                 src_mask: Optional[Tensor] = None,
#                 src_key_padding_mask: Optional[Tensor] = None,
#                 pos: Optional[Tensor] = None, num_frames=None,
#                 grid_pos = None, motion_feat=None,box_pos=None):
#         if self.normalize_before:
#             return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#         return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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

        self.config = config
        self.ms_pool = ms_pool
        self.num_point = num_points
        # self.use_channel_weight = use_channel_weight
        # self.time_attn_type = time_attn_type
        # self.use_motion_attn = use_motion_attn
        # self.time_attn = time_attn
        # self.pyramid = pyramid
        # self.use_grid_pos = use_grid_pos
        # self.mlp_cross_grid_pos = mlp_cross_grid_pos
        self.merge_groups = merge_groups
        # self.update_234 = update_234
        # self.crossattn_last_layer = crossattn_last_layer
        # self.share_sa_layer = share_sa_layer
        # self.add_extra_sa = add_extra_sa
        # self.use_mlp_query_decoder  = self.config.use_mlp_query_decoder
        # self.fc_token = fc_token


        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        # self.share_head = share_head
        # self.use_box_pos = use_box_pos
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # if d_model != dout:
        #     self.proj = nn.Linear(d_model, dout)

        # if self.use_channel_weight=='ct3d':
        #     self.channel_attn = MultiHeadedAttention(d_model,nhead)
        # if self.use_mlp_query_decoder and self.count==3:
        #     self.decoder_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.decoder_ffn = FFN(d_model, dim_feedforward)
        # import pdb;pdb.set_trace()
        # if self.time_attn:
        self.time_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # if self.share_head:
        #     self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # else:
        self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # if use_motion_attn:
        #     self.time_point_attn = MultiHeadedAttentionMotion(nhead, d_model)
        # else:
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
        # if self.share_head:
        #     self.ffn1 = FFN(d_model, dim_feedforward)
        # else:
        self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)
        self.ffn3 = FFN(d_model, dim_feedforward)
        self.ffn4 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type in ['time_mlp', 'crossattn_mlp', 'time_mlp_v2', 'mlp_mixer_v2']:
        self.time_mlp1 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
        self.time_mlp2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
        self.time_mlp3 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
        self.time_mlp4 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 2)
            
            # if self.count>=2 and self.pyramid:
            #     self.time_mlp_fusion = MLP(input_dim = 512*4, hidden_dim = 512, output_dim = 512, num_layers = 4)
            # else:
            # import pdb;pdb.set_trace()
            # if self.config.get('use_semlp',None):
            #     self.time_mlp_fusion = SEMLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model)
            # else:
        self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)

            # self.ffn1 = FFN(d_model, dim_feedforward)
        self.ffn2 = FFN(d_model, dim_feedforward)

        # if self.config.get('only_use_ca_for_ab', None):
        #     self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)

        #     self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        #     # if self.pyramid and self.count==2:
        #     #     #     self.ffn_up = FFNUp(d_model, dim_feedforward, dout=2*d_model)
        #     #     self.proj = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256*2, num_layers = 3)
        #     # else:
        #     #     self.ffn2 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type == 'mlp_merge' and self.count==3:
        #     self.time_mlp1 = MLP(input_dim = 256*(merge_groups-1), hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.ffn1 = FFN(d_model, dim_feedforward)
        #     self.ffn2 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type == 'trans_merge' and self.count==3:
        #     self.time_mlp1 = MLP(input_dim = 256*merge_groups, hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.ffn1 = FFN(d_model, dim_feedforward)
        #     self.ffn2 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type == 'trans_merge_cas':
        #     # self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     # self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.time_mlp_fusion1 = MLP(input_dim = 256*4, hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.time_mlp_fusion2 = MLP(input_dim = 256*3, hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.time_mlp_fusion3 = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.time_mlp_fusion4 = MLP(input_dim = 256*1, hidden_dim = 256, output_dim = 256, num_layers = 4)
        #     self.ffn1 = FFN(d_model, dim_feedforward)
        #     # self.ffn2 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type == 'crossattn_trans':
        #     self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.ffn1 = FFN(d_model, dim_feedforward)

        # if self.time_attn_type == 'mlp_mixer_v2':
        #     self.mlp_merge = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 3)

        # if self.crossattn_last_layer:
        #     self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #     self.ffn_last = FFN(d_model, dim_feedforward)

        # if self.pyramid and self.count==2:
        #     self.proj = MLP(input_dim = 256*2, hidden_dim = 256, output_dim = 256, num_layers = 3)
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
        # self.max2_pool = nn.MaxPool3d((2,2,2), (2,2,2), (0,0,0), ceil_mode=False)
        # self.max4_pool = nn.MaxPool3d((4,4,4), (4,4,4), (0,0,0), ceil_mode=False)

        # if self.config.use_mlp_mixer.enabled:
            # if self.config.use_mlp_mixer.use_v2:
            #     self.mixer = SpatialMixerBlockV2(self.config.use_mlp_mixer.hidden_dim)
            # else:
            #     # v1 use 16
        self.mixer = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,self.config.use_mlp_mixer.get('grid_size', 4),self.config.hidden_dim, self.config.use_mlp_mixer)
        #self.cross_mixer = TimeMixerBlock()




    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     pos_pyramid = None,motion_feat= None,box_pos=None,empty_ball_mask=None):

        # src_ori = src
        # src_rm_token = src[1:]

        # import pdb;pdb.set_trace()
        # if self.count==2 and self.pyramid:
        #     # src_points = self.proj(src[1:])
        #     # src_token = self.proj_token(src[:1])
        #     # src_3d = src_points.permute(1,2,0,).contiguous().view(src_points.shape[1],src_points.shape[2],4,4,4)
        #     # src_3d = self.max2_pool(src_3d).view(src_3d.shape[0],src_3d.shape[1],-1).permute(2,0,1)
        #     # src = torch.cat([src_token,src_3d],0)
        #     src_reducez = src_rm_token.view(32,2,src.shape[1],src.shape[-1]).permute(0,2,1,3).contiguous().view(32,src.shape[1],2*src.shape[-1])
        #     src_reducez = self.proj(src_reducez)
        #     src = torch.cat([src[0:1],src_reducez],0)
        #     src_rm_token = src[1:]


        # if self.pyramid and self.count >= 2:
        #     pos_index =[0] + [i for i in range(1,65,2)]
        #     q = k = self.with_pos_embed(src, pos[pos_index])
        # else:
        #     if self.config.use_fc_token.enabled:
        #         q = k = self.with_pos_embed(src, pos[1:])
        #     else:
        q = k = self.with_pos_embed(src, pos)

        # if self.add_extra_sa and self.count == 1:
        #     src_extra = self.self_attn_extra(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        #     src = self.ffn_extra(src,src_extra)
        #     q = k = self.with_pos_embed(src, pos)

        # if self.share_sa_layer.enabled:
        #     if self.config.use_mlp_mixer.enabled:

                # if not self.config.get('only_use_ca_for_ab', None):

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
        # if self.config.use_mlp_mixer.use_attn_reweight:
        #     src_mixer = src_mixer * attn_weight.permute(2,0,1).clone().detach()
        src = torch.cat([token,src_mixer],0)
        # cls_token = None
                # else:

                #     src_512 = src[1:].contiguous().view((src.shape[0]-1)*4,-1,src.shape[-1])
                #     src1,src2,src3,src4 = src_512.chunk(4,0)
                #     src_fusion = torch.cat([src1,src2,src3,src4],-1)
                #     src_fusion = self.time_mlp_fusion(src_fusion)
                #     k = self.with_pos_embed(src_fusion, pos[1:])
                #     q1 = self.with_pos_embed(src1, pos[1:])
                #     q2 = self.with_pos_embed(src2, pos[1:])
                #     q3 = self.with_pos_embed(src3, pos[1:])
                #     q4 = self.with_pos_embed(src4, pos[1:])
                #     cross_src1 = self.time_attn1(q1, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                #     src1 = self.ffn2(src1,cross_src1)
                #     cross_src2 = self.time_attn2(q2, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                #     src2 = self.ffn2(src2,cross_src2)
                #     cross_src3 = self.time_attn3(q3, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                #     src3 = self.ffn2(src3,cross_src3)
                #     cross_src4 = self.time_attn4(q4, k, value=src_fusion, attn_mask=None,key_padding_mask=src_key_padding_mask)[0]
                #     src4 = self.ffn2(src4,cross_src4)
                #     src1234 = torch.cat([src1,src2,src3,src4],1)

            # else:
            #     src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
            #                             key_padding_mask=src_key_padding_mask)[0]
            #     src = src + self.dropout1(src2)
            #     src = self.norm1(src)
            #     src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            #     src = src + self.dropout2(src2)
            #     src = self.norm2(src)
            #     cls_token = None
            #     if self.config.use_fc_token.enabled:
            #         if self.config.use_fc_token.share:
            #             tokens = self.fc_token(src.permute(1,0,2).contiguous().view(src.shape[1],-1)).view(4,-1,src.shape[-1])
            #         else:
            #             tokens = self.fc_token[self.count-1](src.permute(1,0,2).contiguous().view(src.shape[1],-1)).view(4,-1,src.shape[-1])


        # else:
        #     q1,q2,q3,q4 = q.chunk(4,1)
        #     k1,k2,k3,k4 = k.chunk(4,1)
        #     src1,src2,src3,src4 = src.chunk(4,1)
        #     src_sa1 = self.self_attn1(q1, k1, value=src1, attn_mask=src_mask,
        #                             key_padding_mask=src_key_padding_mask)[0]
        #     src_sa2 = self.self_attn2(q2, k2, value=src2, attn_mask=src_mask,
        #                             key_padding_mask=src_key_padding_mask)[0]
        #     src_sa3 = self.self_attn3(q3, k3, value=src3, attn_mask=src_mask,
        #                             key_padding_mask=src_key_padding_mask)[0]
        #     src_sa4 = self.self_attn4(q4, k4, value=src4, attn_mask=src_mask,
        #                             key_padding_mask=src_key_padding_mask)[0]

        #     if self.share_sa_layer.share_ffn:
        #         src1 = self.sa_ffn(src1,src_sa1)
        #         src2 = self.sa_ffn(src2,src_sa2)
        #         src3 = self.sa_ffn(src3,src_sa3)
        #         src4 = self.sa_ffn(src4,src_sa4)
        #     else:
        #         src1 = self.sa_ffn1(src1,src_sa1)
        #         src2 = self.sa_ffn2(src2,src_sa2)
        #         src3 = self.sa_ffn3(src3,src_sa3)
        #         src4 = self.sa_ffn4(src4,src_sa4)
            
        #     src = torch.cat([src1,src2,src3,src4],1)



        # if self.time_attn:

        # if self.time_attn_type == 'crossattn_mlp':

        if self.count <= self.config.enc_layers-1:

    
            src_512 = src[1:].view((src.shape[0]-1)*4,-1,src.shape[-1])
            src1,src2,src3,src4 = src_512.chunk(4,0)

            src_fusion = torch.cat([src1,src2,src3,src4],-1)
            src_fusion = self.time_mlp_fusion(src_fusion)
            # import pdb;pdb.set_trace()
            # if self.mlp_cross_grid_pos:


            # if self.config.use_mlp_cross_mixer:
            #     src_fusion = self.cross_mixer(src_fusion)

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

            src = torch.cat([src[:1],src1234],0)


            # else:  
            #     raise NotImplementedError

        # if self.config.use_fc_token.enabled:
        #     return src, tokens
        # else:
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

# def build_transformer(args):

#     if args.name=='deit128x384':
#         return TransformerDeit128x384(
#             config = args,
#             d_model=args.hidden_dim,
#             dropout=args.dropout,
#             nhead=args.nheads,
#             dim_feedforward=args.dim_feedforward,
#             num_encoder_layers=args.enc_layers,
#             num_decoder_layers=args.dec_layers,
#             normalize_before=args.pre_norm,
#             return_intermediate_dec=True,
#             num_point = args.num_points,
#             mlp_residual=args.mlp_residual,
#             num_queries = args.num_queries,
#             use_channel_weight = args.deit_channel_weight,
#             time_attn = args.time_attn,
#             time_attn_type=args.time_attn_type,
#             use_learn_time_token = args.use_learn_time_token,
#             use_decoder=args.use_decoder,
#             tgt_before_mean = args.tgt_before_mean,
#             tgt_after_mean = args.tgt_after_mean,
#             multi_decoder = args.multi_decoder,
#             add_cls_token= args.add_cls_token,
#             share_head = args.share_head,
#             p4conv_merge= args.p4conv_merge,
#             num_frames = args.num_frames,
#             fusion_type = args.fusion_type,
#             fusion_mlp_norm  = args.fusion_mlp_norm ,
#             sequence_stride = args.sequence_stride,
#             channel_time = args.channel_time,
#             ms_pool=args.ms_pool,
#             pyramid=args.pyramid,
#             use_grid_pos = args.use_grid_pos,
#             mlp_cross_grid_pos=args.mlp_cross_grid_pos,
#             merge_groups=args.merge_groups,
#             fusion_init_token = args.fusion_init_token,
#             use_box_pos = args.use_box_pos,
#             update_234 = args.update_234,
#             use_1_frame=args.use_1_frame,
#             crossattn_last_layer = args.crossattn_last_layer,
#             share_sa_layer = args.share_sa_layer
#         )

#     else:
#         return Transformer(
#             d_model=args.hidden_dim,
#             dropout=args.dropout,
#             nhead=args.nheads,
#             dim_feedforward=args.dim_feedforward,
#             num_encoder_layers=args.enc_layers,
#             num_decoder_layers=args.dec_layers,
#             normalize_before=args.pre_norm,
#             return_intermediate_dec=True,
#             # split_time=args.split_time
#         )


def build_transformer(args):


    return Transformer(
        config = args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        # num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        # return_intermediate_dec=True,
        num_point = args.num_points,
        # mlp_residual=args.mlp_residual,
        num_queries = args.num_queries,
        # use_channel_weight = args.deit_channel_weight,
        # time_attn = args.time_attn,
        # time_attn_type=args.time_attn_type,
        # use_learn_time_token = args.use_learn_time_token,
        # use_decoder=args.use_decoder,
        # tgt_before_mean = args.tgt_before_mean,
        # tgt_after_mean = args.tgt_after_mean,
        # multi_decoder = args.multi_decoder,
        # add_cls_token= args.add_cls_token,
        # share_head = args.share_head,
        # p4conv_merge= args.p4conv_merge,
        num_frames = args.num_frames,
        # fusion_type = args.fusion_type,
        # fusion_mlp_norm  = args.fusion_mlp_norm ,
        sequence_stride = args.sequence_stride,
        # channel_time = args.channel_time,
        # ms_pool=args.ms_pool,
        # pyramid=args.pyramid,
        # use_grid_pos = args.use_grid_pos,
        # mlp_cross_grid_pos=args.mlp_cross_grid_pos,
        merge_groups=args.merge_groups,
        # fusion_init_token = args.fusion_init_token,
        # use_box_pos = args.use_box_pos,
        # update_234 = args.update_234,
        # use_1_frame=args.use_1_frame,
        # crossattn_last_layer = args.crossattn_last_layer,
        # share_sa_layer = args.share_sa_layer
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