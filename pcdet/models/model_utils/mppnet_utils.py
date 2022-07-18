from os import getgrouplist
import torch.nn as nn
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import matplotlib.pyplot as plt


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
        mixed_x = src_3d + mixed_x
        mixed_x = self.norm_x(mixed_x.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_y = self.mixer_y(mixed_x.permute(0,1,2,4,3)).permute(0,1,2,4,3).contiguous()
        mixed_y =  mixed_x + mixed_y
        mixed_y = self.norm_y(mixed_y.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        mixed_z = self.mixer_z(mixed_y.permute(0,1,4,3,2)).permute(0,1,4,3,2).contiguous()

        mixed_z =  mixed_y + mixed_z
        mixed_z = self.norm_z(mixed_z.permute(0,2,3,4,1)).permute(0,4,1,2,3).contiguous()

        src_mixer = mixed_z.view(src.shape[1],src.shape[2],-1).permute(2,0,1)
        src_mixer = src_mixer + self.ffn(src_mixer)
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

class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                 num_queries=None,num_point=None,share_head=True,num_groups=None,
                 sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups

        encoder_layer = [TransformerEncoderLayerCrossAttn(self.config, d_model, d_model, nhead, dim_feedforward,
                        dropout, activation, normalize_before, num_point,num_groups=num_groups) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)

        self.reg_token = nn.Parameter(torch.zeros(self.num_frames, 1, d_model))

        
        if self.num_frames >4:
  
            if self.num_groups:
                group = self.num_frames // self.num_groups
            self.merge = MLP(input_dim = self.config.hidden_dim*group, hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)
            self.layernorm = nn.LayerNorm(d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None,pos=None,num_frames=None):

        BS, N, C = src.shape
        self.num_point = N//num_frames
        src_merge = None
        group = self.num_groups

        if not pos is None:
            pos = pos.permute(1, 0, 2)
        if self.num_frames == 16:

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
                    groups = torch.cat(groups,-1)
                    src_groups.append(groups)
            src_merge = torch.cat(src_groups,1)

            src = self.fusion_norm(src[:,:group*64],self.merge(src_merge))

            src1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            src2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            src3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            src4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)
            src = torch.cat([src1,src2,src3,src4],dim=0)

        else:

            reg_token1 = self.reg_token[0:1].repeat(BS,1,1)
            reg_token2 = self.reg_token[1:2].repeat(BS,1,1)
            reg_token3 = self.reg_token[2:3].repeat(BS,1,1)
            reg_token4 = self.reg_token[3:4].repeat(BS,1,1)

            group_1 = torch.cat([reg_token1,src[:,0:self.num_point]],dim=1)
            group_2 = torch.cat([reg_token2,src[:,self.num_point:2*self.num_point]],dim=1)
            group_3 = torch.cat([reg_token3,src[:,2*self.num_point:3*self.num_point]],dim=1)
            group_4 = torch.cat([reg_token4,src[:,3*self.num_point:]],dim=1)



            src = torch.cat([group_1,group_2,group_3,group_4],dim=0)


        src = src.permute(1, 0, 2)
        memory,tokens = self.encoder(src, mask = src_mask, num_frames=num_frames,pos=pos) 

        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
        return memory, tokens, src_merge



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
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

            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,token_list

class TransformerEncoderLayerCrossAttn(nn.Module):
    count = 0
    def __init__(self, config, d_model, dout, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,num_groups=None):
        super().__init__()
        TransformerEncoderLayerCrossAttn.count += 1
        self.count = TransformerEncoderLayerCrossAttn.count

        self.config = config
        self.num_point = num_points
        self.num_groups= num_groups
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.time_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.time_attn4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.time_mlp_fusion = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)


        self.ffn2 = FFN(d_model, dim_feedforward)


        self.mixer = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,self.config.use_mlp_mixer.get('grid_size', 4),self.config.hidden_dim, self.config.use_mlp_mixer)





    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, num_frames=None,
                     pos_pyramid = None,motion_feat= None,box_pos=None,empty_ball_mask=None):

        q = k = self.with_pos_embed(src, pos)

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

        src = torch.cat([token,src_mixer],0)


        if self.count <= self.config.enc_layers-1:

    
            src_512 = src[1:].view((src.shape[0]-1)*4,-1,src.shape[-1])
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

            src = torch.cat([src[:1],src1234],0)

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


        tgt = tgt + self.dropout2(tgt_decoder)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

def build_transformer(args):


    return Transformer(
        config = args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        num_point = args.num_points,
        num_frames = args.num_frames,
        sequence_stride = args.get('sequence_stride',1),
        num_groups=args.num_groups,
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