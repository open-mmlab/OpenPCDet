from os import getgrouplist
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_


class PointNetfeat(nn.Module):
    def __init__(self, input_dim, x=1,outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel==256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(input_dim, 64 * x, 1)
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
    def __init__(self, input_dim, joint_feat=False,model_cfg=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat
        channels = model_cfg.TRANS_INPUT

        times=1
        self.feat = PointNetfeat(input_dim, 1)

        self.fc1 = nn.Linear(512, 256 )
        self.fc2 = nn.Linear(256, channels)

        self.pre_bn = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.fc_s1 = nn.Linear(channels*times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
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

        x = F.relu(self.fc_ce1(feat))
        centers = self.fc_ce2(x)

        x = F.relu(self.fc_s1(feat))
        sizes = self.fc_s2(x)

        x = F.relu(self.fc_hr1(feat))
        headings = self.fc_hr2(x)

        return torch.cat([centers, sizes, headings],-1),feat,feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SpatialMixerBlock(nn.Module):

    def __init__(self,hidden_dim,grid_size,channels,config=None,dropout=0.0):
        super().__init__()


        self.mixer_x = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_y = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.mixer_z = MLP(input_dim = grid_size, hidden_dim = hidden_dim, output_dim = grid_size, num_layers = 3)
        self.norm_x = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
                               nn.Linear(channels, 2*channels),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(2*channels, channels),
                               )
        self.config = config
        self.grid_size = grid_size

    def forward(self, src):

        src_3d = src.permute(1,2,0).contiguous().view(src.shape[1],src.shape[2],
                                   self.grid_size,self.grid_size,self.grid_size)
        src_3d = src_3d.permute(0,1,4,3,2).contiguous() 
        mixed_x = self.mixer_x(src_3d)
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

class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                num_lidar_points=None,num_proxy_points=None, share_head=True,num_groups=None,
                sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups
        self.num_proxy_points = num_proxy_points
        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward,dropout, activation, 
                      normalize_before, num_lidar_points,num_groups=num_groups) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)

        self.token = nn.Parameter(torch.zeros(self.num_groups, 1, d_model))

        
        if self.num_frames >4:
  
            self.group_length = self.num_frames // self.num_groups
            self.fusion_all_group = MLP(input_dim = self.config.hidden_dim*self.group_length, 
               hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos=None):

        BS, N, C = src.shape
        if not pos is None:
            pos = pos.permute(1, 0, 2)
            
        if self.num_frames == 16:
            token_list = [self.token[i:(i+1)].repeat(BS,1,1) for i in range(self.num_groups)]
            if self.sequence_stride ==1:
                src_groups = src.view(src.shape[0],src.shape[1]//self.num_groups ,-1).chunk(4,dim=1)

            elif self.sequence_stride ==4:
                src_groups = []

                for i in range(self.num_groups):
                    groups = []
                    for j in range(self.group_length):
                        points_index_start = (i+j*self.sequence_stride)*self.num_proxy_points
                        points_index_end = points_index_start + self.num_proxy_points
                        groups.append(src[:,points_index_start:points_index_end])

                    groups = torch.cat(groups,-1)
                    src_groups.append(groups)

            else:
                raise NotImplementedError

            src_merge = torch.cat(src_groups,1)
            src = self.fusion_norm(src[:,:self.num_groups*self.num_proxy_points],self.fusion_all_group(src_merge))
            src = [torch.cat([token_list[i],src[:,i*self.num_proxy_points:(i+1)*self.num_proxy_points]],dim=1) for i in range(self.num_groups)]
            src = torch.cat(src,dim=0)

        else:
            token_list = [self.token[i:(i+1)].repeat(BS,1,1) for i in range(self.num_groups)]
            src = [torch.cat([token_list[i],src[:,i*self.num_proxy_points:(i+1)*self.num_proxy_points]],dim=1) for i in range(self.num_groups)]
            src = torch.cat(src,dim=0)

        src = src.permute(1, 0, 2)
        memory,tokens = self.encoder(src,pos=pos) 

        memory = torch.cat(memory[0:1].chunk(4,dim=1),0)
        return memory, tokens
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,
                pos: Optional[Tensor] = None):

        token_list = []
        output = src
        for layer in self.layers:
            output,tokens = layer(output,pos=pos)
            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,token_list


class TransformerEncoderLayer(nn.Module):
    count = 0
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
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

        if self.layer_count <= self.config.enc_layers-1:
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(self.num_groups):
                self.cross_attn_layers.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))

            self.ffn = FFN(d_model, dim_feedforward)
            self.fusion_all_groups = MLP(input_dim = d_model*4, hidden_dim = d_model, output_dim = d_model, num_layers = 4)
    

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialMixerBlock(self.config.use_mlp_mixer.hidden_dim,self.config.use_mlp_mixer.get('grid_size', 4),self.config.hidden_dim, self.config.use_mlp_mixer)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):

        src_intra_group_fusion = self.mlp_mixer_3d(src[1:])
        src = torch.cat([src[:1],src_intra_group_fusion],0)

        token = src[:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion

        src_summary = self.self_attn(token, key, value=src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)
        src = torch.cat([token,src[1:]],0)

        if self.layer_count <= self.config.enc_layers-1:
    
            src_all_groups = src[1:].view((src.shape[0]-1)*4,-1,src.shape[-1])
            src_groups_list = src_all_groups.chunk(self.num_groups,0)

            src_all_groups = torch.cat(src_groups_list,-1)
            src_all_groups_fusion = self.fusion_all_groups(src_all_groups)

            key = self.with_pos_embed(src_all_groups_fusion, pos[1:])
            query_list = [self.with_pos_embed(query, pos[1:]) for query in src_groups_list]

            inter_group_fusion_list = []
            for i in range(self.num_groups):
                inter_group_fusion = self.cross_attn_layers[i](query_list[i], key, value=src_all_groups_fusion)[0]
                inter_group_fusion = self.ffn(src_groups_list[i],inter_group_fusion)
                inter_group_fusion_list.append(inter_group_fusion)

            src_inter_group_fusion = torch.cat(inter_group_fusion_list,1)

            src = torch.cat([src[:1],src_inter_group_fusion],0)

        return src, torch.cat(src[:1].chunk(4,1),0)

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src,  pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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

    def forward(self, tgt,tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
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
        num_lidar_points = args.num_lidar_points,
        num_proxy_points = args.num_proxy_points,
        num_frames = args.num_frames,
        sequence_stride = args.get('sequence_stride',1),
        num_groups=args.num_groups,
    )

