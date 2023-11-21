import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from .cgnl import SpatialCGNL
from .utils import build_mlp

class CGNLNet(nn.Module):
    def __init__(self, object_relation_cfg, input_dim=256):
        super(CGNLNet, self).__init__()
        self.global_information = object_relation_cfg.GLOBAL_INFORMATION if 'GLOBAL_INFORMATION' in object_relation_cfg  else None
        self.drop_out = object_relation_cfg.DP_RATIO
        self.skip_connection = False if 'SKIP_CONNECTION' not in object_relation_cfg else object_relation_cfg.SKIP_CONNECTION

        self.cgnl_input_dim = input_dim
        if self.global_information:
            self.cgnl_input_dim += object_relation_cfg.GLOBAL_INFORMATION.MLP_LAYERS[-1] if not self.global_information.CONCATENATED else object_relation_cfg.GLOBAL_INFORMATION.MLP_LAYERS[-1] + 8
        # groups = 8 from the CGNL paper
        self.cgnl_1 = SpatialCGNL(self.cgnl_input_dim, int(self.cgnl_input_dim / 2), use_scale=False, groups=8)
        self.cgnl_2 = SpatialCGNL(self.cgnl_input_dim, int(self.cgnl_input_dim / 2), use_scale=False, groups=8)
        self.conv = torch.nn.Conv1d(self.cgnl_input_dim,self.cgnl_input_dim,1)
        self.bn = torch.nn.BatchNorm1d(self.cgnl_input_dim)

        if self.global_information:
            global_mlp_input_dim = input_dim + 8 if self.global_information.CONCATENATED else 8
            self.global_info_mlp = build_mlp(global_mlp_input_dim, self.global_information.MLP_LAYERS, activation='ReLU', bn=True, drop_out=self.drop_out)
        
        self.init_weights()


    def forward(self, batch_dict):
        (B, N, C) = batch_dict['pooled_features'].shape
        assert math.sqrt(N) == int(math.sqrt(N)), "N must be a square number"
        pooled_features = batch_dict['pooled_features']
        initial_pooled_features = pooled_features
        proposal_boxes = batch_dict['rois']
        proposal_labels = batch_dict['roi_labels']

        if self.global_information:
            global_information = torch.cat((proposal_boxes, proposal_labels.unsqueeze(-1)), dim=-1).view(B*N, -1)
            embedded_global_information = self.global_info_mlp(global_information)
            pooled_features = torch.cat([pooled_features, embedded_global_information.view(B,N,-1)], dim=-1)
            C = pooled_features.shape[-1]

        # permute to form image plane
        pooled_features = pooled_features.permute((0,2,1)).contiguous()
        pooled_features_plane = pooled_features.view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        pooled_features_plane = self.cgnl_1(pooled_features_plane)
        pooled_features_plane = self.cgnl_2(pooled_features_plane)

        related_features = pooled_features_plane.view(B, C, N)
        related_features = F.relu(self.bn(self.conv(related_features)))
        # permute back to (B,N,C)
        related_features = related_features.permute((0,2,1)).contiguous().view(B,N,C)

        if self.skip_connection:
            related_features = torch.cat([related_features, initial_pooled_features], dim=-1)

        batch_dict['related_features'] = related_features
        return batch_dict

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_uniform_
        if self.global_information:
            for m in self.global_info_mlp:
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
    
    def get_output_dim(self):
        return self.input_dim



