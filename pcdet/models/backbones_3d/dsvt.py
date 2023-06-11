import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from math import ceil

from pcdet.models.model_utils.dsvt_utils import get_window_coors, get_inner_win_inds_cuda, get_pooling_index, get_continous_inds
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned


class DSVT(nn.Module):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    '''
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.input_layer = DSVTInputLayer(self.model_cfg.INPUT_LAYER)
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        # save GPU memory
        self.use_torch_ckpt = self.model_cfg.get('USE_CHECKPOINT', False)
 
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_module = _get_block_module(block_name_this_stage)
            block_list=[]
            norm_list=[]
            for i in range(num_blocks_this_stage):
                block_list.append(
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                 dropout, activation, batch_first=True)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num-1:
                downsample_window = self.model_cfg.INPUT_LAYER.downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id+1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.output_shape = self.model_cfg.output_shape
        self.stage_num = stage_num
        self.set_info = set_info
        self.num_point_features = self.model_cfg.conv_out_channel

        self._reset_parameters()

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''
        voxel_info = self.input_layer(batch_dict)

        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]
        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s+1}'] for s in range(self.stage_num-1)]

        output = voxel_feat
        block_id = 0
        for stage_id in range(self.stage_num):
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                if self.use_torch_ckpt==False:
                    output = block(output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], \
                                block_id=block_id)
                else:
                    output = checkpoint(block, output, set_voxel_inds_list[stage_id], set_voxel_masks_list[stage_id], pos_embed_list[stage_id][i], block_id)
                output = residual_norm_layers[i](output + residual)
                block_id += 1
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats[stage_id].type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index[stage_id], pooling_index_in_pool[stage_id]] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)
                
                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = output
        batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class DSVTBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()

        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    def forward(
            self,
            src,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
    ):
        num_shifts = 2
        output = src
        # TODO: bug to be fixed, mismatch of pos_embed
        for i in range(num_shifts):
            set_id = i
            shift_id = block_id % 2
            pos_embed_id = i
            set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
            set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
            pos_embed = pos_embed_list[pos_embed_id]
            layer = self.encoder_list[i]
            output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)

        return output


class DSVT_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.win_attn = SetAttention(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self,src,set_voxel_inds,set_voxel_masks,pos=None):
        identity = src
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds)
        src = src + identity
        src = self.norm(src)

        return src

class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0):
        super().__init__()
        self.nhead = nhead
        if batch_first:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos=None, key_padding_mask=None, voxel_inds=None):
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
        Returns:
            src (Tensor[float]): Voxel features.
        '''
        set_features = src[voxel_inds]
        if pos is not None:
            set_pos = pos[voxel_inds]
        else:
            set_pos = None
        if pos is not None:
            query = set_features + set_pos
            key = set_features + set_pos
            value = set_features

        if key_padding_mask is not None:
            src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        else:
            src2 = self.self_attn(query, key, value)[0]

        # map voxel featurs from set space to voxel space: (set_num, set_size, C) --> (N, C)
        flatten_inds = voxel_inds.reshape(-1)
        unique_flatten_inds, inverse = torch.unique(flatten_inds, return_inverse=True)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique_flatten_inds.size(0)).scatter_(0, inverse, perm)
        src2 = src2.reshape(-1, self.d_model)[perm]

        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Stage_Reduction_Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=False)
        self.norm = nn.LayerNorm(output_channel)

    def forward(self, x):
        src = x
        src = self.norm(self.linear1(x))
        return src


class Stage_ReductionAtt_Block(nn.Module):
    def __init__(self, input_channel, pool_volume):
        super().__init__()
        self.pool_volume = pool_volume
        self.query_func = torch.nn.MaxPool1d(pool_volume)
        self.norm = nn.LayerNorm(input_channel)
        self.self_attn = nn.MultiheadAttention(input_channel, 8, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(pool_volume, input_channel))
        nn.init.normal_(self.pos_embedding, std=.01)

    def forward(self, x, key_padding_mask):
        # x: [voxel_num, c_dim, pool_volume]
        src = self.query_func(x).permute(0, 2, 1)  # voxel_num, 1, c_dim
        key = value = x.permute(0, 2, 1)
        key = key + self.pos_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query = src.clone()
        output = self.self_attn(query, key, value, key_padding_mask)[0]
        src = self.norm(output + src).squeeze(1)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_block_module(name):
    """Return an block module given a string"""
    if name == "DSVTBlock":
        return DSVTBlock
    raise RuntimeError(F"This Block not exist.")


class DSVTInputLayer(nn.Module):
    ''' 
    This class converts the output of vfe to dsvt input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        normalize_pos (bool): Whether to normalize coordinates in position embedding.
    '''
    def __init__(self, model_cfg):
        super().__init__()
        
        self.model_cfg = model_cfg 
        self.sparse_shape = self.model_cfg.sparse_shape
        self.window_shape = self.model_cfg.window_shape
        self.downsample_stride = self.model_cfg.downsample_stride
        self.d_model = self.model_cfg.d_model
        self.set_info = self.model_cfg.set_info
        self.stage_num = len(self.d_model)

        self.hybrid_factor = self.model_cfg.hybrid_factor
        self.window_shape = [[self.window_shape[s_id], [self.window_shape[s_id][coord_id] * self.hybrid_factor[coord_id] \
                                                for coord_id in range(3)]] for s_id in range(self.stage_num)]
        self.shift_list = self.model_cfg.shifts_list
        self.normalize_pos = self.model_cfg.normalize_pos
            
        self.num_shifts = [2,] * len(self.window_shape)

        self.sparse_shape_list = [self.sparse_shape]
        # compute sparse shapes for each stage
        for ds_stride in self.downsample_stride:
            last_sparse_shape = self.sparse_shape_list[-1]
            self.sparse_shape_list.append((ceil(last_sparse_shape[0]/ds_stride[0]), ceil(last_sparse_shape[1]/ds_stride[1]), ceil(last_sparse_shape[2]/ds_stride[2])))
        
        # position embedding layers
        self.posembed_layers = nn.ModuleList()
        for i in range(len(self.set_info)):
            input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2
            stage_posembed_layers = nn.ModuleList()
            for j in range(self.set_info[i][1]):
                block_posembed_layers = nn.ModuleList()
                for s in range(self.num_shifts[i]):
                    block_posembed_layers.append(PositionEmbeddingLearned(input_dim, self.d_model[i]))
                stage_posembed_layers.append(block_posembed_layers)
            self.posembed_layers.append(stage_posembed_layers)
       
    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region index used in pooling operation between stage_{i-1} and stage_{i}
                    with shape (N_{i-1}). 
                - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner region with shape (N_{i-1}). Combined with pooling_mapping_index_stage{i}, 
                    we can map each voxel in satge_{i-1} to pooling_preholder_feats_stage{i}, which are input of downsample operation.
                - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder features initial with value 0. 
                    Shape of (N_{i}, downsample_stride[i-1].prob(), d_moel[i-1]), where prob() returns the product of all elements.
                - ...
        '''
        voxel_feats = batch_dict['voxel_features']
        voxel_coors = batch_dict['voxel_coords'].long()

        voxel_info = {}
        voxel_info['voxel_feats_stage0'] = voxel_feats.clone()
        voxel_info['voxel_coors_stage0'] = voxel_coors.clone()
        
        for stage_id in range(self.stage_num):
            # window partition of corrsponding stage-map
            voxel_info = self.window_partition(voxel_info, stage_id)
            # generate set id of corrsponding stage-map
            voxel_info = self.get_set(voxel_info, stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    voxel_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                    self.get_pos_embed(voxel_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
            
            # compute pooling information
            if stage_id < self.stage_num - 1:
                voxel_info = self.subm_pooling(voxel_info, stage_id)
        
        return voxel_info
    
    @torch.no_grad()
    def subm_pooling(self, voxel_info, stage_id):
        # x,y,z stride
        cur_stage_downsample = self.downsample_stride[stage_id]
        # batch_win_coords is from 1 of x, y
        batch_win_inds, _, index_in_win, batch_win_coors = get_pooling_index(voxel_info[f'voxel_coors_stage{stage_id}'], self.sparse_shape_list[stage_id], cur_stage_downsample)
        # compute pooling mapping index
        unique_batch_win_inds, contiguous_batch_win_inds = torch.unique(batch_win_inds, return_inverse=True)
        voxel_info[f'pooling_mapping_index_stage{stage_id+1}'] = contiguous_batch_win_inds
        
        # generate empty placeholder features
        placeholder_prepool_feats = voxel_info[f'voxel_feats_stage0'].new_zeros((len(unique_batch_win_inds), 
                    torch.prod(torch.IntTensor(cur_stage_downsample)).item(), self.d_model[stage_id]))
        voxel_info[f'pooling_index_in_pool_stage{stage_id+1}'] = index_in_win
        voxel_info[f'pooling_preholder_feats_stage{stage_id+1}'] = placeholder_prepool_feats

        # compute pooling coordinates
        unique, inverse = unique_batch_win_inds.clone(), contiguous_batch_win_inds.clone()
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        pool_coors = batch_win_coors[perm]

        voxel_info[f'voxel_coors_stage{stage_id+1}'] = pool_coors
        
        return voxel_info
    
    def get_set(self, voxel_info, stage_id):
        '''
        This is one of the core operation of DSVT. 
        Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
        To make it clear and easy to follow, we do not use loop to process two shifts.
        Args:
            voxel_info (dict): 
                The dict contains the following keys
                - batch_win_inds_s{i} (Tensor[float]): Windows indexs of each voxel with shape (N), computed by 'window_partition'.
                - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner window of each voxel with shape (N, 3), computed by 'window_partition'.
                    Each row is (z, y, x). 
                - ...
        
        Returns:
            See from 'forward' function.
        '''
        batch_win_inds_shift0 = voxel_info[f'batch_win_inds_stage{stage_id}_shift0']
        coors_in_win_shift0 = voxel_info[f'coors_in_win_stage{stage_id}_shift0']
        set_voxel_inds_shift0 = self.get_set_single_shift(batch_win_inds_shift0, stage_id, shift_id=0, coors_in_win=coors_in_win_shift0)
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0  
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s0 = torch.roll(set_voxel_inds_shift0.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s0[ :, :, 0] = -1
        set_voxel_mask_s0 = (set_voxel_inds_shift0 == prefix_set_voxel_inds_s0)
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0

        batch_win_inds_shift1 = voxel_info[f'batch_win_inds_stage{stage_id}_shift1']
        coors_in_win_shift1 = voxel_info[f'coors_in_win_stage{stage_id}_shift1']
        set_voxel_inds_shift1 = self.get_set_single_shift(batch_win_inds_shift1, stage_id, shift_id=1, coors_in_win=coors_in_win_shift1)
        voxel_info[f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1  
        # compute key masks, voxel duplication must happen continuously
        prefix_set_voxel_inds_s1 = torch.roll(set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
        prefix_set_voxel_inds_s1[ :, :, 0] = -1
        set_voxel_mask_s1 = (set_voxel_inds_shift1 == prefix_set_voxel_inds_s1)
        voxel_info[f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1

        return voxel_info
    
    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        device = batch_win_inds.device
        # the number of voxels assigned to a set
        voxel_num_set = self.set_info[stage_id][0]
        # max number of voxels in a window
        max_voxel = self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2]
        # get unique set indexs
        contiguous_win_inds = torch.unique(batch_win_inds, return_inverse=True)[1]
        voxelnum_per_win = torch.bincount(contiguous_win_inds)
        win_num = voxelnum_per_win.shape[0]
        setnum_per_win_float = voxelnum_per_win / voxel_num_set
        setnum_per_win = torch.ceil(setnum_per_win_float).long()
        set_win_inds, set_inds_in_win = get_continous_inds(setnum_per_win)
        
        # compution of Eq.3 in 'DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets' - https://arxiv.org/abs/2301.06051, 
        # for each window, we can get voxel indexs belong to different sets.
        offset_idx = set_inds_in_win[:,None].repeat(1, voxel_num_set) * voxel_num_set
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * voxelnum_per_win[set_win_inds][:,None]
        base_select_idx = base_select_idx.double() / (setnum_per_win[set_win_inds] * voxel_num_set)[:,None].double()
        base_select_idx = torch.floor(base_select_idx)
        # obtain unique indexs in whole space
        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel
           
        # this function will return unordered inner window indexs of each voxel
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
        _, order1 = torch.sort(global_voxel_inds)

        # get y-axis partition results
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
                coors_in_win[:,1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + \
                coors_in_win[:,2] * self.window_shape[stage_id][shift_id][2] + \
                coors_in_win[:,0]
        _, order2 = torch.sort(global_voxel_inds_sorty)
        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[order1]) # get y-axis ordered inner window indexs of each voxel
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * contiguous_win_inds
        voxel_inds_padding_sorty = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device)
        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]

        # get x-axis partition results
        global_voxel_inds_sortx = contiguous_win_inds * max_voxel + \
                coors_in_win[:,2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + \
                coors_in_win[:,1] * self.window_shape[stage_id][shift_id][2] + \
                coors_in_win[:,0]
        _, order2 = torch.sort(global_voxel_inds_sortx)
        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(dim=0,index=order2, src=inner_voxel_inds[order1]) # get x-axis ordered inner window indexs of each voxel
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device)
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]

        all_set_voxel_inds = torch.stack((set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)
        return all_set_voxel_inds

    @torch.no_grad()
    def window_partition(self, voxel_info, stage_id):
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'voxel_coors_stage{stage_id}'], 
                                                        self.sparse_shape_list[stage_id], self.window_shape[stage_id][i], i == 1, self.shift_list[stage_id][i])
                                            
            voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win
        
        return voxel_info

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id]
       
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location)

        return pos_embed

