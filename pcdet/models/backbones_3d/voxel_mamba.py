import torch
import torch.nn as nn

import math
from functools import partial
from mamba_ssm.models.mixer_seq_simple import create_block
from ..model_utils.voxel_mamba_utils import get_hilbert_index_3d_mamba_lite

# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import post_act_block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Voxel_Mamba(nn.Module):
    '''Group-free Voxel Mamba Backbone.
    '''
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        # self.hilbert_input_layer = HilbertCurveInputLayer(self.model_cfg.INPUT_LAYER)

        num_stage = self.model_cfg.num_stage
        self.num_stage = num_stage
        self.d_model = self.model_cfg.d_model
        self.rms_norm = self.model_cfg.rms_norm
        self.norm_epsilon = self.model_cfg.norm_epsilon
        self.fused_add_norm = self.model_cfg.fused_add_norm
        self.device = self.model_cfg.device
        self.residual_in_fp32 = self.model_cfg.residual_in_fp32
        self.extra_down = self.model_cfg.extra_down
        self.dtype = torch.float32
        initializer_cfg = None
        
        # for downsampling
        self.down_kernel_size = self.model_cfg.down_kernel_size
        self.down_stride = self.model_cfg.down_stride
        self.num_down = self.model_cfg.num_down
        self.down_resolution = self.model_cfg.down_resolution
        self.downsample_lvl = self.model_cfg.downsample_lvl
        self.norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # Build Hilbert tempalte 
        self.curve_template = {}
        self.hilbert_spatial_size = {}
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank9, 9)
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank8, 8)
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank7, 7)

        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        block_list = []
        for i, num_s in enumerate(num_stage):
            for ns in range(num_s):
                block_list.append(
                    DSB(self.d_model, ssm_cfg=None, norm_epsilon=self.norm_epsilon, rms_norm=self.rms_norm, 
                        down_kernel_size=self.down_kernel_size[i], down_stride=self.down_stride[i], num_down=self.num_down[i], 
                        norm_fn=self.norm_fn, indice_key=f'stem{i}_layer{ns}', sparse_shape=self.sparse_shape, hilbert_config=self.model_cfg.INPUT_LAYER,
                        downsample_lvl=self.downsample_lvl[i],
                        down_resolution=self.down_resolution[i], residual_in_fp32=True, fused_add_norm=self.fused_add_norm, 
                        device=self.device, dtype=self.dtype)
                )
        self.block_list = nn.ModuleList(block_list)

        downZ_list = []
        for i in range(len(num_stage)):
            downZ_list.append(
                spconv.SparseSequential(
                spconv.SparseConv3d(self.d_model, self.d_model, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key=f'downz_{i}'),
                self.norm_fn(self.d_model),
                nn.ReLU(),)
            )
        self.downZ_list = nn.ModuleList(downZ_list)

        self.conv_out = spconv.SparseSequential(
                spconv.SparseConv3d(self.d_model, self.d_model, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False, indice_key=f'final_conv_out'),
                self.norm_fn(self.d_model),
                nn.ReLU(),)
        
        self.pos_embed =  nn.Sequential(
                    nn.Linear(9, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_model, self.d_model),
                    )

        self._reset_parameters()
        self.apply(
            partial(
                _init_weights,
                n_layer=sum(num_stage),
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.output_shape = self.model_cfg.output_shape
        self.num_point_features = self.model_cfg.conv_out_channel

    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]

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

        # with self.timer.timing('3d_backbone'):
        debug = False
        batch_size = batch_dict['voxel_coords'][:, 0].max().item() + 1
        feat_3d = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        with torch.no_grad():
            for name, _ in self.curve_template.items():
                self.curve_template[name] = self.curve_template[name].to(voxel_coords.device)
        
        down_sparse_shape = self.sparse_shape
        for i, block in enumerate(self.block_list):

            feat_3d, voxel_coords = block(feat_3d, voxel_coords, batch_size, down_sparse_shape, self.curve_template, self.hilbert_spatial_size, self.pos_embed, i, debug)
            
            if (i > 0) and (i % 2 == 1):
                xd = spconv.SparseConvTensor(
                    features=feat_3d,
                    indices=voxel_coords.int(),
                    spatial_shape=down_sparse_shape,
                    batch_size=batch_size
                )

                if i == self.extra_down:
                    xd = self.conv_out(xd)

                xd = self.downZ_list[i//2](xd)

                feat_3d = xd.features
                voxel_coords = xd.indices
                down_sparse_shape = xd.spatial_shape


        if self.training and torch.isnan(feat_3d).any().item():
            replacement_value = 0.0
            feat_3d = torch.where(torch.isnan(feat_3d), replacement_value, feat_3d) 

        batch_dict['voxel_coords'] = voxel_coords
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = feat_3d
      
        return batch_dict

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class DSB(nn.Module):
    ''' Dual-scale State Space Models Block
    '''

    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm,
                 down_kernel_size,
                 down_stride,
                 num_down,
                 norm_fn,
                 indice_key,
                 sparse_shape,
                 hilbert_config,
                 downsample_lvl,
                 down_resolution=True,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None):
        super().__init__()

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}

        # mamba layer
        mamba_encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            **factory_kwargs,
        )

        mamba_encoder_2 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_1, mamba_encoder_2])

        # downsampling operation #
        self.conv_encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.conv_encoder.append(
                DownSp(d_model, down_kernel_size[idx], down_stride[idx], num_down[idx], norm_fn, f"{indice_key}_{idx}"))
        
        # upsampling operation #
        downsample_times = len(down_stride[1:])
        self.conv_decoder = nn.ModuleList()
        self.conv_decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]):
            if down_resolution:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                        indice_key=f'spconv_{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
            else:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='subm',
                        indice_key=f'{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
        
        self.sparse_shape = sparse_shape
        self.downsample_lvl = downsample_lvl

        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.norm = norm_cls(d_model)
        self.norm_back = norm_cls(d_model)

    def forward(
        self,
        voxel_features,
        voxel_coords,
        batch_size,
        curt_spatial_shape,
        curve_template,
        hilbert_spatial_size,
        pos_embed,
        num_stage,
        debug=False,
        ):

        mamba_layer1 = self.mamba_encoder_list[0]
        mamba_layer2 = self.mamba_encoder_list[1]
        
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=curt_spatial_shape,
            batch_size=batch_size
        )

        features = []
        for conv in self.conv_encoder:
            x = conv(x)
            features.append(x)
        
        x_s1 = features[0]
        x_s2 = features[1]
        feats_s2 = features[1].features
        coords_s2 = features[1].indices
        feats_s1 = features[0].features
        coords_s1 = features[0].indices

        clvl_cruve_template_s1 = curve_template['curve_template_rank9']
        clvl_hilbert_spatial_size_s1 = hilbert_spatial_size['curve_template_rank9']
        index_info_s1 = get_hilbert_index_3d_mamba_lite(clvl_cruve_template_s1, coords_s1, batch_size, x_s1.spatial_shape[0], \
                                                        clvl_hilbert_spatial_size_s1, shift=(num_stage, num_stage, num_stage))
        inds_curt_to_next_s1 = index_info_s1['inds_curt_to_next']
        inds_next_to_curt_s1 = index_info_s1['inds_next_to_curt']

        clvl_cruve_template_s2 = curve_template[self.downsample_lvl]
        clvl_hilbert_spatial_size_s2 = hilbert_spatial_size[self.downsample_lvl]
        index_info_s2 = get_hilbert_index_3d_mamba_lite(clvl_cruve_template_s2, coords_s2, batch_size, x_s2.spatial_shape[0], 
                                                        clvl_hilbert_spatial_size_s2, shift=(num_stage, num_stage, num_stage))
        inds_curt_to_next_s2 = index_info_s2['inds_curt_to_next']
        inds_next_to_curt_s2 = index_info_s2['inds_next_to_curt']

        new_features = []
        # Low Resolution
        out_feat_3d_s2 = torch.zeros_like(feats_s2)
        out_feat_3d_s1 = torch.zeros_like(feats_s1)

        # Pos Embedding
        pos_embed_coords_s2 = torch.zeros([coords_s2.shape[0], 9], device=coords_s2.device, dtype=torch.float32)
        pos_embed_coords_s2[:, 0] = coords_s2[:, 1] / x_s2.spatial_shape[0]
        pos_embed_coords_s2[:, 1:3] = (coords_s2[:, 2:] // 12) / (x_s2.spatial_shape[1]//12 + 1)
        pos_embed_coords_s2[:, 3:5] = (coords_s2[:, 2:] % 12) / 12.0
        pos_embed_coords_s2[:, 5:7] = ((coords_s2[:, 2:] + 6) // 12) / (x_s2.spatial_shape[1]//12 + 1)
        pos_embed_coords_s2[:, 7:9] = ((coords_s2[:, 2:] + 6) % 12) / 12.0
        pos_embed_s2 = pos_embed(pos_embed_coords_s2.float())

        feats_s2 = feats_s2 + pos_embed_s2

        # Borward SSMs
        for i in range(batch_size):
            b_mask_m2 = coords_s2[:, 0] == i
            feat_m2 = feats_s2[b_mask_m2][inds_curt_to_next_s2[i]][None]
            out_feat_m2 = mamba_layer1(feat_m2, None)
            out_feat_3d_s2[b_mask_m2] = (out_feat_m2[0]).squeeze(0)[inds_next_to_curt_s2[i]]

        x_s2 = replace_feature(x_s2, self.norm(out_feat_3d_s2))

        # Fackward SSMs
        pos_embed_coords_s1 = torch.zeros([coords_s1.shape[0], 9], device=coords_s1.device, dtype=torch.float32)
        pos_embed_coords_s1[:, 0] = coords_s1[:, 1] / x_s1.spatial_shape[0]
        pos_embed_coords_s1[:, 1:3] = (coords_s1[:, 2:] // 12) / (x_s1.spatial_shape[1]//12 + 1)
        pos_embed_coords_s1[:, 3:5] = (coords_s1[:, 2:] % 12) / 12.0
        pos_embed_coords_s1[:, 5:7] = ((coords_s1[:, 2:] + 6) // 12) / (x_s1.spatial_shape[1]//12 + 1)
        pos_embed_coords_s1[:, 7:9] = ((coords_s1[:, 2:] + 6) % 12) / 12.0
        pos_embed_s1 = pos_embed(pos_embed_coords_s1.float())

        feats_s1 = feats_s1 + pos_embed_s1
        for i in range(batch_size):
            b_mask_m1 = coords_s1[:, 0] == i
            feat_m1 = feats_s1[b_mask_m1][inds_curt_to_next_s1[i]][None]
            feat_back = feat_m1.flip(1)
            out_feat_back = mamba_layer2(feat_back, None)
            out_feat_3d_s1[b_mask_m1] = (out_feat_back[0]).squeeze(0).flip(0)[inds_next_to_curt_s1[i]]

        x_s1 = replace_feature(x_s1, self.norm_back(out_feat_3d_s1))

        # new_features.append(features[0])
        new_features.append(x_s1)
        new_features.append(x_s2)

        x = x_s2

        for deconv, norm, up_x in zip(self.conv_decoder, self.conv_decoder_norm, new_features[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, x.features + up_x.features + features[0].features)
            x = replace_feature(x, norm(x.features))

        return x.features, x.indices


#####  downsampling operation  #####

class Sparse1ConvBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(Sparse1ConvBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out
    

class DownSp(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_down, norm_fn, indice_key):
        super(DownSp, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_down):
            block_list.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)
