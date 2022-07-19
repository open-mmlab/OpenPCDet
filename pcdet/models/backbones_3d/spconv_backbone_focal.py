from functools import partial

import torch
from pcdet.utils.spconv_utils import spconv
import torch.nn as nn

from .focal_sparse_conv.focal_sparse_conv import FocalSparseConv
from .focal_sparse_conv.SemanticSeg.pyramid_ffn import PyramidFeat2D


class objDict:
    @staticmethod
    def to_object(obj: object, **data):
        obj.__dict__.update(data)

class ConfigDict:
    def __init__(self, name):
        self.name = name
    def __getitem__(self, item):
        return getattr(self, item)


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        loss = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, batch_dict, _loss = module(input, batch_dict)
                loss += _loss
            else:
                input = module(input)
        return input, batch_dict, loss


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(True),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class VoxelBackBone8xFocal(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(True),
        )

        block = post_act_block

        use_img = model_cfg.get('USE_IMG', False)
        topk = model_cfg.get('TOPK', True)
        threshold = model_cfg.get('THRESHOLD', 0.5)
        kernel_size = model_cfg.get('KERNEL_SIZE', 3)
        mask_multi = model_cfg.get('MASK_MULTI', False)
        skip_mask_kernel = model_cfg.get('SKIP_MASK_KERNEL', False)
        skip_mask_kernel_image =  model_cfg.get('SKIP_MASK_KERNEL_IMG', False)
        enlarge_voxel_channels = model_cfg.get('ENLARGE_VOXEL_CHANNELS', -1)
        img_pretrain = model_cfg.get('IMG_PRETRAIN', "../checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth")

        if use_img:
            model_cfg_seg=dict(
                name='SemDeepLabV3',
                backbone='ResNet50',
                num_class=21, # pretrained on COCO
                args={"feat_extract_layer": ["layer1"],
                    "pretrained_path": img_pretrain},
                channel_reduce={
                    "in_channels": [256],
                    "out_channels": [16],
                    "kernel_size": [1],
                    "stride": [1],
                    "bias": [False]
                }
            )
            cfg_dict = ConfigDict('SemDeepLabV3')
            objDict.to_object(cfg_dict, **model_cfg_seg)
            self.semseg = PyramidFeat2D(optimize=True, model_cfg=cfg_dict)

            self.conv_focal_multimodal = FocalSparseConv(16, 16, image_channel=model_cfg_seg['channel_reduce']['out_channels'][0],
                                        topk=topk, threshold=threshold, use_img=True, skip_mask_kernel=skip_mask_kernel_image,
                                        voxel_stride=1, norm_fn=norm_fn, indice_key='spconv_focal_multimodal')

        special_spconv_fn = partial(FocalSparseConv, mask_multi=mask_multi, enlarge_voxel_channels=enlarge_voxel_channels, 
                                    topk=topk, threshold=threshold, kernel_size=kernel_size, padding=kernel_size//2, 
                                    skip_mask_kernel=skip_mask_kernel)
        self.use_img = use_img

        self.conv1 = SparseSequentialBatchdict(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            special_spconv_fn(16, 16, voxel_stride=1, norm_fn=norm_fn, indice_key='focal1'),
        )

        self.conv2 =SparseSequentialBatchdict(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            special_spconv_fn(32, 32, voxel_stride=2, norm_fn=norm_fn, indice_key='focal2'),
        )

        self.conv3 = SparseSequentialBatchdict(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            special_spconv_fn(64, 64, voxel_stride=4, norm_fn=norm_fn, indice_key='focal3'),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(True),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        
        self.forward_ret_dict = {}
        
    def get_loss(self, tb_dict=None):
        loss = self.forward_ret_dict['loss_box_of_pts']
        if tb_dict is None:
            tb_dict = {}
        tb_dict['loss_box_of_pts'] = loss.item()
        return loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        loss_img = 0

        x = self.conv_input(input_sp_tensor)
        x_conv1, batch_dict, loss1 = self.conv1(x, batch_dict)

        if self.use_img:
            x_image = self.semseg(batch_dict['images'])['layer1_feat2d']
            x_conv1, batch_dict, loss_img = self.conv_focal_multimodal(x_conv1, batch_dict, x_image)

        x_conv2, batch_dict, loss2 = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict, loss3 = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict, loss4 = self.conv4(x_conv3, batch_dict)

        self.forward_ret_dict['loss_box_of_pts'] = loss1 + loss2 + loss3 + loss4 + loss_img
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
