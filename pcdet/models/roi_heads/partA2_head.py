import numpy as np
import spconv
import torch
import torch.nn as nn

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from .roi_head_template import RoIHeadTemplate


class PartA2FCHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        block = self.post_act_block

        c0 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES // 2
        self.conv_part = spconv.SparseSequential(
            block(4, 64, 3, padding=1, indice_key='rcnn_subm1'),
            block(64, c0, 3, padding=1, indice_key='rcnn_subm1_1'),
        )
        self.conv_rpn = spconv.SparseSequential(
            block(input_channels, 64, 3, padding=1, indice_key='rcnn_subm2'),
            block(64, c0, 3, padding=1, indice_key='rcnn_subm1_2'),
        )

        shared_fc_list = []
        pool_size = self.model_cfg.ROI_AWARE_POOL.POOL_SIZE
        pre_channel = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES * pool_size * pool_size * pool_size
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=self.model_cfg.ROI_AWARE_POOL.POOL_SIZE,
            max_pts_each_voxel=self.model_cfg.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL
        )
        self.init_weights(weight_init='xavier')

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
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    def roiaware_pool(self, batch_dict):
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
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['point_coords'][:, 0]
        point_coords = batch_dict['point_coords'][:, 1:4]
        point_features = batch_dict['point_features']
        part_features = torch.cat((
            batch_dict['point_part_offset'] if not self.model_cfg.get('DISABLE_PART', False) else point_coords,
            batch_dict['point_cls_scores'].view(-1, 1).detach()
        ), dim=1)
        part_features[part_features[:, -1] < self.model_cfg.SEG_MASK_SCORE_THRESH, 0:3] = 0

        rois = batch_dict['rois']

        pooled_part_features_list, pooled_rpn_features_list = [], []

        for bs_idx in range(batch_size):
            bs_mask = (batch_idx == bs_idx)
            cur_point_coords = point_coords[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = point_features[bs_mask]
            cur_roi = rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            pooled_part_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_part_features, pool_method='avg'
            )  # (N, out_x, out_y, out_z, 4)
            pooled_rpn_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_rpn_features, pool_method='max'
            )  # (N, out_x, out_y, out_z, C)

            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)

        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)  # (B * N, out_x, out_y, out_z, 4)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)  # (B * N, out_x, out_y, out_z, C)

        return pooled_part_features, pooled_rpn_features

    @staticmethod
    def fake_sparse_idx(sparse_idx, batch_size_rcnn):
        print('Warning: Sparse_Idx_Shape(%s) \r' % (str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_part_features, pooled_rpn_features = self.roiaware_pool(batch_dict)
        batch_size_rcnn = pooled_part_features.shape[0]  # (B * N, out_x, out_y, out_z, 4)

        # transform to sparse tensors
        sparse_shape = np.array(pooled_part_features.shape[1:4], dtype=np.int32)
        sparse_idx = pooled_part_features.sum(dim=-1).nonzero()  # (non_empty_num, 4) ==> [bs_idx, x_idx, y_idx, z_idx]
        if sparse_idx.shape[0] < 3:
            sparse_idx = self.fake_sparse_idx(sparse_idx, batch_size_rcnn)
            if self.training:
                # these are invalid samples
                targets_dict['rcnn_cls_labels'].fill_(-1)
                targets_dict['reg_valid_mask'].fill_(-1)

        part_features = pooled_part_features[sparse_idx[:, 0], sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]]
        rpn_features = pooled_rpn_features[sparse_idx[:, 0], sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]]
        coords = sparse_idx.int()
        part_features = spconv.SparseConvTensor(part_features, coords, sparse_shape, batch_size_rcnn)
        rpn_features = spconv.SparseConvTensor(rpn_features, coords, sparse_shape, batch_size_rcnn)

        # forward rcnn network
        x_part = self.conv_part(part_features)
        x_rpn = self.conv_rpn(rpn_features)

        merged_feature = torch.cat((x_rpn.features, x_part.features), dim=1)  # (N, C)
        shared_feature = spconv.SparseConvTensor(merged_feature, coords, sparse_shape, batch_size_rcnn)
        shared_feature = shared_feature.dense().view(batch_size_rcnn, -1, 1)

        shared_feature = self.shared_fc_layer(shared_feature)

        rcnn_cls = self.cls_layers(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        return batch_dict
