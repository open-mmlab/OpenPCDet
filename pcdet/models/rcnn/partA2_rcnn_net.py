import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv
from ..model_utils import pytorch_utils as pt_utils

from ...config import cfg

from ..model_utils.proposal_target_layer import proposal_target_layer
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils, box_coder_utils


class RCNNHead(nn.Module):
    def __init__(self, rcnn_target_config):
        super().__init__()
        self.forward_ret_dict = None
        self.rcnn_target_config = rcnn_target_config
        self.box_coder = getattr(box_coder_utils, rcnn_target_config.BOX_CODER)()
        losses_cfg = cfg.MODEL.LOSSES
        code_weights = losses_cfg.LOSS_WEIGHTS['code_weights']
        self.reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=code_weights)

    def assign_targets(self, batch_size, rcnn_dict):
        with torch.no_grad():
            targets_dict = proposal_target_layer(rcnn_dict, roi_sampler_cfg=self.rcnn_target_config)

        rois = targets_dict['rois']  # (B, N, 7)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry
        for k in range(batch_size):
            # transfer LiDAR coords to local coords
            gt_of_rois[k] = common_utils.rotate_pc_along_z_torch(
                gt_of_rois[k].unsqueeze(dim=1), -(roi_ry[k] + np.pi / 2)
            ).squeeze(dim=1)

        # flip orientation if rois have opposite orientation
        ry_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = ry_label > np.pi
        ry_label[flag] = ry_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        ry_label = torch.clamp(ry_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = ry_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_loss(self, forward_ret_dict=None):
        loss_cfgs = cfg.MODEL.LOSSES
        LOSS_WEIGHTS = loss_cfgs.LOSS_WEIGHTS

        forward_ret_dict = self.forward_ret_dict if forward_ret_dict is None else forward_ret_dict
        code_size = self.box_coder.code_size
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].float().view(-1)
        reg_valid_mask = forward_ret_dict['reg_valid_mask']
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = rcnn_cls_labels.shape[0]

        rcnn_loss = 0
        if loss_cfgs.RCNN_CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels, reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = rcnn_loss_cls * LOSS_WEIGHTS['rcnn_cls_weight']
        else:
            raise NotImplementedError

        rcnn_loss += rcnn_loss_cls
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum == 0:
            # To be consistent with DistributedDataParallel
            # Faked a rcnn_loss to make gradient of regression branch be zero
            temp_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            faked_reg_target = temp_rcnn_reg.detach()
            rcnn_loss_reg = self.reg_loss_func(temp_rcnn_reg, faked_reg_target)  # [N, M]
            rcnn_loss_reg = rcnn_loss_reg.sum() / 1.0
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        else:
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

            if loss_cfgs.RCNN_REG_LOSS == 'smooth-l1':
                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                reg_targets = self.box_coder.encode_torch(
                    gt_boxes3d_ct.view(rcnn_batch_size, code_size)[fg_mask], rois_anchor[fg_mask]
                )
                rcnn_loss_reg = self.reg_loss_func(
                    rcnn_reg.view(rcnn_batch_size, -1)[fg_mask].unsqueeze(dim=0),
                    reg_targets.unsqueeze(dim=0)
                )  # [N, M]
                rcnn_loss_reg = rcnn_loss_reg.sum() / max(fg_sum, 0)
                rcnn_loss_reg = rcnn_loss_reg * LOSS_WEIGHTS['rcnn_reg_weight']
                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

                if loss_cfgs.CORNER_LOSS_REGULARIZATION:
                    # TODO: NEED to BE CHECK
                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.box_coder.decode_torch(
                        fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                    ).view(-1, code_size)

                    rcnn_boxes3d = common_utils.rotate_pc_along_z_torch(
                        rcnn_boxes3d.unsqueeze(dim=1), (roi_ry + np.pi / 2)
                    ).squeeze(dim=1)
                    rcnn_boxes3d[:, 0:3] += roi_xyz

                    loss_corner = loss_utils.get_corner_loss_lidar(
                        rcnn_boxes3d[:, 0:7],
                        gt_of_rois_src[fg_mask][:, 0:7]
                    )
                    loss_corner = loss_corner.mean()
                    loss_corner = loss_corner * LOSS_WEIGHTS['rcnn_corner_weight']

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError

        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict


class SpConvRCNN(RCNNHead):
    def __init__(self, num_point_features, rcnn_cfg, **kwargs):
        super().__init__(rcnn_target_config=cfg.MODEL.RCNN.TARGET_CONFIG)

        self.SA_modules = nn.ModuleList()
        block = self.post_act_block

        self.conv_part = spconv.SparseSequential(
            block(4, 64, 3, padding=1, indice_key='rcnn_subm1'),
            block(64, 64, 3, padding=1, indice_key='rcnn_subm1_1'),
        )
        self.conv_rpn = spconv.SparseSequential(
            block(num_point_features, 64, 3, padding=1, indice_key='rcnn_subm2'),
            block(64, 64, 3, padding=1, indice_key='rcnn_subm1_2'),
        )
        self.conv_down = spconv.SparseSequential(
            # [14, 14, 14] -> [7, 7, 7]
            block(128, 128, 3, padding=1, indice_key='rcnn_subm2'),
            block(128, 128, 3, padding=1, indice_key='rcnn_subm2'),
            spconv.SparseMaxPool3d(kernel_size=2, stride=2),
            block(128, 128, 3, padding=1, indice_key='rcnn_subm3'),
            block(128, rcnn_cfg.SHARED_FC[0], 3, padding=1, indice_key='rcnn_subm3'),
        )

        shared_fc_list = []
        pool_size = rcnn_cfg.ROI_AWARE_POOL_SIZE // 2
        pre_channel = rcnn_cfg.SHARED_FC[0] * pool_size * pool_size * pool_size
        for k in range(1, rcnn_cfg.SHARED_FC.__len__()):
            shared_fc_list.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.SHARED_FC[k], bn=True))
            pre_channel = rcnn_cfg.SHARED_FC[k]

            if k != rcnn_cfg.SHARED_FC.__len__() - 1 and rcnn_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(rcnn_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        channel_in = rcnn_cfg.SHARED_FC[-1]
        # Classification layer
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.CLS_FC[k], bn=True))
            pre_channel = rcnn_cfg.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        # Regression layer
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.REG_FC[k], bn=True))
            pre_channel = rcnn_cfg.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, self.box_coder.code_size, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=rcnn_cfg.ROI_AWARE_POOL_SIZE, max_pts_each_voxel=128
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
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

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

    def roiaware_pool(self, batch_rois, rcnn_dict):
        """
        :param batch_rois: (B, N, 7 + ?) [x, y, z, w, l, h, rz] in LiDAR coords
        :param rcnn_dict:
        :return:
        """
        voxel_centers = rcnn_dict['voxel_centers']  # (npoints, 3)
        rpn_features = rcnn_dict['rpn_seg_features']  # (npoints, C)
        coords = rcnn_dict['coordinates']  # (npoints, 4)

        rpn_seg_score = rcnn_dict['rpn_seg_scores'].detach()  # (npoints)
        rpn_seg_mask = (rpn_seg_score > cfg.MODEL.RPN.BACKBONE.SEG_MASK_SCORE_THRESH)
        rpn_part_offsets = rcnn_dict['rpn_part_offsets'].clone().detach()
        rpn_part_offsets[rpn_seg_mask == 0] = 0
        part_features = torch.cat((rpn_part_offsets, rpn_seg_score.view(-1, 1)), dim=1)  # (npoints, 4)

        batch_size = batch_rois.shape[0]
        pooled_part_features_list, pooled_rpn_features_list = [], []

        for bs_idx in range(batch_size):
            bs_mask = (coords[:, 0] == bs_idx)
            cur_voxel_centers = voxel_centers[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = rpn_features[bs_mask]
            cur_roi = batch_rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            pooled_part_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_voxel_centers, cur_part_features, pool_method='avg'
            )  # (N, out_x, out_y, out_z, 4)
            pooled_rpn_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_voxel_centers, cur_rpn_features, pool_method='max'
            )  # (N, out_x, out_y, out_z, C)

            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)

        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)  # (B * N, out_x, out_y, out_z, 4)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)  # (B * N, out_x, out_y, out_z, C)

        return pooled_part_features, pooled_rpn_features

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def fake_sparse_idx(self, sparse_idx, batch_size_rcnn):
        print('Warning: GPU_%d: Sparse_Idx_Shape(%s) \r' % (cfg.LOCAL_RANK, str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def forward(self, rcnn_dict):
        """
        :param input_data: input dict
        :return:
        """
        rois = rcnn_dict['rois']
        batch_size = rois.shape[0]
        if self.training:
            targets_dict = self.assign_targets(batch_size, rcnn_dict)
            rois = targets_dict['rois']  # (B, N, 7)
            rcnn_dict['roi_raw_scores'] = targets_dict['roi_raw_scores']
            rcnn_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_part_features, pooled_rpn_features = self.roiaware_pool(rois, rcnn_dict)
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

        x = self.conv_down(shared_feature)  #

        shared_feature = x.dense().view(batch_size_rcnn, -1, 1)
        shared_feature = self.shared_fc_layer(shared_feature)
        rcnn_cls = self.cls_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        ret_dict = {
            'rcnn_cls': rcnn_cls,
            'rcnn_reg': rcnn_reg,
            'rois': rois,
            'roi_raw_scores': rcnn_dict['roi_raw_scores'],
            'roi_labels': rcnn_dict['roi_labels']
        }

        if self.training:
            ret_dict.update(targets_dict)

        self.forward_ret_dict = ret_dict
        return ret_dict


class FCRCNN(RCNNHead):
    def __init__(self, num_point_features, rcnn_cfg, **kwargs):
        super().__init__(rcnn_target_config=cfg.MODEL.RCNN.TARGET_CONFIG)

        self.SA_modules = nn.ModuleList()
        block = self.post_act_block

        c0 = rcnn_cfg.SHARED_FC[0] // 2
        self.conv_part = spconv.SparseSequential(
            block(4, 64, 3, padding=1, indice_key='rcnn_subm1'),
            block(64, c0, 3, padding=1, indice_key='rcnn_subm1_1'),
        )
        self.conv_rpn = spconv.SparseSequential(
            block(num_point_features, 64, 3, padding=1, indice_key='rcnn_subm2'),
            block(64, c0, 3, padding=1, indice_key='rcnn_subm1_2'),
        )

        shared_fc_list = []
        pool_size = rcnn_cfg.ROI_AWARE_POOL_SIZE
        pre_channel = rcnn_cfg.SHARED_FC[0] * pool_size * pool_size * pool_size
        for k in range(1, rcnn_cfg.SHARED_FC.__len__()):
            shared_fc_list.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.SHARED_FC[k], bn=True))
            pre_channel = rcnn_cfg.SHARED_FC[k]

            if k != rcnn_cfg.SHARED_FC.__len__() - 1 and rcnn_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(rcnn_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        channel_in = rcnn_cfg.SHARED_FC[-1]
        # Classification layer
        cls_channel = 1
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.CLS_FC[k], bn=True))
            pre_channel = rcnn_cfg.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        # Regression layer
        reg_layers = []
        pre_channel = channel_in
        for k in range(0, rcnn_cfg.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, rcnn_cfg.REG_FC[k], bn=True))
            pre_channel = rcnn_cfg.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, self.box_coder.code_size, activation=None))
        if rcnn_cfg.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(rcnn_cfg.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=rcnn_cfg.ROI_AWARE_POOL_SIZE, max_pts_each_voxel=128
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
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

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

    def roiaware_pool(self, batch_rois, rcnn_dict):
        """
        :param batch_rois: (B, N, 7 + ?) [x, y, z, w, l, h, rz] in LiDAR coords
        :param rcnn_dict:
        :return:
        """
        voxel_centers = rcnn_dict['voxel_centers']  # (npoints, 3)
        rpn_features = rcnn_dict['rpn_seg_features']  # (npoints, C)
        coords = rcnn_dict['coordinates']  # (npoints, 4)

        rpn_seg_score = rcnn_dict['rpn_seg_scores'].detach()  # (npoints)
        rpn_seg_mask = (rpn_seg_score > cfg.MODEL.RPN.BACKBONE.SEG_MASK_SCORE_THRESH)
        rpn_part_offsets = rcnn_dict['rpn_part_offsets'].clone().detach()
        rpn_part_offsets[rpn_seg_mask == 0] = 0
        part_features = torch.cat((rpn_part_offsets, rpn_seg_score.view(-1, 1)), dim=1)  # (npoints, 4)

        batch_size = batch_rois.shape[0]
        pooled_part_features_list, pooled_rpn_features_list = [], []

        for bs_idx in range(batch_size):
            bs_mask = (coords[:, 0] == bs_idx)
            cur_voxel_centers = voxel_centers[bs_mask]
            cur_part_features = part_features[bs_mask]
            cur_rpn_features = rpn_features[bs_mask]
            cur_roi = batch_rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            pooled_part_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_voxel_centers, cur_part_features, pool_method='avg'
            )  # (N, out_x, out_y, out_z, 4)
            pooled_rpn_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_voxel_centers, cur_rpn_features, pool_method='max'
            )  # (N, out_x, out_y, out_z, C)

            pooled_part_features_list.append(pooled_part_features)
            pooled_rpn_features_list.append(pooled_rpn_features)

        pooled_part_features = torch.cat(pooled_part_features_list, dim=0)  # (B * N, out_x, out_y, out_z, 4)
        pooled_rpn_features = torch.cat(pooled_rpn_features_list, dim=0)  # (B * N, out_x, out_y, out_z, C)

        return pooled_part_features, pooled_rpn_features

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def fake_sparse_idx(self, sparse_idx, batch_size_rcnn):
        print('Warning: GPU_%d: Sparse_Idx_Shape(%s) \r' % (cfg.LOCAL_RANK, str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def forward(self, rcnn_dict):
        """
        :param input_data: input dict
        :return:
        """
        rois = rcnn_dict['rois']
        batch_size = rois.shape[0]
        if self.training:
            targets_dict = self.assign_targets(batch_size, rcnn_dict)
            rois = targets_dict['rois']  # (B, N, 7)
            rcnn_dict['roi_raw_scores'] = targets_dict['roi_raw_scores']
            rcnn_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_part_features, pooled_rpn_features = self.roiaware_pool(rois, rcnn_dict)
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

        rcnn_cls = self.cls_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(shared_feature).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        ret_dict = {
            'rcnn_cls': rcnn_cls,
            'rcnn_reg': rcnn_reg,
            'rois': rois,
            'roi_raw_scores': rcnn_dict['roi_raw_scores'],
            'roi_labels': rcnn_dict['roi_labels']
        }

        if self.training:
            ret_dict.update(targets_dict)

        self.forward_ret_dict = ret_dict
        return ret_dict
