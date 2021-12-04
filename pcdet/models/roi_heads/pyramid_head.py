import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PyramidRoIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # mlps are shared with each grid point
        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        self.num_pyramid_levels = len(mlps)

        self.radius_by_rois = self.model_cfg.ROI_GRID_POOL.RADIUS_BY_ROIS
        self.radii = self.model_cfg.ROI_GRID_POOL.POOL_RADIUS
        self.enlarge_ratios = self.model_cfg.ROI_GRID_POOL.ENLARGE_RATIO
        self.grid_sizes = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.nsamples = self.model_cfg.ROI_GRID_POOL.NSAMPLE

        self.num_heads = self.model_cfg.ROI_GRID_POOL.NUM_HEADS
        self.head_dims = self.model_cfg.ROI_GRID_POOL.HEAD_DIMS
        self.attention_op = self.model_cfg.ROI_GRID_POOL.ATTENTION_OP
        assert len(self.radii) == len(self.enlarge_ratios) == len(self.grid_sizes) == len(self.nsamples) == self.num_pyramid_levels

        self.dp_value = self.model_cfg.ROI_GRID_POOL.get('DP_RATIO', 0.1)
        self.tr_mode = self.model_cfg.ROI_GRID_POOL.get('TR_MODE', 'Normal')

        self.roi_grid_pool_layer = pointnet2_stack_modules.PyramidModule(
            input_channels = input_channels,
            nsamples = self.nsamples,
            grid_sizes = self.grid_sizes,
            num_heads = self.num_heads,
            head_dims = self.head_dims,
            attention_op = self.attention_op,
            dp_value = self.dp_value,
            tr_mode = self.tr_mode,
        )

        pre_channel = 0
        for i in range(self.num_pyramid_levels):
            pre_channel += (self.grid_sizes[i] ** 3) * mlps[i][-1]

        """
        shared_fc_list = []
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
        """

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
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

    def roi_grid_pool(self, batch_dict):
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
        rois = batch_dict['rois']
        num_rois = rois.shape[1]
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        new_xyz_list = []
        new_xyz_r_list = []
        new_xyz_batch_cnt_list = []
        for i in range(len(self.grid_sizes)):
            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_enlarged_roi(
                rois, grid_size = self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i]
            )
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3).contiguous() # (B, N x grid_size^3, 3)
            if self.radius_by_rois:
                roi_grid_radius = self.get_radius_by_enlarged_roi(
                    rois, grid_size= self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i], radius_ratio = self.radii[i]
                )
                roi_grid_radius = roi_grid_radius.view(batch_size, -1, 1).contiguous() # (B, N x grid_size^3, 1)
            else:
                roi_grid_radius = rois.new_zeros(batch_size, num_rois * self.grid_sizes[i] * self.grid_sizes[i] * self.grid_sizes[i], 1).fill_(self.radii[i])

            new_xyz_list.append(global_roi_grid_points)
            new_xyz_r_list.append(roi_grid_radius)
            new_xyz_batch_cnt_list.append(roi_grid_radius.new_zeros(batch_size).int().fill_(roi_grid_radius.shape[1]))

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        cls_features, reg_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz_list=new_xyz_list,
            new_xyz_r_list=new_xyz_r_list,
            new_xyz_batch_cnt_list=new_xyz_batch_cnt_list,
            features=point_features.contiguous(),
            batch_size = batch_size,
            num_rois = num_rois,
        )  # (BN, \sum(grid_size^3), C)

        return cls_features, reg_features

    def get_radius_by_enlarged_roi(self, rois, grid_size, enlarged_ratio, radius_ratio):
        rois = rois.view(-1, rois.shape[-1])

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        roi_grid_radius = (enlarged_rois[:, 3:6] ** 2).sum(dim = 1).sqrt() # base_radius
        roi_grid_radius *= radius_ratio
        roi_grid_radius = roi_grid_radius.view(-1, 1, 1).repeat(1, grid_size ** 3, 1).contiguous() # (BN, grid_size^3, 1)
        return roi_grid_radius

    def get_global_grid_points_of_enlarged_roi(self, rois, grid_size, enlarged_ratio):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        local_roi_grid_points = self.get_dense_grid_points(enlarged_rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), enlarged_rois[:, 6]
        ) #.squeeze(dim=1)
        global_center = enlarged_rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (BN, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        cls_features, reg_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        batch_size_rcnn = cls_features.shape[0]
        cls_features = cls_features.reshape(batch_size_rcnn, -1, 1)
        reg_features = reg_features.reshape(batch_size_rcnn, -1, 1)

        rcnn_cls = self.cls_layers(cls_features).squeeze(dim=-1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(reg_features).squeeze(dim=-1).contiguous()  # (B, C)

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

class PyramidRoIHeadV2(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        # mlps are shared with each grid point
        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        self.num_pyramid_levels = len(mlps)

        self.radius_by_rois = self.model_cfg.ROI_GRID_POOL.RADIUS_BY_ROIS
        self.radii = self.model_cfg.ROI_GRID_POOL.POOL_RADIUS
        self.enlarge_ratios = self.model_cfg.ROI_GRID_POOL.ENLARGE_RATIO
        self.grid_sizes = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.nsamples = self.model_cfg.ROI_GRID_POOL.NSAMPLE

        self.num_heads = self.model_cfg.ROI_GRID_POOL.NUM_HEADS
        self.head_dims = self.model_cfg.ROI_GRID_POOL.HEAD_DIMS
        self.attention_op = self.model_cfg.ROI_GRID_POOL.ATTENTION_OP
        assert len(self.radii) == len(self.enlarge_ratios) == len(self.grid_sizes) == len(self.nsamples) == self.num_pyramid_levels

        self.predict_radii = self.model_cfg.ROI_GRID_POOL.PRE_RADII
        self.predict_ns = self.model_cfg.ROI_GRID_POOL.PRE_NS
        self.predict_norm = self.model_cfg.ROI_GRID_POOL.PRE_NORM
        self.use_weights_before = self.model_cfg.ROI_GRID_POOL.USE_WEIGHTS_BEFORE

        self.roi_grid_pool_layer = pointnet2_stack_modules.PyramidModuleV2(
            input_channels = input_channels,
            nsamples = self.nsamples,
            grid_sizes = self.grid_sizes,
            num_heads = self.num_heads,
            head_dims = self.head_dims,
            attention_op = self.attention_op,
            predict_radii = self.predict_radii,
            predict_ns = self.predict_ns,
            norm_factors = self.predict_norm,
            pre_weights = self.use_weights_before,
        )

        pre_channel = 0
        for i in range(self.num_pyramid_levels):
            pre_channel += (self.grid_sizes[i] ** 3) * mlps[i][-1]

        """
        shared_fc_list = []
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
        """

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
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

    def roi_grid_pool(self, batch_dict):
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
        rois = batch_dict['rois']
        num_rois = rois.shape[1]
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        new_xyz_list = []
        new_xyz_r_list = []
        new_xyz_batch_cnt_list = []
        for i in range(len(self.grid_sizes)):
            global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_enlarged_roi(
                rois, grid_size = self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i]
            )
            global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3).contiguous() # (B, N x grid_size^3, 3)
            if self.radius_by_rois:
                roi_grid_radius = self.get_radius_by_enlarged_roi(
                    rois, grid_size= self.grid_sizes[i], enlarged_ratio = self.enlarge_ratios[i], radius_ratio = self.radii[i]
                )
                roi_grid_radius = roi_grid_radius.view(batch_size, -1, 1).contiguous() # (B, N x grid_size^3, 1)
            else:
                roi_grid_radius = rois.new_zeros(batch_size, num_rois * self.grid_sizes[i] * self.grid_sizes[i] * self.grid_sizes[i], 1).fill_(self.radii[i])

            new_xyz_list.append(global_roi_grid_points)
            new_xyz_r_list.append(roi_grid_radius)
            new_xyz_batch_cnt_list.append(roi_grid_radius.new_zeros(batch_size).int().fill_(roi_grid_radius.shape[1]))

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        anchor_xyz =  rois[..., :3].contiguous().view(-1, 3) # take center of each roi as anchor points
        anchor_batch_cnt = anchor_xyz.new_zeros(batch_size).fill_(num_rois).int()
        temperature = batch_dict['temperature']

        cls_features, reg_features = self.roi_grid_pool_layer(
            anchor_xyz = anchor_xyz,
            anchor_batch_cnt = anchor_batch_cnt,
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz_list=new_xyz_list,
            new_xyz_r_list=new_xyz_r_list,
            new_xyz_batch_cnt_list=new_xyz_batch_cnt_list,
            features=point_features.contiguous(),
            batch_size = batch_size,
            num_rois = num_rois,
            temperature = temperature,
        )  # (BN, \sum(grid_size^3), C)

        return cls_features, reg_features

    def get_radius_by_enlarged_roi(self, rois, grid_size, enlarged_ratio, radius_ratio):
        rois = rois.view(-1, rois.shape[-1])

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        roi_grid_radius = (enlarged_rois[:, 3:6] ** 2).sum(dim = 1).sqrt() # base_radius
        roi_grid_radius *= radius_ratio
        roi_grid_radius = roi_grid_radius.view(-1, 1, 1).repeat(1, grid_size ** 3, 1).contiguous() # (BN, grid_size^3, 1)
        return roi_grid_radius

    def get_global_grid_points_of_enlarged_roi(self, rois, grid_size, enlarged_ratio):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        enlarged_rois = rois.clone()

        if len(enlarged_ratio) == 1:
            enlarged_rois[:, 3:6] = enlarged_ratio * enlarged_rois[:, 3:6]
        elif len(enlarged_ratio) == 3:
            enlarged_rois[:, 3] = enlarged_ratio[0] * enlarged_rois[:, 3]
            enlarged_rois[:, 4] = enlarged_ratio[1] * enlarged_rois[:, 4]
            enlarged_rois[:, 5] = enlarged_ratio[2] * enlarged_rois[:, 5]
        else:
            raise Exception("enlarged_ratio has to be int or list of 3 int")

        local_roi_grid_points = self.get_dense_grid_points(enlarged_rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), enlarged_rois[:, 6]
        ) #.squeeze(dim=1)
        global_center = enlarged_rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (BN, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict, targets_dict):
        """
        :param input_data: input dict
        :return:
        """

        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        """

        # RoI aware pooling
        cls_features, reg_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        batch_size_rcnn = cls_features.shape[0]
        cls_features = cls_features.reshape(batch_size_rcnn, -1, 1)
        reg_features = reg_features.reshape(batch_size_rcnn, -1, 1)

        rcnn_cls = self.cls_layers(cls_features).squeeze(dim=-1).contiguous()  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(reg_features).squeeze(dim=-1).contiguous()  # (B, C)

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
