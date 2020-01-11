import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.voxel_encoder as voxel_encoder
import importlib
import models.rpn_head as rpn_head
import spconv
import utils.loss_utils as loss_utils
import utils.box_coder as box_coder_utils
import utils.kitti_utils as kitti_utils
import utils.iou3d.iou3d_utils as iou3d_utils
import utils.box_torch_ops as box_torch_ops
from rpn.proposal_layer import proposal_layer
from config import cfg
from datasets.all_datasets import get_dataset_class


def get_model():
    return PartA2Net


class PartA2Net(nn.Module):
    def __init__(self, num_class, target_assigner, output_shape):
        super().__init__()

        self.num_class = num_class
        self.box_coder = target_assigner.box_coder
        self.output_shape = output_shape
        self.sparse_shape = output_shape + [1, 0, 0]
        self.num_anchors_per_location = target_assigner.num_anchors_per_location

        voxel_encoder_cfg = cfg.RPN_STAGE.VOXEL_ENCODER
        if 'Pillar' not in cfg.RPN_STAGE.NET:
            self.voxel_feature_extractor = getattr(voxel_encoder, voxel_encoder_cfg.NAME)(
                num_input_features=voxel_encoder_cfg.INPUT_FEATURES,
                num_filters=voxel_encoder_cfg.NUM_FILTERS,
                with_distance=voxel_encoder_cfg.WITH_DISTANCE
            )
        else:
            import models.pointpillar as pointpillar_utils
            self.voxel_feature_extractor = pointpillar_utils.get_model(voxel_encoder_cfg.NAME)(
                num_input_features=voxel_encoder_cfg.INPUT_FEATURES, use_norm=True,
                num_filters=voxel_encoder_cfg.NUM_FILTERS,
                with_distance=voxel_encoder_cfg.WITH_DISTANCE,
                voxel_size=cfg.VOXEL_GENERATOR.VOXEL_SIZE,
                pc_range=cfg.VOXEL_GENERATOR.POINT_CLOUD_RANGE,
            )

        voxel_feature_num = voxel_encoder_cfg.NUM_FILTERS

        if cfg.RPN_STAGE.UNET.PART_SEG_ENABLED:
            RPN_MODEL = importlib.import_module(cfg.RPN_STAGE.NET)
            self.rpn_net = RPN_MODEL.get_model()(voxel_feature_num[-1])
        elif 'Pillar' in cfg.RPN_STAGE.NET:
            import models.pointpillar as pointpillar_utils
            dense_shape = [1] + self.output_shape.tolist() + [voxel_feature_num[-1]]

            self.rpn_net = pointpillar_utils.get_model(cfg.RPN_STAGE.NET)(
                output_shape=dense_shape,
                use_norm=True,
                num_input_features=voxel_feature_num[-1]
            )
        else:
            import models.rpn_backbone as rpn_backbone
            self.rpn_net = rpn_backbone.get_model(cfg.RPN_STAGE.NET, input_channels=voxel_feature_num[-1])

        rpn_head_cfg = cfg.RPN_STAGE.RPN_HEAD
        self.rpn_head = getattr(rpn_head, rpn_head_cfg.NET)(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_head_cfg.LAYER_NUMS,
            layer_strides=rpn_head_cfg.LAYER_STRIDES,
            num_filters=rpn_head_cfg.NUM_FILTERS,
            upsample_strides=rpn_head_cfg.UPSAMPLE_STRIDES,
            num_upsample_filters=rpn_head_cfg.NUM_UPSAMPLE_FILTERS,
            num_input_features=rpn_head_cfg.NUM_INPUT_FEATURES,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS,
            use_direction_classifier=cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER,
            use_groupnorm=rpn_head_cfg.USE_GROUPNORM,
            num_groups=rpn_head_cfg.NUM_GROUPS,
            box_code_size=target_assigner.box_coder.code_size,
            num_direction_bins=cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS
        )

        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        # loss function definition
        self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        code_weights = cfg.RPN_STAGE.RPN_HEAD.LOSS_WEIGHTS['code_weights']
        if not cfg.WITH_VELOCITY:
            code_weights = code_weights[:7]
        if cfg.RPN_STAGE.RPN_HEAD.LOSS_REG == 'bin-based':
            rpn_code_weights = code_weights[3:7]
        else:
            rpn_code_weights = code_weights

        self.rpn_reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0, code_weights=rpn_code_weights)
        self.rpn_dir_loss_func = loss_utils.WeightedSoftmaxClassificationLoss()

        if cfg.RCNN_STAGE.ENABLED:
            if cfg.RCNN_STAGE.LOSS_REG == 'smooth-l1':
                self.rcnn_reg_loss_func = loss_utils.WeightedSmoothL1LocalizationLoss(sigma=3.0,
                                                                                      code_weights=code_weights)
                self.rcnn_box_coder = self.box_coder
            elif cfg.RCNN_STAGE.LOSS_REG == 'bin-based':
                self.rcnn_reg_loss_func = loss_utils.get_bin_based_reg_loss_lidar
                bin_loss_cfg = cfg.RCNN_STAGE.BIN_LOSS_CFG
                self.rcnn_box_coder = box_coder_utils.BinBasedCoder(
                    loc_scope=bin_loss_cfg['LOC_SCOPE'],
                    loc_bin_size=bin_loss_cfg['LOC_BIN_SIZE'],
                    num_head_bin=bin_loss_cfg['NUM_HEAD_BIN'],
                    get_xz_fine=True, get_y_by_bin=bin_loss_cfg['LOC_Y_BY_BIN'],
                    loc_y_scope=bin_loss_cfg['LOC_Y_SCOPE'],
                    loc_y_bin_size=bin_loss_cfg['LOC_Y_BIN_SIZE'],
                    get_ry_fine=True,
                    canonical_transform=cfg.RCNN_STAGE.INPUT_ROTATE
                )
            else:
                raise NotImplementedError

            RCNN_MODEL = importlib.import_module('%s' % cfg.RCNN_STAGE.NET)
            rcnn_model = RCNN_MODEL.get_model()
            self.rcnn_net = rcnn_model(code_size=self.rcnn_box_coder.code_size)

    def update_global_step(self):
        self.global_step += 1

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            if cfg.RPN_STAGE.FIXED:
                # set RPN batch norm in eval mode during training
                self.voxel_feature_extractor.apply(set_bn_eval)
                self.rpn_net.apply(set_bn_eval)
                self.rpn_head.apply(set_bn_eval)

    def forward(self, input_dict):
        batch_anchors = input_dict['anchors']
        batch_size = batch_anchors.shape[0]
        voxels = input_dict['voxels']
        num_points = input_dict['num_points']
        coors = input_dict['coordinates']
        voxel_centers = input_dict['voxel_centers']

        ret_dict = {}
        tb_dict = {}
        disp_dict = {}

        # RPN inference
        with torch.set_grad_enabled((cfg.RPN_STAGE.FIXED is False and self.training)):
            coors = coors.int()

            if 'Pillar' not in cfg.RPN_STAGE.NET:
                voxel_features = self.voxel_feature_extractor(voxels, num_points)
                input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
                try:
                    unet_ret_dict = self.rpn_net(input_sp_tensor, voxel_centers)
                except:
                    unet_ret_dict = self.rpn_net(input_sp_tensor)
            else:
                voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
                batch_canvas = self.rpn_net(voxel_features, coors, batch_size)
                unet_ret_dict = {'spatial_features': batch_canvas}

            rpn_preds_dict = self.rpn_head(unet_ret_dict['spatial_features'])

            if cfg.RPN_STAGE.UNET.DECODER_ENABLED and cfg.RPN_STAGE.UNET.PART_SEG_ENABLED:
                u_cls_preds = unet_ret_dict['u_cls_preds'].view(-1)
                u_reg_preds = unet_ret_dict['u_reg_preds']
                seg_features = unet_ret_dict['seg_features']
                seg_score = torch.sigmoid(u_cls_preds)
                ret_dict['rpn_u_cls_preds'] = u_cls_preds
                ret_dict['rpn_u_reg_preds'] = u_reg_preds
            else:
                u_cls_preds = u_reg_preds = None

            rpn_box_preds = rpn_preds_dict['box_preds']
            rpn_cls_preds = rpn_preds_dict['cls_preds']
            rpn_dir_cls_preds = rpn_preds_dict['dir_cls_preds'] if 'dir_cls_preds' in rpn_preds_dict else None
            ret_dict['rpn_cls_preds'] = rpn_cls_preds
            ret_dict['rpn_box_preds'] = rpn_box_preds
            ret_dict['rpn_dir_cls_preds'] = rpn_dir_cls_preds

        if cfg.RCNN_STAGE.ENABLED:
            # proposal layer
            with torch.no_grad():
                batch_anchors = batch_anchors.view(batch_size, -1, batch_anchors.shape[-1])  # (B, N, 7 + ?)
                num_anchors = batch_anchors.shape[1]
                batch_cls_preds = rpn_cls_preds.view(batch_size, num_anchors, -1)
                batch_box_preds = rpn_box_preds.view(batch_size, -1,
                                                     rpn_box_preds.shape[-1] // self.num_anchors_per_location)

                if cfg.RPN_STAGE.RPN_HEAD.LOSS_REG == 'smooth-l1':
                    batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
                elif cfg.RPN_STAGE.RPN_HEAD.LOSS_REG == 'bin-based':
                    def decode_size_angle(box_encodings, anchors):
                        """
                        :param box_encodings: (N, 4)
                        :param anchors: (N, 4)
                        :return:
                        """
                        wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
                        wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

                        lg = torch.exp(lt) * la
                        wg = torch.exp(wt) * wa
                        hg = torch.exp(ht) * ha
                        rg = rt + ra

                        return torch.cat([wg, lg, hg, rg], dim=-1)

                    batch_size_angle = decode_size_angle(batch_box_preds[..., -4:], batch_anchors[..., -4:])

                    bin_cfg = cfg.RPN_STAGE.RPN_HEAD.BIN_LOSS_CFG
                    center_pred = batch_box_preds[..., :-4].view(-1, batch_box_preds.shape[-1] - 4)
                    center_anchors = batch_anchors[..., :-4].view(-1, 3)
                    batch_center = box_coder_utils.decode_center_by_bin(
                        center_pred, center_anchors, loc_scope=bin_cfg.LOC_SCOPE, loc_bin_size=bin_cfg.LOC_BIN_SIZE
                    ).view(batch_size_angle.shape[0], -1, 3)

                    batch_box_preds = torch.cat((batch_center, batch_size_angle), dim=2)
                else:
                    raise NotImplementedError

                if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                    dir_preds = rpn_preds_dict['dir_cls_preds']  # (bs, H, W, 2*anchor)
                    dir_preds = dir_preds.view(batch_size, -1, cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)
                    dir_labels = torch.max(dir_preds, dim=-1)[1]

                    if cfg.RPN_STAGE.RPN_HEAD.USE_OLD_ORT:
                        opp_labels = (batch_box_preds[..., 6] > 0) ^ dir_labels.byte()
                        batch_box_preds[..., 6] += torch.where(opp_labels,
                                                               torch.tensor(np.pi).type_as(batch_box_preds),
                                                               torch.tensor(0.0).type_as(batch_box_preds))
                    else:
                        period = (2 * np.pi / cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)
                        dir_rot = box_torch_ops.limit_period(batch_box_preds[..., 6] - cfg.RPN_STAGE.RPN_HEAD.DIR_OFFSET,
                                                             cfg.RPN_STAGE.RPN_HEAD.DIR_LIMIT_OFFSET, period)
                        batch_box_preds[..., 6] = dir_rot + cfg.RPN_STAGE.RPN_HEAD.DIR_OFFSET \
                                                  + period * dir_labels.to(batch_box_preds.dtype)

                roi_dict = proposal_layer(batch_size, batch_cls_preds, batch_box_preds,
                                          code_size=self.box_coder.code_size,
                                          mode='TRAIN' if self.training else 'TEST')

            # RCNN inference
            rcnn_input_dict = {
                'voxel_centers': voxel_centers,
                'coordinates': coors,
                'part_seg_score': seg_score,
                'seg_features': seg_features,
                'bev_features': unet_ret_dict['spatial_features']
            }
            rcnn_input_dict.update(roi_dict)
            if 'gt_boxes' in input_dict:
                rcnn_input_dict['gt_boxes'] = input_dict['gt_boxes']

            if cfg.RPN_STAGE.UNET.PART_REG_ENABLED:
                part_reg_offset = torch.sigmoid(u_reg_preds)
                rcnn_input_dict['part_reg_offset'] = part_reg_offset

            rcnn_ret_dict = self.rcnn_net.forward(rcnn_input_dict)
            ret_dict['rois'] = rcnn_ret_dict['rois']
            ret_dict['rcnn_cls'] = rcnn_ret_dict['rcnn_cls']
            ret_dict['rcnn_reg'] = rcnn_ret_dict['rcnn_reg']
            ret_dict['roi_raw_scores'] = rcnn_ret_dict['roi_raw_scores']
            ret_dict['roi_labels'] = rcnn_ret_dict['roi_labels']
            if cfg.RCNN_STAGE.REG_2D_BBOX:
                ret_dict['rcnn_reg_2d'] = rcnn_ret_dict['rcnn_reg_2d']

        if self.training:
            loss = 0
            if not cfg.RPN_STAGE.FIXED:
                # RPN loss
                rpn_loss, rpn_tb_dict = self.get_rpn_loss(u_cls_preds, u_reg_preds, rpn_cls_preds, rpn_box_preds,
                                                          input_dict, dir_cls_preds=rpn_dir_cls_preds)
                loss += rpn_loss
                tb_dict.update(rpn_tb_dict)

            if cfg.RCNN_STAGE.ENABLED:
                # RCNN loss
                rcnn_loss, rcnn_tb_dict = self.get_rcnn_loss(rcnn_ret_dict, input_dict)
                loss += rcnn_loss

                # for visualization
                rcnn_cls_labels = rcnn_ret_dict['rcnn_cls_labels'].float().view(-1)
                fg_thresh = min(cfg.RCNN_STAGE.REG_FG_THRESH, cfg.RCNN_STAGE.CLS_FG_THRESH)
                fg_num = (rcnn_cls_labels > fg_thresh).sum().item()
                bg_num = (rcnn_cls_labels == 0).sum().item()
                # print('FG_Num: %d' % fg_num)
                tb_dict['rcnn_fg_num'] = fg_num
                tb_dict['rcnn_bg_num'] = bg_num
                tb_dict.update(rcnn_tb_dict)
                disp_dict['rcnn_fg_num'] = fg_num

                if 'rcnn_loss_dense_iou' in rcnn_ret_dict:
                    w = cfg.TRAIN.LOSS_WEIGHTS['rcnn_loss_dense_iou'] if 'rcnn_loss_dense_iou' in cfg.TRAIN.LOSS_WEIGHTS else 1.0
                    rcnn_ret_dict['rcnn_loss_dense_iou'] = rcnn_ret_dict['rcnn_loss_dense_iou'] * w

                    loss += rcnn_ret_dict['rcnn_loss_dense_iou']
                    tb_dict['rcnn_loss_dense_iou'] = rcnn_ret_dict['rcnn_loss_dense_iou'].item()

                if 'rcnn_loss_dense_corner' in rcnn_ret_dict:
                    loss += rcnn_ret_dict['rcnn_loss_dense_corner']
                    tb_dict['rcnn_loss_dense_corner'] = rcnn_ret_dict['rcnn_loss_dense_corner'].item()

            ret_dict['loss'] = loss

            return ret_dict, tb_dict, disp_dict
        else:
            # prediction mode
            dataset = get_dataset_class(cfg.DATASET)
            batch_rois = batch_roi_labels = batch_roi_raw_scores = batch_rcnn_bbox = None
            batch_gt_boxes = input_dict['gt_boxes'] if 'gt_boxes' in input_dict else None

            if not cfg.RCNN_STAGE.ENABLED:
                # generate RPN boxes
                batch_anchors = input_dict['anchors'].view(batch_size, -1, input_dict['anchors'].shape[-1])
                assert 'anchors_mask' not in input_dict

                num_class_with_bg = self.num_class + 1 if not cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS else self.num_class
                batch_box_preds = rpn_box_preds.view(batch_size, -1, self.box_coder.code_size)
                batch_cls_preds = rpn_cls_preds.view(batch_size, -1, num_class_with_bg).float()
                batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors).float()

                if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
                    batch_dir_preds = rpn_dir_cls_preds.view(batch_size, -1, cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)
                    batch_dir_labels = torch.max(batch_dir_preds, dim=-1)[1]

                    if cfg.RPN_STAGE.RPN_HEAD.USE_OLD_ORT == -1:
                        opp_labels = (batch_box_preds[..., 6] > 0) ^ batch_dir_labels.byte()
                        batch_box_preds[..., 6] += torch.where(opp_labels, torch.tensor(np.pi).type_as(batch_box_preds),
                                                              torch.tensor(0.0).type_as(batch_box_preds))
                    else:
                        period = (2 * np.pi / cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)
                        dir_rot = box_torch_ops.limit_period(batch_box_preds[..., 6] - cfg.RPN_STAGE.RPN_HEAD.DIR_OFFSET,
                                                             cfg.RPN_STAGE.RPN_HEAD.DIR_LIMIT_OFFSET, period)
                        batch_box_preds[..., 6] = dir_rot + cfg.RPN_STAGE.RPN_HEAD.DIR_OFFSET \
                                                 + period * batch_dir_labels.to(batch_box_preds.dtype)

            else:
                # generate RCNN boxes
                rcnn_cls = ret_dict['rcnn_cls']  # (B * N, 1)
                rcnn_reg = ret_dict['rcnn_reg']  # (B * N, C)
                rois = ret_dict['rois']  # (B, N, 7)
                roi_size = rois[:, :, 3:6]
                code_size = self.rcnn_box_coder.code_size

                if cfg.RCNN_STAGE.LOSS_REG == 'bin-based':
                    rcnn_boxes3d = self.rcnn_box_coder.decode_torch(rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                                                    rois.view(-1, code_size),
                                                                    anchor_size=roi_size)  # (N, 7)
                    rcnn_boxes3d = rcnn_boxes3d.view(batch_size, -1, code_size)
                elif cfg.RCNN_STAGE.LOSS_REG == 'smooth-l1':
                    roi_ry = rois[:, :, 6].view(-1)
                    roi_xyz = rois[:, :, 0:3].view(-1, 3)
                    local_rois = rois.clone().detach()
                    local_rois[:, :, 0:3] = 0
                    rcnn_boxes3d = self.rcnn_box_coder.decode_torch(rcnn_reg.view(local_rois.shape[0], -1, code_size),
                                                                    local_rois).view(-1, code_size)

                    if 'GENERATE_BOX_BY_DENSELOC' in cfg.RCNN_STAGE and cfg.RCNN_STAGE.GENERATE_BOX_BY_DENSELOC:
                        if 'USE_FINAL_REG' not in cfg.TEST or (cfg.TEST.USE_FINAL_REG is False):
                            # import pdb
                            # pdb.set_trace()
                            rcnn_boxes3d[:, 0:3] = rcnn_ret_dict['ans_pred'][:, 0:3]
                            # rcnn_boxes3d[:, 3:6] = rois.view(-1, code_size)[:, 3:6]
                            rcnn_boxes3d[:, 3:6] = rcnn_ret_dict['ans_pred'][:, 3:6]
                            rcnn_boxes3d[:, -1] = roi_ry + rcnn_ret_dict['ans_pred'][:, -1]

                    # rcnn_boxes3d[:, [0, 1]] = (rcnn_boxes3d[:, [0, 1]] + rcnn_ret_dict['ans_pred'][:, [0, 1]]) / 2
                    # rcnn_boxes3d[:, -1] = roi_ry + (rcnn_reg[:, -1] + rcnn_ret_dict['ans_pred'][:, -1]) / 2
                    # import pdb
                    # pdb.set_trace()

                    rcnn_boxes3d = box_coder_utils.rotate_pc_along_z_torch(rcnn_boxes3d, (roi_ry + np.pi / 2))
                    rcnn_boxes3d[:, 0:3] += roi_xyz
                    rcnn_boxes3d = rcnn_boxes3d.view(batch_size, -1, code_size)
                else:
                    raise NotImplementedError

                if cfg.RCNN_STAGE.REG_2D_BBOX:
                    # only available for kttii
                    batch_rcnn_bbox = dataset.predict_decode_2d_bbox(input_dict, ret_dict, batch_size)

                batch_cls_preds = rcnn_cls.view(batch_size, -1)
                batch_box_preds = rcnn_boxes3d
                # batch_box_preds = rois
                # batch_cls_preds = torch.sigmoid(ret_dict['roi_raw_scores'])
                batch_rois = rois
                batch_roi_raw_scores = ret_dict['roi_raw_scores']  # (B, N)
                batch_roi_labels = ret_dict['roi_labels']  # (B, N)

            mode = 'RPN' if not cfg.RCNN_STAGE.ENABLED else 'RCNN'
            pred_dicts, recall_dicts = self.predict(dataset, input_dict, batch_cls_preds, batch_box_preds,
                                                    batch_gt_boxes, batch_rois, batch_roi_labels, batch_roi_raw_scores,
                                                    batch_rcnn_bbox, mode=mode)
            ret_dict.update(recall_dicts)
            return pred_dicts, ret_dict

    def predict(self, dataset, input_dict, batch_cls_preds, batch_box_preds, batch_gt_boxes=None,
                batch_rois=None, batch_roi_labels=None, batch_roi_raw_scores=None, batch_rcnn_bbox=None, mode='RPN'):
        recall_dict = {'roi_05': 0, 'rcnn_05': 0, 'gt': 0, 'roi_07': 0, 'rcnn_07': 0,
                       'rcnn_rank_acc': 0, 'valid_sample_num': 0, 'rcnn_iou_diff': 0}
        predictions_dicts = []

        batch_size = batch_cls_preds.shape[0]
        batch_index = np.arange(batch_size)

        for index, cls_preds, box_preds in zip(batch_index, batch_cls_preds, batch_box_preds):
            if mode == 'RPN':
                if cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS:
                    # this don't support softmax
                    assert cfg.RPN_STAGE.USE_SIGMOID_SCORE is True
                    rank_scores = torch.sigmoid(cls_preds)
                    rank_raw_scores = cls_preds
                else:
                    # encode background as first element in one-hot vector
                    if cfg.RPN_STAGE.USE_SIGMOID_SCORE:
                        rank_scores = torch.sigmoid(cls_preds)[..., 1:]
                    else:
                        rank_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
                    rank_raw_scores = cls_preds[..., 1:]

                if cfg.TEST.USE_MULTI_CLASSES_NMS:
                    top_scores = rank_scores
                    top_raw_scores = rank_raw_scores
                    top_labels = None
                else:
                    top_scores, top_labels = torch.max(rank_scores, dim=-1)
                    top_labels = top_labels + 1  # shift to [1, num_classes]
                    top_raw_scores, _ = torch.max(rank_raw_scores, dim=-1)
            else:
                if batch_gt_boxes is not None:
                    gt_boxes = batch_gt_boxes[index]
                    gt_iou = self.calculate_recall(cls_preds, box_preds, batch_rois[index], gt_boxes, recall_dict)

                    pred_mat = kitti_utils.score_to_compare_matrix(cls_preds)
                    gt_mat = kitti_utils.score_to_compare_matrix(gt_iou)
                    mask = (torch.abs(gt_mat) > 1e-3)
                    fg = ((pred_mat > 0) == (gt_mat > 0))
                    rank_acc = (fg * mask).float().sum() / torch.clamp_min(mask.float().sum(), min=1.0)
                    recall_dict['rcnn_rank_acc'] += rank_acc.item()
                    recall_dict['valid_sample_num'] += (gt_iou.max() > 0).item()
                    iou_diff = torch.abs((torch.sigmoid(cls_preds) - gt_iou))
                    recall_dict['rcnn_iou_diff'] += iou_diff.mean().item()
                else:
                    gt_iou = cls_preds
                rank_raw_scores, rank_scores = self.get_rank_scores(cls_preds, batch_roi_raw_scores[index], gt_iou)

                top_scores = rank_scores.squeeze(-1)
                top_labels = batch_roi_labels[index]
                top_raw_scores = rank_raw_scores

            thresh = torch.tensor([cfg.TEST.SCORE_THRESH], device=rank_scores.device).type_as(rank_scores)
            if cfg.TEST.USE_MULTI_CLASSES_NMS and mode == 'RPN':
                boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds)

                selected_list = []
                selected_top_scores = []
                selected_top_raw_scores = []
                selected_labels = []
                num_classes = len(cfg.CLASSES)
                for k in range(0, num_classes):
                    class_scores = rank_scores[:, k]
                    class_scores_keep = class_scores >= thresh

                    if class_scores_keep.int().sum() > 0:
                        original_idxs = class_scores_keep.nonzero().view(-1)
                        cur_boxes_for_nms = boxes_for_nms[class_scores_keep]
                        cur_top_raw_scores = top_raw_scores[class_scores_keep, k]
                        cur_selected = iou3d_utils.nms_gpu(cur_boxes_for_nms, cur_top_raw_scores, cfg.TEST.NMS_THRESH)

                        if cur_selected.shape[0] > 0:
                            selected_list.append(original_idxs[cur_selected])
                            selected_top_scores.append(class_scores[class_scores_keep][cur_selected])
                            selected_top_raw_scores.append(cur_top_raw_scores[cur_selected])
                            selected_labels.append(
                                torch.full([cur_selected.shape[0]], k + 1,
                                           dtype=torch.int64, device=box_preds.device)
                            )

                if selected_list.__len__() >= 1:
                    selected = torch.cat(selected_list, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_top_scores = torch.cat(selected_top_scores, dim=0)
                    selected_top_raw_scores = torch.cat(selected_top_raw_scores, dim=0)
                else:
                    selected = []
                    selected_top_scores = top_scores[selected]
                    selected_top_raw_scores = top_raw_scores[selected]
                    selected_labels = selected_top_scores
                selected_boxes = box_preds[selected]
            else:
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] > 0:
                    top_raw_scores = top_raw_scores.masked_select(top_scores_keep)
                    box_preds = box_preds[top_scores_keep]

                    # NMS in birdeye view
                    # TODO: use my rotated nms, not checked
                    # boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds)
                    # selected = iou3d_utils.nms_gpu(boxes_for_nms, top_raw_scores, cfg.TEST.NMS_THRESH)
                    # selected = selected[:cfg.MAX_OBJECTS_EACH_SCENE]  # max boxes 500

                    top_raw_scores_nms, indices = torch.topk(top_raw_scores,
                                                             k=min(cfg.TEST.FINAL_NMS_PRE_MAXSIZE,
                                                                   top_raw_scores.shape[0]))
                    box_preds_nms = box_preds[indices]
                    boxes_for_nms = kitti_utils.boxes3d_to_bev_torch_lidar(box_preds_nms)#

                    if cfg.TEST.NMS_TYPE == 'rotated':
                        keep_idx = iou3d_utils.nms_gpu(boxes_for_nms, top_raw_scores_nms, cfg.TEST.NMS_THRESH)
                        selected = indices[keep_idx[:cfg.TEST.FINAL_NMS_POST_MAXSIZE]]
                    elif cfg.TEST.NMS_TYPE == 'soft_rotated_nms':
                        import utils.nms.nms_utils as nms_utils
                        top_scores_nms = torch.sigmoid(top_raw_scores_nms)
                        keep_idx, soft_scores = nms_utils.soft_nms_cpu(boxes_for_nms, top_scores_nms,
                                                                       score_thresh=cfg.TEST.SOFTNMS_SCORE_THRESH,
                                                                       sigma=cfg.TEST.NMS_SIGMA,
                                                                       Nt=0.3,
                                                                       soft_type=cfg.TEST.NMS_SOFT_TYPE)
                        selected = indices[keep_idx[:cfg.TEST.FINAL_NMS_POST_MAXSIZE]]
                        top_raw_scores[selected] = top_scores[selected] = soft_scores
                    else:
                        raise NotImplementedError
                    selected = selected[:cfg.MAX_OBJECTS_EACH_SCENE]  # max boxes 500
                else:
                    selected = []

                selected_labels = top_labels[top_scores_keep][selected]
                selected_top_scores = top_scores[selected]
                selected_top_raw_scores = top_raw_scores[selected]
                selected_boxes = box_preds[selected]

            record_dict = {
                'boxes': selected_boxes,
                'scores': selected_top_scores,
                'raw_scores': selected_top_raw_scores,
                'labels': selected_labels
            }

            if mode == 'RCNN':
                record_dict['roi_raw_scores'] = batch_roi_raw_scores[index][top_scores_keep][selected]
                record_dict['rois'] = batch_rois[index][top_scores_keep][selected]
                record_dict['gt_iou'] = gt_iou[top_scores_keep][selected]

                if batch_rcnn_bbox is not None:
                    record_dict['predict_bbox'] = batch_rcnn_bbox[index][top_scores_keep][selected]

                # filter invalid RoIs
                mask = (record_dict['rois'][:, 3:6].sum(dim=1) > 0)
                if mask.sum() != record_dict['rois'].shape[0]:
                    kitti_utils.dict_select(record_dict, mask)

            predictions_dict = dataset.generate_prediction_dict(input_dict, index, record_dict)

            predictions_dicts.append(predictions_dict)
        return predictions_dicts, recall_dict

    def get_rank_scores(self, cls_preds, roi_rawscores, gt_iou):
        if cfg.TEST.SCORE_TYPE == 'rcnn':
            rank_scores = torch.sigmoid(cls_preds)
            rank_raw_scores = cls_preds.squeeze(-1)
        elif cfg.TEST.SCORE_TYPE == 'roi':
            rank_scores = torch.sigmoid(roi_rawscores)
            rank_raw_scores = roi_rawscores
        elif cfg.TEST.SCORE_TYPE == 'mul':
            rank_scores = rank_raw_scores = torch.sigmoid(roi_rawscores) * torch.sigmoid(cls_preds)
        elif cfg.TEST.SCORE_TYPE == 'gt':
            rank_scores = rank_raw_scores = raw_scores = gt_iou
        elif cfg.TEST.SCORE_TYPE == 'avg':
            rank_scores = rank_raw_scores = (torch.sigmiod(roi_rawscores) + torch.sigmoid(cls_preds)) / 2
        elif cfg.TEST.SCORE_TYPE == 'combine':
            rank_raw_scores = cls_preds.clone().detach()
            mask = torch.sigmoid(roi_rawscores) < 0.5
            rank_raw_scores[mask] = rank_raw_scores[mask] / 1.5
            rank_scores = torch.sigmoid(rank_raw_scores)
        elif cfg.TEST.SCORE_TYPE == 'mul_raw':
            roi_scores = torch.sigmoid(roi_rawscores)
            rank_raw_scores = roi_scores * cls_preds
            rank_scores = torch.sigmoid(rank_raw_scores)
        else:
            raise NotImplementedError

        return rank_raw_scores, rank_scores

    def calculate_recall(self, cls_preds, box_preds, rois, gt_boxes, recall_dict):
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.sum() == 0:
            gt_iou = cls_preds * 0
        else:
            box_type = 'nuscenes' if cfg.DATASET == 'nuscenes' else 'lidar'
            iou3d_roi = iou3d_utils.boxes_iou3d_gpu(rois, cur_gt, box_type=box_type)
            iou3d_rcnn = iou3d_utils.boxes_iou3d_gpu(box_preds, cur_gt, box_type=box_type)

            roi_recalled = (iou3d_roi.max(dim=0)[0] > 0.7).sum().item()
            rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > 0.7).sum().item()
            recall_dict['roi_07'] += roi_recalled
            recall_dict['rcnn_07'] += rcnn_recalled
            roi_recalled = (iou3d_roi.max(dim=0)[0] > 0.5).sum().item()
            rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > 0.5).sum().item()
            recall_dict['roi_05'] += roi_recalled
            recall_dict['rcnn_05'] += rcnn_recalled
            recall_dict['gt'] += cur_gt.shape[0]

            iou3d_rcnn = iou3d_rcnn.max(dim=1)[0]
            gt_iou = iou3d_rcnn

        return gt_iou

    def get_rpn_loss(self, u_cls_preds, u_reg_preds, cls_preds, box_preds, input_dict, dir_cls_preds=None):
        tb_dict = {}
        if cfg.RPN_STAGE.UNET.DECODER_ENABLED and cfg.RPN_STAGE.UNET.PART_SEG_ENABLED:
            # segmentation and part prediction losses
            u_cls_labels, u_reg_labels = input_dict['seg_labels'], input_dict['part_labels']
            u_cls_target = (u_cls_labels > 0).float()
            pos_mask = u_cls_labels > 0
            pos = pos_mask.float()
            neg = (u_cls_labels == 0).float()
            u_cls_weights = pos + neg
            pos_normalizer = pos.sum()
            u_cls_weights = u_cls_weights / torch.clamp(pos_normalizer, min=1.0)
            u_loss_cls = self.rpn_cls_loss_func(u_cls_preds, u_cls_target, weights=u_cls_weights)
            u_loss_cls_pos = (u_loss_cls * pos).sum()
            u_loss_cls_neg = (u_loss_cls * neg).sum()
            u_loss_cls = u_loss_cls.sum()

            loss_unet = u_loss_cls

            if cfg.RPN_STAGE.UNET.PART_REG_ENABLED and pos_normalizer > 0:
                u_loss_reg = F.binary_cross_entropy(torch.sigmoid(u_reg_preds[pos_mask]), u_reg_labels[pos_mask])
                loss_unet += u_loss_reg
                tb_dict['rpn_u_loss_reg'] = u_loss_reg.item()

            # fg_preds = (torch.sigmoid(u_cls_preds) > 0.4).int()
            # fg_mask = (u_cls_target > 0).int()
            # correct = ((fg_preds == u_cls_target.int()).int() * fg_mask).float().sum()
            # union = torch.clamp(fg_mask.float().sum() + fg_preds.float().sum() - correct, min=1.0)
            # fg_iou = correct / union

            # fg_reg_preds = torch.sigmoid(u_reg_preds[pos_mask])
            # fg_reg_labels = u_reg_labels[pos_mask]
            # dis_diff = (fg_reg_preds - fg_reg_labels).abs().sum(dim=0) / (fg_reg_preds.shape[0] + 1e-7)
            # print(fg_iou, dis_diff)

            tb_dict['rpn_u_loss_cls'] = u_loss_cls.item()
            tb_dict['rpn_u_loss_cls_pos'] = u_loss_cls_pos.item()
            tb_dict['rpn_u_loss_cls_neg'] = u_loss_cls_neg.item()
            tb_dict['rpn_loss_unet'] = loss_unet.item()
            tb_dict['rpn_pos_num'] = pos_normalizer.item()
        else:
            loss_unet = 0

        # rpn head losses
        labels, reg_targets = input_dict['labels'], input_dict['reg_targets']

        cared = labels >= 0  # [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)

        box_code_size = self.box_coder.code_size
        batch_size = int(box_preds.shape[0])
        num_class = self.num_class

        cls_targets = cls_targets.squeeze(-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), num_class + 1, dtype=box_preds.dtype,
                                      device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)

        if cfg.RPN_STAGE.ENCODE_BG_AS_ZEROS:
            cls_preds = cls_preds.view(batch_size, -1, num_class)
            one_hot_targets = one_hot_targets[..., 1:]
        else:
            cls_preds = cls_preds.view(batch_size, -1, num_class + 1)

        loss_weights_dict = cfg.RPN_STAGE.RPN_HEAD.LOSS_WEIGHTS
        cls_loss = self.rpn_cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss_reduced = cls_loss.sum() / batch_size
        cls_loss_reduced = cls_loss_reduced * loss_weights_dict['cls_weight']

        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location)
        if cfg.RPN_STAGE.RPN_HEAD.LOSS_REG == 'smooth-l1':
            if cfg.RPN_STAGE.RPN_HEAD.ENCODE_RAD_ERROR_BY_SIN:
                # sin(a - b) = sinacosb-cosasinb
                box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, reg_targets)
                loc_loss = self.rpn_reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            else:
                loc_loss = self.rpn_reg_loss_func(box_preds, reg_targets, weights=reg_weights)  # [N, M]

            loc_loss_reduced = loc_loss.sum() / batch_size
        elif cfg.RPN_STAGE.RPN_HEAD.LOSS_REG == 'bin-based':
            if cfg.RPN_STAGE.RPN_HEAD.ENCODE_RAD_ERROR_BY_SIN:
                box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, reg_targets, dim=box_preds.shape[-1])
                size_angle_loss = self.rpn_reg_loss_func(box_preds_sin[..., -4:],
                                                         reg_targets_sin[..., -4:],
                                                         weights=reg_weights)  # [N, M]
            else:
                size_angle_loss = self.rpn_reg_loss_func(box_preds[..., -4:],
                                                         reg_targets[..., -4:], weights=reg_weights)  # [N, M]

            size_angle_loss = size_angle_loss.sum() / batch_size

            pos_mask = labels > 0
            bin_cfg = cfg.RPN_STAGE.RPN_HEAD.BIN_LOSS_CFG
            center_targets = reg_targets[..., :3]

            # decode center_targets to residual since box_coder has been applied to reg_targets
            batch_anchors = input_dict['anchors']
            wa, la, ha = batch_anchors[:, :, 3], batch_anchors[:, :, 4], batch_anchors[:, :, 5]
            ht = reg_targets[..., 5]
            hg = torch.exp(ht) * ha

            diagonal = torch.sqrt(la ** 2 + wa ** 2)
            center_targets[..., 0] = center_targets[..., 0] * diagonal
            center_targets[..., 1] = center_targets[..., 1] * diagonal
            center_targets[..., 2] = center_targets[..., 2] * ha + ha / 2 - hg / 2

            center_loss = loss_utils.get_binbased_center_loss(
                box_preds[pos_mask][..., :-4], center_targets[pos_mask],
                loc_scope=bin_cfg.LOC_SCOPE, loc_bin_size=bin_cfg.LOC_BIN_SIZE
            )

            loc_loss_reduced = center_loss + size_angle_loss
            tb_dict['rpn_center_loss'] = center_loss.item()
            tb_dict['rpn_size_angle_loss'] = size_angle_loss.item()
        else:
            raise NotImplementedError

        loc_loss_reduced = loc_loss_reduced * loss_weights_dict['loc_weight']

        rpn_loss = loss_unet + loc_loss_reduced + cls_loss_reduced
        tb_dict['rpn_loss_loc'] = loc_loss_reduced.item()
        tb_dict['rpn_loss_cls'] = cls_loss_reduced.item()

        if cfg.RPN_STAGE.RPN_HEAD.USE_DIRECTION_CLASSIFIER:
            dir_targets = self.get_direction_target_v2(input_dict['anchors'], reg_targets,
                                                       dir_offset=cfg.RPN_STAGE.RPN_HEAD.DIR_OFFSET,
                                                       num_bins=cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)

            dir_logits = dir_cls_preds.view(batch_size, -1, cfg.RPN_STAGE.RPN_HEAD.NUM_DIR_BINS)
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.rpn_dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * loss_weights_dict['dir_weight']
            rpn_loss += dir_loss
            tb_dict['rpn_dir_loss_reduced'] = dir_loss.item()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_rcnn_loss(self, rcnn_ret_dict, input_dict):
        code_size = self.rcnn_box_coder.code_size
        rcnn_cls = rcnn_ret_dict['rcnn_cls']
        rcnn_cls_labels = rcnn_ret_dict['rcnn_cls_labels'].float().view(-1)
        reg_valid_mask = rcnn_ret_dict['reg_valid_mask']
        gt_boxes3d_ct = rcnn_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = rcnn_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = rcnn_ret_dict['rcnn_reg']
        roi_boxes3d = rcnn_ret_dict['rois']
        anchor_size = roi_boxes3d[:, :, 3:6].view(-1, 3)

        rcnn_batch_size = rcnn_cls_labels.shape[0]

        rcnn_loss = 0
        LOSS_WEIGHTS = cfg.TRAIN.LOSS_WEIGHTS if 'LOSS_WEIGHTS' in cfg.TRAIN else {}
        # rcnn classification loss
        if cfg.RCNN_STAGE.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels, reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            w = LOSS_WEIGHTS['rcnn_loss_cls'] if 'rcnn_loss_cls' in LOSS_WEIGHTS else 1.0
            rcnn_loss_cls = rcnn_loss_cls * w
        else:
            raise NotImplementedError

        rcnn_loss += rcnn_loss_cls
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}

        if 'LOSS_PAIRWISE_COMPARE' in cfg.RCNN_STAGE and cfg.RCNN_STAGE.LOSS_PAIRWISE_COMPARE:
            batch_size = input_dict['batch_size']
            batch_rcnn_pred = torch.sigmoid(rcnn_cls.view(batch_size, -1))
            batch_iou_labels = rcnn_ret_dict['rcnn_cls_labels'].view(batch_size, -1)

            batch_pred_mat = batch_rcnn_pred.view(batch_size, -1, 1) - batch_rcnn_pred.view(batch_size, 1, -1)
            batch_gt_mat = batch_iou_labels.view(batch_size, -1, 1) - batch_iou_labels.view(batch_size, 1, -1)
            batch_mask = (torch.abs(batch_gt_mat) > 1e-3)

            if batch_mask.sum() > 0:
                pairwise_compare_loss = F.l1_loss(batch_pred_mat[batch_mask], batch_gt_mat[batch_mask])
            else:
                pairwise_compare_loss = rcnn_loss * 0
            w = LOSS_WEIGHTS['rcnn_loss_pairwise_compare'] if 'rcnn_loss_pairwise_compare' in LOSS_WEIGHTS else 1.0
            pairwise_compare_loss = pairwise_compare_loss * w

            rcnn_loss += pairwise_compare_loss
            tb_dict['rcnn_loss_pairwise_compare'] = pairwise_compare_loss.item()

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum == 0:
            # To be consistent with DistributedDataParallel
            # Faked a rcnn_loss to make gradient of regression branch be zero
            temp_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            faked_reg_target = temp_rcnn_reg.detach()
            rcnn_loss_reg = self.rcnn_reg_loss_func(temp_rcnn_reg, faked_reg_target)  # [N, M]
            rcnn_loss_reg = rcnn_loss_reg.sum() / 1.0
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            # if cfg.RCNN_STAGE.REG_2D_BBOX:
            #     temp_rcnn_reg_2d = rcnn_ret_dict['rcnn_reg_2d'].view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            #     faked_reg_target_2d = temp_rcnn_reg_2d.detach()
            #     rcnn_loss_reg_2d = self.rcnn_reg_loss_func(temp_rcnn_reg_2d, faked_reg_target_2d)  # [N, M]
            #     rcnn_loss_reg_2d = rcnn_loss_reg_2d.sum() / 1.0
            #     tb_dict['rcnn_loss_reg_2d'] = rcnn_loss_reg_2d.item()
            #     rcnn_loss_reg += rcnn_loss_reg_2d
        else:
            fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
            if cfg.RCNN_STAGE.LOSS_REG == 'bin-based':
                anchor_size = anchor_size[fg_mask]
                bin_loss_cfg = cfg.RCNN_STAGE.BIN_LOSS_CFG

                loss_loc, loss_angle, loss_size, reg_loss_dict = \
                    self.rcnn_reg_loss_func(
                        fg_rcnn_reg,
                        gt_boxes3d_ct.view(rcnn_batch_size, -1)[fg_mask],
                        loc_scope=bin_loss_cfg['LOC_SCOPE'],
                        loc_bin_size=bin_loss_cfg['LOC_BIN_SIZE'],
                        num_head_bin=bin_loss_cfg['NUM_HEAD_BIN'],
                        anchor_size=anchor_size,
                        get_xz_fine=True, get_y_by_bin=bin_loss_cfg['LOC_Y_BY_BIN'],
                        loc_y_scope=bin_loss_cfg['LOC_Y_SCOPE'],
                        loc_y_bin_size=bin_loss_cfg['LOC_Y_BIN_SIZE'],
                        get_ry_fine=True
                    )

                loss_size = 3 * loss_size  # consistent with old codes
                rcnn_loss_reg = loss_loc + loss_angle + loss_size

                w = LOSS_WEIGHTS['rcnn_loss_reg'] if 'rcnn_loss_reg' in LOSS_WEIGHTS else 1.0
                rcnn_loss_reg = rcnn_loss_reg * w

                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
                tb_dict['rcnn_loss_loc'] = loss_loc.item()
                tb_dict['rcnn_loss_angle'] = loss_angle.item()
                tb_dict['rcnn_loss_size'] = loss_size.item()

                if cfg.RCNN_STAGE.CORNER_LOSS_REGULARIZATION:
                    rcnn_boxes3d = self.rcnn_box_coder.decode_torch(fg_rcnn_reg, fg_roi_boxes3d,
                                                                    anchor_size=anchor_size)
                    loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d, gt_of_rois_src[fg_mask])
                    loss_corner = loss_corner.mean()

                    w = LOSS_WEIGHTS['rcnn_loss_corner'] if 'rcnn_loss_corner' in LOSS_WEIGHTS else 1.0
                    loss_corner = loss_corner * w

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner

            elif cfg.RCNN_STAGE.LOSS_REG == 'smooth-l1':
                rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
                rois_anchor[:, 0:3] = 0
                rois_anchor[:, 6] = 0
                reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size)[fg_mask],
                                                          rois_anchor[fg_mask])
                rcnn_loss_reg = self.rcnn_reg_loss_func(rcnn_reg.view(rcnn_batch_size, -1)[fg_mask].unsqueeze(dim=0),
                                                        reg_targets.unsqueeze(dim=0))  # [N, M]
                rcnn_loss_reg = rcnn_loss_reg.sum() / max(fg_sum, 0)

                w = LOSS_WEIGHTS['rcnn_loss_reg'] if 'rcnn_loss_reg' in LOSS_WEIGHTS else 1.0
                rcnn_loss_reg = rcnn_loss_reg * w

                tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

                if cfg.RCNN_STAGE.CORNER_LOSS_REGULARIZATION:
                    # TODO: can this loss BP?
                    fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                    batch_anchors = fg_roi_boxes3d.clone().detach()
                    roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                    roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                    batch_anchors[:, :, 0:3] = 0
                    rcnn_boxes3d = self.rcnn_box_coder.decode_torch(fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size),
                                                                    batch_anchors).view(-1, code_size)

                    rcnn_boxes3d = box_coder_utils.rotate_pc_along_z_torch(rcnn_boxes3d, (roi_ry + np.pi / 2))
                    rcnn_boxes3d[:, 0:3] += roi_xyz

                    loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                    loss_corner = loss_corner.mean()

                    w = LOSS_WEIGHTS['rcnn_loss_corner'] if 'rcnn_loss_corner' in LOSS_WEIGHTS else 1.0
                    loss_corner = loss_corner * w

                    rcnn_loss_reg += loss_corner
                    tb_dict['rcnn_loss_corner'] = loss_corner
            else:
                raise NotImplementedError

            # if cfg.RCNN_STAGE.REG_2D_BBOX:
            #     assert cfg.DATASET == 'kitti'
            #     batch_rect = input_dict['rect']
            #     batch_Trv2c = input_dict['Trv2c']
            #     batch_P2 = input_dict['P2']
            #     batch_imgshape = input_dict['image_shape']
            #     batch_size = batch_rect.shape[0]
            #     with torch.no_grad():
            #         gt_boxes3d = gt_of_rois_src.view(batch_size, -1, gt_of_rois_src.shape[-1])
            #         gt_bbox_list, roi_bbox_list, reg_2d_valid_mask_list = [], [], []
            #         for k in range(batch_size):
            #             cur_gt_bbox = kitti_utils.boxes3d_to_bbox(gt_boxes3d[k], P2=batch_P2[k], r_rect=batch_rect[k],
            #                                                       velo2cam=batch_Trv2c[k], box_type='lidar')
            #             cur_roi_bbox = kitti_utils.boxes3d_to_bbox(roi_boxes3d[k], P2=batch_P2[k], r_rect=batch_rect[k],
            #                                                        velo2cam=batch_Trv2c[k], box_type='lidar')
            #
            #             max_width, max_length = batch_imgshape[k][1], batch_imgshape[k][0]
            #
            #             margin_thresh = 5  # pixel
            #             cur_roi_valid_mask = (cur_roi_bbox[:, 0] > -margin_thresh) & (cur_roi_bbox[:, 1] > -margin_thresh) \
            #                     & (cur_roi_bbox[:, 2] < max_width + margin_thresh) & (cur_roi_bbox[:, 3] < max_length + margin_thresh)
            #             cur_gt_valid_mask = (cur_gt_bbox[:, 0] > -margin_thresh) & (cur_gt_bbox[:, 1] > -margin_thresh) \
            #                     & (cur_gt_bbox[:, 2] < max_width + margin_thresh) & (cur_gt_bbox[:, 3] < max_length + margin_thresh)
            #
            #             gt_bbox_list.append(cur_gt_bbox)
            #             roi_bbox_list.append(cur_roi_bbox)
            #             reg_2d_valid_mask_list.append(cur_roi_valid_mask & cur_gt_valid_mask)
            #
            #         gt_bbox = torch.cat(gt_bbox_list, dim=0)
            #         roi_bbox = torch.cat(roi_bbox_list, dim=0)
            #         reg_2d_valid_mask = torch.cat(reg_2d_valid_mask_list, dim=0)
            #
            #         reg_2d_mask = reg_2d_valid_mask & fg_mask
            #
            #     fg_2d_sum = reg_2d_mask.sum()
            #     if fg_2d_sum == 0:
            #         temp_rcnn_reg_2d = rcnn_ret_dict['rcnn_reg_2d'].view(rcnn_batch_size, -1)[0].unsqueeze(dim=0)
            #         faked_reg_target_2d = temp_rcnn_reg_2d.detach()
            #         rcnn_loss_reg_2d = self.rcnn_reg_loss_func(temp_rcnn_reg_2d, faked_reg_target_2d)  # [N, M]
            #         rcnn_loss_reg_2d = rcnn_loss_reg_2d.sum() / 1.0
            #         tb_dict['rcnn_loss_reg_2d'] = rcnn_loss_reg_2d.item()
            #         rcnn_loss_reg += rcnn_loss_reg_2d
            #     else:
            #         fg_reg_targets_2d = box_coder_utils.bbox2delta(roi_bbox[reg_2d_mask], gt_bbox[reg_2d_mask])
            #         fg_rcnn_reg_2d = rcnn_ret_dict['rcnn_reg_2d'].view(rcnn_batch_size, -1)[reg_2d_mask]
            #         rcnn_loss_reg_2d = self.rcnn_reg_loss_func(fg_rcnn_reg_2d, fg_reg_targets_2d)  # [N, M]
            #
            #         rcnn_loss_reg_2d = rcnn_loss_reg_2d.sum() / max(fg_2d_sum, 0)
            #         tb_dict['rcnn_loss_reg_2d'] = rcnn_loss_reg_2d.item()
            #         rcnn_loss_reg += rcnn_loss_reg_2d

        rcnn_loss += rcnn_loss_reg
        tb_dict['rcnn_loss'] = rcnn_loss
        return rcnn_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim+1]) * torch.cos(boxes2[..., dim:dim+1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim+1]) * torch.sin(boxes2[..., dim:dim+1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim+1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim+1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        dir_cls_targets = (rot_gt > 0).long()
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), 2, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    @staticmethod
    def get_direction_target_v2(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

