import torch
import spconv
import torch.nn.functional as F
from .detector3d import Detector3D
from ..model_utils.proposal_layer import proposal_layer
from ...config import cfg


class PartA2Net(Detector3D):
    def __init__(self, num_class, dataset):
        super().__init__(num_class, dataset)

        self.sparse_shape = dataset.voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.build_networks(cfg.MODEL)
        self.build_losses(cfg.MODEL.LOSSES)

    def forward_rpn(self, voxels, num_points, coords, batch_size, voxel_centers, **kwargs):
        # RPN inference
        with torch.set_grad_enabled((not cfg.MODEL.RPN.PARAMS_FIXED) and self.training):
            voxel_features = self.vfe(
                features=voxels,
                num_voxels=num_points,
                coords=coords
            )

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=coords,
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )

            unet_ret_dict = self.rpn_net(input_sp_tensor, voxel_centers)
            rpn_preds_dict = self.rpn_head(unet_ret_dict['spatial_features'])

        rpn_ret_dict = {
            'rpn_cls_preds': rpn_preds_dict['cls_preds'],
            'rpn_box_preds': rpn_preds_dict['box_preds'],
            'rpn_dir_cls_preds': rpn_preds_dict.get('dir_cls_preds', None),
            'rpn_seg_scores': torch.sigmoid(rpn_preds_dict['u_cls_preds'].view(-1)),
            'rpn_seg_features': rpn_preds_dict['seg_features'],
            'rpn_bev_features': unet_ret_dict['spatial_features'],
            'rpn_part_offsets': torch.sigmoid(unet_ret_dict['u_reg_preds'])
        }
        return rpn_ret_dict

    def forward_rcnn(self, batch_anchors, batch_size, voxel_centers, coords, rpn_ret_dict, input_dict):
        with torch.no_grad():
            batch_anchors = batch_anchors.view(batch_size, -1, batch_anchors.shape[-1])  # (B, N, 7 + ?)
            num_anchors = batch_anchors.shape[1]
            batch_cls_preds = rpn_ret_dict['rpn_cls_preds'].view(batch_size, num_anchors, -1)
            batch_box_preds = self.box_coder.decode_with_head_direction_torch(
                box_preds=rpn_ret_dict['rpn_box_preds'].view(batch_size, num_anchors, -1),
                anchors=batch_anchors,
                dir_cls_preds=rpn_ret_dict.get('rpn_dir_cls_preds', None),
                num_dir_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('num_direction_bins', None),
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_offset', None),
                dir_limit_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_limit_offset', None)
            )

            roi_dict = proposal_layer(
                batch_size, batch_cls_preds, batch_box_preds,
                code_size=self.box_coder.code_size, mode=self.mode
            )

        rcnn_input_dict = {
            'voxel_centers': voxel_centers,
            'coordinates': coords,
            'rpn_seg_scores': rpn_ret_dict['rpn_seg_scores'],
            'rpn_seg_features': rpn_ret_dict['rpn_seg_features'],
            'rpn_bev_features': rpn_ret_dict['rpn_bev_features'],
            'rpn_part_offsets': rpn_ret_dict['rpn_part_offsets'],
            'rois': roi_dict['rois'],
            'roi_raw_scores': roi_dict['roi_raw_scores'],
            'roi_labels': roi_dict['roi_labels'],
            'gt_boxes': input_dict.get('gt_boxes', None)
        }

        rcnn_ret_dict = self.rcnn_net.forward(rcnn_input_dict)
        return rcnn_ret_dict

    def forward(self, input_dict):
        batch_anchors = input_dict['anchors']
        batch_size = batch_anchors.shape[0]
        coords = input_dict['coordinates'].int()
        voxel_centers = input_dict['voxel_centers']

        rpn_ret_dict = self.forward_rpn(**input_dict)
        if cfg.MODEL.RCNN.ENABLED:
            rcnn_ret_dict = self.forward_rcnn(
                batch_anchors, batch_size, voxel_centers, coords, rpn_ret_dict, input_dict
            )
        else:
            rcnn_ret_dict = None

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(rpn_ret_dict, rcnn_ret_dict, input_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict, input_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, rpn_ret_dict, rcnn_ret_dict, input_dict):
        loss = 0
        tb_dict = {}
        disp_dict = {}
        if not cfg.MODEL.RPN.PARAMS_FIXED:
            rpn_loss, rpn_tb_dict = self.get_rpn_loss(
                u_cls_preds=rpn_ret_dict['u_cls_preds'],
                u_reg_preds=rpn_ret_dict['u_reg_preds'],
                rpn_cls_preds=rpn_ret_dict['rpn_cls_preds'],
                rpn_box_preds=rpn_ret_dict['rpn_box_preds'],
                rpn_dir_cls_preds=rpn_ret_dict['rpn_dir_cls_preds'],
                input_dict=input_dict
            )
            loss += rpn_loss
            tb_dict.update(rpn_tb_dict)

        if not cfg.MODEL.RCNN.ENABLED:
            # RCNN loss
            rcnn_loss, rcnn_tb_dict = self.get_rcnn_loss(rcnn_ret_dict)
            loss += rcnn_loss
            tb_dict.update(rcnn_tb_dict)

            # logging to tensorboard
            rcnn_cls_labels = rcnn_ret_dict['rcnn_cls_labels'].float().view(-1)
            fg_thresh = min(cfg.MODEL.RCNN.ROI_SAMPLER.REG_FG_THRESH, cfg.MODEL.RCNN.ROI_SAMPLER.CLS_FG_THRESH)
            fg_num = (rcnn_cls_labels > fg_thresh).sum().item()
            bg_num = (rcnn_cls_labels == 0).sum().item()
            tb_dict['rcnn_fg_num'] = fg_num
            tb_dict['rcnn_bg_num'] = bg_num

            disp_dict['rcnn_fg_num'] = fg_num

        return loss, tb_dict, disp_dict

    def get_rpn_loss(self, u_cls_preds, u_reg_preds, rpn_cls_preds, rpn_box_preds, rpn_dir_cls_preds, input_dict):
        loss_unet, tb_dict_1 = self.get_unet_loss(
            u_cls_preds=u_cls_preds, u_reg_preds=u_reg_preds, input_dict=input_dict
        )
        loss_anchor_box, tb_dict_2 = self.get_anchor_box_loss(
            cls_preds=rpn_cls_preds, box_preds=rpn_box_preds,
            box_cls_labels=input_dict['box_cls_labels'],
            box_reg_targets=input_dict['box_reg_targets'],
            box_dir_cls_preds=rpn_dir_cls_preds, anchors=input_dict['anchors']
        )
        loss_rpn = loss_unet + loss_anchor_box,
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict_1,
            **tb_dict_2
        }
        return loss_rpn, tb_dict

    def get_unet_loss(self, u_cls_preds, u_reg_preds, input_dict):
        tb_dict = {}

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

        if pos_normalizer > 0:
            u_loss_reg = F.binary_cross_entropy(torch.sigmoid(u_reg_preds[pos_mask]), u_reg_labels[pos_mask])
            loss_unet += u_loss_reg
            tb_dict['rpn_u_loss_reg'] = u_loss_reg.item()

        tb_dict['rpn_loss_u_cls'] = u_loss_cls.item()
        tb_dict['rpn_loss_u_cls_pos'] = u_loss_cls_pos.item()
        tb_dict['rpn_loss_u_cls_neg'] = u_loss_cls_neg.item()
        tb_dict['rpn_loss_unet'] = loss_unet.item()
        tb_dict['rpn_pos_num'] = pos_normalizer.item()

        return loss_unet, tb_dict
