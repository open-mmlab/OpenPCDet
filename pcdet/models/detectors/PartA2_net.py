import torch
import spconv
from .detector3d import Detector3D
from ..model_utils.proposal_layer import proposal_layer
from ...config import cfg


class PartA2Net(Detector3D):
    def __init__(self, num_class, dataset):
        super().__init__(num_class, dataset)

        self.sparse_shape = dataset.voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.build_networks(cfg.MODEL)

    def forward_rpn(self, voxels, num_points, coordinates, batch_size, **kwargs):
        # RPN inference
        with torch.set_grad_enabled((not cfg.MODEL.RPN.PARAMS_FIXED) and self.training):
            voxel_features = self.vfe(
                features=voxels,
                num_voxels=num_points,
                coords=coordinates
            )

            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=coordinates,
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )

            unet_ret_dict = self.rpn_net(
                input_sp_tensor,
                **kwargs
            )

            rpn_preds_dict = self.rpn_head(
                unet_ret_dict['spatial_features'],
                **{'gt_boxes': kwargs.get('gt_boxes', None)}
            )
            rpn_preds_dict.update(unet_ret_dict)

        rpn_ret_dict = {
            'rpn_cls_preds': rpn_preds_dict['cls_preds'],
            'rpn_box_preds': rpn_preds_dict['box_preds'],
            'rpn_dir_cls_preds': rpn_preds_dict.get('dir_cls_preds', None),
            'rpn_seg_features': rpn_preds_dict['seg_features'],
            'rpn_bev_features': rpn_preds_dict['spatial_features'],
            'u_cls_preds': rpn_preds_dict['u_seg_preds'].view(-1),
            'u_reg_preds': rpn_preds_dict['u_reg_preds'],
            'rpn_seg_scores': torch.sigmoid(rpn_preds_dict['u_seg_preds'].view(-1)),
            'rpn_part_offsets': torch.sigmoid(rpn_preds_dict['u_reg_preds']),
            'anchors': rpn_preds_dict['anchors']
        }

        if self.training:
            rpn_ret_dict.update({
                'box_cls_labels': rpn_preds_dict['box_cls_labels'],
                'box_reg_targets': rpn_preds_dict['box_reg_targets'],
                'reg_src_targets': rpn_preds_dict['reg_src_targets'],
                'reg_weights': rpn_preds_dict['reg_weights'],
            })
        return rpn_ret_dict

    def forward_rcnn(self, batch_anchors, batch_size, voxel_centers, coords, rpn_ret_dict, input_dict):
        with torch.no_grad():
            batch_anchors = batch_anchors.view(batch_size, -1, batch_anchors.shape[-1])  # (B, N, 7 + ?)
            num_anchors = batch_anchors.shape[1]
            batch_cls_preds = rpn_ret_dict['rpn_cls_preds'].view(batch_size, num_anchors, -1)

            batch_box_preds = self.rpn_head.box_coder.decode_with_head_direction_torch(
                box_preds=rpn_ret_dict['rpn_box_preds'].view(batch_size, num_anchors, -1),
                anchors=batch_anchors,
                dir_cls_preds=rpn_ret_dict.get('rpn_dir_cls_preds', None),
                num_dir_bins=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('num_direction_bins', None),
                dir_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_offset', None),
                dir_limit_offset=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('dir_limit_offset', None),
                use_binary_dir_classifier=cfg.MODEL.RPN.RPN_HEAD.ARGS.get('use_binary_dir_classifier', False)
            )

            roi_dict = proposal_layer(
                batch_size, batch_cls_preds, batch_box_preds,
                code_size=self.rpn_head.box_coder.code_size, mode=self.mode
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
        batch_size = input_dict['batch_size']
        coords = input_dict['coordinates'].int()
        voxel_centers = input_dict['voxel_centers']

        rpn_ret_dict = self.forward_rpn(**input_dict)
        if cfg.MODEL.RCNN.ENABLED:
            anchors = rpn_ret_dict['anchors']
            anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
            rcnn_ret_dict = self.forward_rcnn(
                anchors,
                batch_size, voxel_centers, coords, rpn_ret_dict, input_dict
            )
        else:
            rcnn_ret_dict = None

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(rcnn_ret_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.predict_boxes(rpn_ret_dict, rcnn_ret_dict, input_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, rcnn_ret_dict):
        loss = 0
        tb_dict = {}
        disp_dict = {}
        if not cfg.MODEL.RPN.PARAMS_FIXED:
            loss_unet, tb_dict_1 = self.rpn_net.get_loss()
            loss_anchor_box, tb_dict_2 = self.rpn_head.get_loss()
            loss_rpn = loss_unet + loss_anchor_box
            rpn_tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict_1,
                **tb_dict_2
            }

            loss += loss_rpn
            tb_dict.update(rpn_tb_dict)

        if cfg.MODEL.RCNN.ENABLED:
            # RCNN loss
            rcnn_loss, rcnn_tb_dict = self.rcnn_net.get_loss()
            loss += rcnn_loss
            tb_dict.update(rcnn_tb_dict)

            # logging to tensorboard
            rcnn_cls_labels = rcnn_ret_dict['rcnn_cls_labels'].float().view(-1)
            fg_thresh = min(cfg.MODEL.RCNN.TARGET_CONFIG.REG_FG_THRESH, cfg.MODEL.RCNN.TARGET_CONFIG.CLS_FG_THRESH)
            fg_num = (rcnn_cls_labels > fg_thresh).sum().item()
            bg_num = (rcnn_cls_labels == 0).sum().item()
            tb_dict['rcnn_fg_num'] = fg_num
            tb_dict['rcnn_bg_num'] = bg_num

            disp_dict['rcnn_fg_num'] = fg_num

        return loss, tb_dict, disp_dict

