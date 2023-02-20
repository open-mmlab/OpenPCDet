import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.cuda_slicer import cuda_slicer

from sbnet.layers import SparseBlock_Conv2d_BN_ReLU
import time

class CenterHeadGroupSbnet(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.tcount=self.model_cfg.TILE_COUNT # There should be a better way but its fine for now
        post_process_cfg = self.model_cfg.POST_PROCESSING
        self.max_obj_per_sample = post_process_cfg.MAX_OBJ_PER_SAMPLE

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        ksize=3

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        num_heads = len(self.class_names_each_head)
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), \
                f'class_names_each_head={self.class_names_each_head}'
        use_bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        # This is not a group or merged conv
        self.shared_conv = SparseBlock_Conv2d_BN_ReLU(input_channels,
                self.model_cfg.SHARED_CONV_CHANNEL, ksize, stride=1, 
                bias=use_bias, bn_eps=1e-05, bn_momentum=0.1,
                bcount=self.tcount, transpose=True)

        #############
        # We are not going to construct seperate heads. Instead, we will utilize group convolution to merge
        # all heads. However, the heatmaps has to be merged seperately, as their output can be used to
        # slice the input of other convolutions.
        #############

        hm_list = []
        inp_channels = self.model_cfg.SHARED_CONV_CHANNEL
        outp_channels, groups = inp_channels * num_heads, 1
        for k in range(self.model_cfg.NUM_HM_CONV - 1):
            hm_list.append(nn.Conv2d(inp_channels, outp_channels, kernel_size=ksize,
                    stride=1, padding=1, groups=groups, bias=use_bias))
            hm_list.append(nn.GroupNorm(num_heads, outp_channels))
            hm_list.append(nn.ReLU())
            if k == 0:
                inp_channels = outp_channels
                groups = num_heads

        if self.model_cfg.NUM_HM_CONV <= 1:
            outp_channels = inp_channels

        hm_outp_channels = [len(cur_class_names) for cur_class_names in self.class_names_each_head]
        hm_max_ch = max(hm_outp_channels)
        hm_total_outp_channels = hm_max_ch * num_heads
        hm_list.append(nn.Conv2d(outp_channels, hm_total_outp_channels, kernel_size=ksize,
            stride=1, padding=1, groups=groups, bias=True))
        hm_list[-1].bias.data.fill_(-2.19)

        #TODO simple hack for now
        # out_C x in_C x ksize x ksize
        #hm_list[-1].weight.data[[1,7],...] = 0.
        #hm_list[-1].bias.data[[1,7],...] = 0.

        if len(hm_list) > 1:
            self.heatmap_convs = nn.Sequential(*hm_list)
        else:
            self.heatmap_convs = hm_list[0]
        self.heatmap_outp_inds = [(i*hm_max_ch, i*hm_max_ch+ch) \
                for i, ch in enumerate(hm_outp_channels)]
        # The channels that fall outside of these indices are useless, don't train their filters
        print('Heatmap output indices:', self.heatmap_outp_inds)

        # Now, we need seperate merged conv and group conv for each head because their inputs
        # will be different.
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        head_dict = self.separate_head_cfg.HEAD_DICT
        num_convs = [v['num_conv'] for v in head_dict.values()]
        assert all([num_convs[0] == nc for nc in num_convs])
        num_convs = num_convs[0]

        attr_outp_channels = [v['out_channels'] for v in head_dict.values()]
        attr_max_ch = max(attr_outp_channels)
        attr_total_outp_channels = attr_max_ch * len(head_dict)

        self.det_heads=nn.ModuleList()
        attr_conv_names = self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER
        self.slice_size = 1
        for head_idx in range(num_heads):
            inp_channels = self.model_cfg.SHARED_CONV_CHANNEL
            outp_channels, groups = inp_channels * len(head_dict), 1
            attr_list = []

            for k in range(num_convs - 1):
                attr_list.append(nn.Conv2d(inp_channels, outp_channels, kernel_size=ksize,
                        stride=1, padding=0, groups=groups, bias=use_bias))
                attr_list.append(nn.GroupNorm(len(head_dict), outp_channels))
                attr_list.append(nn.ReLU())
                if k == 0:
                    inp_channels = outp_channels
                    groups = len(head_dict)
                if head_idx == 0:
                    self.slice_size = (self.slice_size-1) + ksize

            if num_convs <= 1:
                outp_channels = inp_channels

            attr_list.append(nn.Conv2d(outp_channels, attr_total_outp_channels,
                kernel_size=ksize, stride=1, padding=0, groups=groups, bias=True))

            attr_convs = nn.Sequential(*attr_list)
            if head_idx == 0:
                self.slice_size = (self.slice_size-1) + ksize

            for m in attr_convs:
                if isinstance(m, nn.Conv2d):
                    kaiming_normal_(m.weight.data)
                    if hasattr(m, "bias") and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            #TODO simple hack for now
            #attr_convs[-1].weight.data[[2,4,5,11,14],...] = 0.
            #attr_convs[-1].bias.data[[2,4,5,11,14],...] = 0.

            self.det_heads.append(attr_convs)

        name_ind_dict = {}
        for conv_idx, conv_name in enumerate(attr_conv_names):
            tmp1 = conv_idx * attr_max_ch
            # BEWARE OF ORDER
            name_ind_dict[conv_name] = (tmp1, tmp1 + attr_outp_channels[conv_idx])
        self.attr_outp_inds = name_ind_dict

        print('Attribute output indices:', self.attr_outp_inds)
        #############

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    # give topk_outps to this guy if it is available!
    def generate_predicted_boxes_eval(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            # this loop runs only once for kitti but multiple times for nuscenes (single vs multihead)
            if 'topk_outp' in pred_dict:
                batch_hm = pred_dict['hm']
            else:
                batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = [d.exp() for d in pred_dict['dim']]

            # This part might not work
            #batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            #batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_rot_cos = [r[..., 0].unsqueeze(dim=-1) for r in pred_dict['rot']]
            batch_rot_sin = [r[..., 1].unsqueeze(dim=-1) for r in pred_dict['rot']]

            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap_sliced(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
                topk_outp=(pred_dict['topk_outp'] if 'topk_outp' in pred_dict else None)
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict


    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        if not self.training:
            return self.forward_eval(data_dict)
        else:
            return self.forward_train(data_dict)

    def forward_train(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        data_dict['sbnet_x'] = spatial_features_2d
        data_dict = self.shared_conv(data_dict)
        shr_conv_outp = data_dict['sbnet_y']

        # Change is to NCHW
        shr_conv_outp_nchw = shr_conv_outp.permute(0,3,1,2).contiguous()
        heatmaps = self.heatmap_convs(shr_conv_outp_nchw)

        pred_dicts = [{'hm' : heatmaps[:, inds[0]:inds[1]]} \
                for inds in self.heatmap_outp_inds]

        # default padding is 1
        pad_size = p = 1

        # Forward through all heads one by one
        for det_head, pd in zip(self.det_heads, pred_dicts):
            x = shr_conv_outp_nchw
            for m in det_head:
                if isinstance(m, nn.Conv2d):
                    x = torch.nn.functional.pad(x, (p,p,p,p))
                x = m(x)

            for name, inds in self.attr_outp_inds.items():
                pd[name] = x[:, inds[0]:inds[1], :, :].contiguous()

        feature_map_size=spatial_features_2d.size()[1:3]
        target_dict = self.assign_targets(
            data_dict['gt_boxes'], feature_map_size=feature_map_size,
            feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
        )
        self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )
            rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
            data_dict['rois'] = rois
            data_dict['roi_scores'] = roi_scores
            data_dict['roi_labels'] = roi_labels
            data_dict['has_class_labels'] = True

        return data_dict

    def forward_eval(self, data_dict):
        data_dict['sbnet_x'] = data_dict['spatial_features_2d']

        data_dict = self.shared_conv(data_dict)
        shr_conv_outp = data_dict['sbnet_y']

        # Run heatmap convolutions and gather the actual channels
        shr_conv_outp_nchw = shr_conv_outp.permute(0,3,1,2).contiguous()
        heatmaps = self.sigmoid(self.heatmap_convs(shr_conv_outp_nchw))
        # Convert to NCHW
        pred_dicts = [{'hm' : heatmaps[:, inds[0]:inds[1]]} \
                for inds in self.heatmap_outp_inds]

        ##########
        # For each heatmap, do slicing and forwarding
        # Every head has a pred dict,
        # TODO Definetely try cuda streams here
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thres = post_process_cfg.SCORE_THRESH
        p = pad_size = self.slice_size//2
        padded_x = torch.nn.functional.pad(shr_conv_outp, (0,0,p,p,p,p))
        for det_head, pd in zip(self.det_heads, pred_dicts):
            heatmap = pd['hm']
            topk_score, topk_inds, topk_classes, topk_ys, topk_xs = \
                    centernet_utils._topk(heatmap, K=self.max_obj_per_sample)

            final_outputs, topk_outp, batch_id_tensors = [], [], []

            masks = topk_score > score_thres
            # This loop runs for each batch
            for scores, inds, classes, ys, xs, mask in zip(\
                    topk_score, topk_inds, topk_classes, topk_ys, topk_xs, masks):
                # IDK if this mask ig going to make it slower a lot
                scores, inds, classes, ys, xs = \
                        scores[mask], inds[mask], classes[mask], ys[mask], xs[mask]
                batch_id_tensors.append(torch.full((scores.size(0),),
                    len(batch_id_tensors), dtype=torch.short, device=scores.device))
                topk_outp.append((scores, inds, classes, ys, xs))

            pd['topk_outp'] = topk_outp

            # Slice for all batches
            num_slc_per_batch = [t.size(0) for t in batch_id_tensors]
            b_id_cat = torch.cat(batch_id_tensors)
            ys_cat = torch.cat([to[3] for to in topk_outp]) + pad_size
            xs_cat = torch.cat([to[4] for to in topk_outp]) + pad_size
            indices = torch.stack((b_id_cat, ys_cat.short(), xs_cat.short()), dim=1)
            slices = cuda_slicer.slice_and_batch_nhwc(padded_x, indices, self.slice_size)
            outp = det_head(slices)

            # finally, split the output according to the batches they belong
            for name, inds in self.attr_outp_inds.items():
                #turn it to (num_slices x C) from (num_slices x C x 1 x 1)
                outp_slices = outp[:, inds[0]:inds[1]].flatten(-3)
                outp_slices_split, idx = [], 0
                for num_slc in num_slc_per_batch:
                    outp_slices_split.append(outp_slices[idx:(idx+num_slc),:])
                    idx += num_slc
                pd[name] = outp_slices_split

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        pred_dicts = self.generate_predicted_boxes_eval(
            data_dict['batch_size'], pred_dicts
        )
        data_dict['final_box_dicts'] = pred_dicts

        return data_dict

    def calibrate(self, data_dict):
        return self(data_dict) # just forward it
