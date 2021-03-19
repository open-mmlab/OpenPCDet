import copy
import numpy as np
import torch
from torch import nn
from ...utils import loss_utils
from .target_assigner.center_assigner import CenterAssigner
import numba
from ...ops.iou3d_nms.iou3d_nms_utils import nms_gpu, nms_normal_gpu
import pdb


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SepHead(nn.Module):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, init_bias=-2.19, bn=False, init=True,
                 **kwargs):
        super(SepHead, self).__init__()

        self.heads = heads  # {cat: [classes, num_conv]}
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    nn.Conv2d(c_in, head_conv, final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if bn:
                    conv_layers.append(nn.BatchNorm2d(head_conv))
                conv_layers.append(nn.ReLU())
                c_in = head_conv

            conv_layers.append(
                nn.Conv2d(head_conv, classes, final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)
        if init:
            self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'hm':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


# TODO
class DCNSepHead(nn.Module):
    r"""DCNSeperateHead for CenterHead.

    .. code-block:: none
            /-----> DCN for heatmap task -----> heatmap task.
    feature
            \-----> DCN for regression tasks -----> regression tasks

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        dcn_config (dict): Config of dcn layer.
        num_cls (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 num_cls,
                 heads,
                 dcn_config,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 **kwargs):
        super(DCNSepHead, self).__init__()
        if 'hm' in heads:
            heads.pop('hm')
        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = build_conv_layer(dcn_config)

        self.feature_adapt_reg = build_conv_layer(dcn_config)

        # heatmap prediction head
        cls_head = [
            ConvModule(
                in_channels,
                head_conv,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                bias=bias,
                norm_cfg=norm_cfg),
            build_conv_layer(
                conv_cfg,
                head_conv,
                num_cls,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
        ]
        self.cls_head = nn.Sequential(*cls_head)
        self.init_bias = init_bias
        # other regression target
        self.task_head = SepHead(
            in_channels,
            heads,
            head_conv=head_conv,
            final_kernel=final_kernel,
            bias=bias)

    def init_weights(self):
        """Initialize weights."""
        self.cls_head[-1].bias.data.fill_(self.init_bias)
        self.task_head.init_weights()

    def forward(self, x):
        """Forward function for DCNSepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -hm (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


class CenterHead(nn.Module):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.

    """

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super(CenterHead, self).__init__()

        self.model_cfg = model_cfg
        self.post_cfg = model_cfg.TEST_CONFIG
        self.in_channels = input_channels
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training
        self.dataset = model_cfg.DATASET

        self.num_classes = [len(t['class_names']) for t in model_cfg.TASKS]  # task number
        self.class_names = [t['class_names'] for t in model_cfg.TASKS]
        self.forward_ret_dict = {}  # return dict filtered by assigner
        self.code_weights = self.model_cfg.LOSS_CONFIG.code_weights  # weights between different heads
        self.weight = self.model_cfg.LOSS_CONFIG.weight  # weight between local loss and hm loss
        self.no_log = self.model_cfg.get('NO_LOG', False)
        self.init = self.model_cfg.get('INIT', True)

        # a shared convolution
        share_conv_channel = model_cfg.PARAMETERS.share_conv_channel
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)  # change input data
        )

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.init_bias = model_cfg.PARAMETERS.init_bias
        self.task_heads = nn.ModuleList()

        self.use_dcn = model_cfg.USE_DCN
        for num_cls in self.num_classes:
            heads = copy.deepcopy(self.common_heads)
            # need to complete
            if self.use_dcn:
                self.task_heads.append(
                    DCNSepHead(share_conv_channel, heads, final_kernel=3, bn=True, init_bias=self.init_bias))
            else:
                heads.update(dict(hm=(num_cls, 2)))
                self.task_heads.append(
                    SepHead(share_conv_channel, heads, final_kernel=3, bn=True, init_bias=self.init_bias,
                            init=self.init))

        self.target_assigner = CenterAssigner(
            model_cfg.TARGET_ASSIGNER_CONFIG,
            self.num_class,
            self.no_log,
            self.grid_size,
            self.point_cloud_range,
            self.voxel_size,
            self.dataset
        )
        self.build_loss()

    def assign_target(self, gt_boxes):
        """

        Args:
            gt_boxes: (B, M, 8)

        Returns:

        """
        targets_dict = self.target_assigner.assign_targets((gt_boxes))

        return targets_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        if self.shared_conv is not None:
            spatial_features_2d = self.shared_conv(spatial_features_2d)

        multi_head_features = []
        for task in self.task_heads:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features

        # there is something ambiguous that need to be understood
        if self.training:
            targets_dict = self.assign_target(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            data_dict = self.generate_predicted_boxes(data_dict)

        return data_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        pred_dicts = self.forward_ret_dict['multi_head_features']
        center_loss = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.clip_sigmoid(pred_dict['hm'])

            target_box_encoding = self.forward_ret_dict['box_encoding'][task_id]
            if self.dataset == 'nuscenes':
                # nuscenes encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['height'],
                    pred_dict['dim'],
                    pred_dict['rot'],
                    pred_dict['vel']
                ], dim=1).contiguous()  # (B, 10, H, W)
            elif self.dataset == 'waymo':
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['height'],
                    pred_dict['dim'],
                    pred_dict['rot']
                ], dim=1).contiguous()  # (B, 8, H, W)
            else:
                raise NotImplementedError("Only Support KITTI and nuScene for Now!")

            self.forward_ret_dict['pred_box_encoding'][task_id] = pred_box_encoding

            hm_loss = self.crit(pred_dict['hm'], self.forward_ret_dict['heatmap'][task_id],
                                self.forward_ret_dict['mask'][task_id],
                                self.forward_ret_dict['ind'][task_id],
                                self.forward_ret_dict['cat'][task_id])

            box_loss = self.crit_reg(
                pred_box_encoding,
                target_box_encoding,
                self.forward_ret_dict['mask'][task_id],
                self.forward_ret_dict['ind'][task_id]
            )
            # local offset loss
            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            loss = hm_loss + self.weight * loc_loss
            center_loss.append(loss)
            # TODO: update tbdict

        return sum(center_loss), tb_dict

    def build_loss(self):
        # criterion
        self.add_module(
            'crit', loss_utils.CenterNetFocalLossV2()
        )
        self.add_module(
            'crit_reg', loss_utils.CenterNetRegLoss()
        )

    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        """
        Generate box predictions with decode, topk and circular nms
        used in self.forward

        Args:
            data_dict:

        Returns:

        """
        self.double_flip = not self.training and self.post_cfg.get('double_flip', False)  # type: bool
        pred_dicts = self.forward_ret_dict['multi_head_features']  # output of forward func.

        task_preds = {}
        task_preds['bboxes'] = {}
        task_preds['scores'] = {}
        task_preds['labels'] = {}

        for task_id, pred_dict in enumerate(pred_dicts):
            batch_size = pred_dict['hm'].shape[
                0]  # can't use data_dict['batch_size'], because it will change after double flip
            # must change dataset.py and waymo.py, batchsize = batchsize * 4
            if self.double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                batch_hm, batch_rot_sine, batch_rot_cosine, batch_height, batch_dim, batch_vel, batch_reg = self._double_flip(
                    pred_dict, batch_size)
                # convert data_dict format
                data_dict['batch_size'] = batch_size
            else:
                batch_hm = pred_dict['hm'].sigmoid()
                batch_reg = pred_dict['reg']
                batch_height = pred_dict['height']
                if not self.no_log:
                    batch_dim = torch.exp(pred_dict['dim'])
                    # clamp for good init, otherwise it will goes inf with exp
                    # batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
                else:
                    batch_dim = pred_dict['dim']
                batch_rot_sine = pred_dict['rot'][:, 0].unsqueeze(1)
                batch_rot_cosine = pred_dict['rot'][:, 1].unsqueeze(1)

                batch_vel = pred_dict['vel'] if self.dataset == 'nuscenes' else None

            # decode
            boxes = self.proposal_layer(batch_hm, batch_rot_sine, batch_rot_cosine, batch_height, batch_dim, batch_vel,
                                        reg=batch_reg, cfg=self.post_cfg, task_id=task_id)

            task_preds['bboxes'][task_id] = [box['bboxes'] for box in boxes]
            task_preds['scores'][task_id] = [box['scores'] for box in boxes]
            task_preds['labels'][task_id] = [box['labels'] for box in boxes]  # labels are local here

        pred_dicts = []
        nms_cfg = self.post_cfg.nms
        num_rois = nms_cfg.nms_pre_max_size * self.num_class
        batch_size = len(task_preds['bboxes'][0])
        for batch_idx in range(batch_size):
            # Initially, i write this in next loop, this will cause other class won't be detect and accuracy drop
            offset = 1
            final_bboxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                final_bboxes.append(task_preds['bboxes'][task_id][batch_idx])
                final_scores.append(task_preds['scores'][task_id][batch_idx])
                # convert to global labels
                final_labels.append(task_preds['labels'][task_id][batch_idx] + offset)
                # predict class in local categories
                offset += len(class_name)

            final_bboxes = torch.cat(final_bboxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)

            # sort filter
            # select_num = 200
            # if len(final_scores) > select_num:
            #     sorted, indices = torch.sort(final_scores, descending=True)
            #     final_bboxes = final_bboxes[indices[:select_num]]
            #     final_scores = final_scores[indices[:select_num]]
            #     final_labels = final_labels[indices[:select_num]]
            record_dict = {
                'pred_boxes': final_bboxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        data_dict['pred_dicts'] = pred_dicts

        return data_dict

    @torch.no_grad()
    def proposal_layer(self, heat, rot_sine, rot_cosine, height, dim, vel=None, reg=None,
                       cfg=None, raw_rot=False, task_id=-1):
        """Decode bboxes.

                Args:
                    heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
                    rot_sine (torch.Tensor): Sine of rotation with the shape of
                        [B, 1, W, H].
                    rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                        [B, 1, W, H].
                    height (torch.Tensor): Height of the boxes with the shape
                        of [B, 1, W, H].
                    dim (torch.Tensor): Dim of the boxes with the shape of
                        [B, 1, W, H].
                    vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
                    reg (torch.Tensor): Regression value of the boxes in 2D with
                        the shape of [B, 2, W, H]. Default: None.
                    task_id (int): Index of task. Default: -1.

                Returns:
                    list[dict]: Decoded boxes.
                """

        # nms_cfg = cfg.nms.train if self.training else cfg.nms.test
        nms_cfg = cfg.nms
        self.pc_range = cfg.pc_range
        self.score_threshold = cfg.score_threshold
        self.post_center_range = cfg.post_center_limit_range
        assert self.post_center_range is not None
        self.post_center_range = torch.tensor(self.post_center_range, device=heat.device)
        self.out_size_factor = cfg.out_size_factor
        self.use_circle_nms = nms_cfg.get("use_circle_nms", False)
        self.use_rotate_nms = nms_cfg.get("use_rotate_nms", False)
        self.nms_iou_threshold = nms_cfg.nms_iou_threshold
        self.use_multi_class_nms = nms_cfg.get("use_multi_class_nms", False)
        self.use_max_pool_nms = nms_cfg.get("use_max_pool_nms", False)
        self.nms_post_max_size = nms_cfg.nms_post_max_size
        self.nms_pre_max_size = nms_cfg.nms_pre_max_size
        if self.use_max_pool_nms:
            heat = self._nms(heat)

        rot = torch.atan2(rot_sine, rot_cosine)
        heat = heat.permute(0, 2, 3, 1)
        batch, H, W, cat = heat.size()
        heat = heat.reshape(batch, H * W, cat)
        rot = rot.permute(0, 2, 3, 1)
        rot = rot.reshape(batch, H * W, 1)
        height = height.permute(0, 2, 3, 1)
        height = height.reshape(batch, H * W, 1)
        dim = dim.permute(0, 2, 3, 1)
        dim = dim.reshape(batch, H * W, 3)

        ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(heat.device).float()
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(heat.device).float()
        if reg is not None:
            reg = reg.permute(0, 2, 3, 1)
            reg = reg.reshape(batch, H * W, 2)
            xs = xs.view(batch, -1, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        if vel is not None:
            vel = vel.permute(0, 2, 3, 1)
            vel = vel.reshape(batch, H * W, 2)
            final_box_preds = torch.cat([xs, ys, height, dim, rot, vel], dim=2)
        else:
            final_box_preds = torch.cat([xs, ys, height, dim, rot], dim=2)

        predictions_dicts = self.post_process(final_box_preds, heat)

        return predictions_dicts

    def post_process(self, batch_box_preds, batch_hm):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > self.score_threshold
            distance_mask = (box_preds[..., :3] >= self.post_center_range[:3]).all(1) \
                            & (box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            mask = distance_mask & score_mask

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            boxes_for_nms = box_preds[:, 0:7]
            # code below may cause strong AP drop
            # boxes_for_nms[:, -1] = -boxes_for_nms[:, -1] - np.pi / 2

            selected, _ = nms_gpu(boxes_for_nms, scores,
                                  thresh=self.nms_iou_threshold,
                                  pre_maxsize=self.nms_pre_max_size,
                                  post_max_size=self.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'bboxes': selected_boxes,
                'scores': selected_scores,
                'labels': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts

    def _topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() /
                   torch.tensor(width, dtype=torch.float)).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                      topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                    topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                    topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def clip_sigmoid(self, x, eps=1e-4):
        """Sigmoid function for input feature.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
            eps (float): Lower bound of the range to be clamped to. Defaults
                to 1e-4.

        Returns:
            torch.Tensor: Feature map after sigmoid.
        """
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
        return y

    def _double_flip(self, pred_dict, batch_size):
        """
        transform the prediction map back to their original coordinate before flipping
        the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
        the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is
        X and Y flip pointcloud(x=-x, y=-y).
        Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
        it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
        the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
        Args:
            pred_dict:
            batch_size:

        Returns:

        """
        for k in pred_dict.keys():
            _, C, H, W = pred_dict[k].shape
            # pdb.set_trace()
            pred_dict[k] = pred_dict[k].reshape(int(batch_size), 4, C, H, W)
            pred_dict[k][:, 1] = torch.flip(pred_dict[k][:, 1], dims=[2])  # y = -y
            pred_dict[k][:, 2] = torch.flip(pred_dict[k][:, 2], dims=[3])
            pred_dict[k][:, 3] = torch.flip(pred_dict[k][:, 3], dims=[2, 3])

        batch_hm = pred_dict['hm'].sigmoid()

        batch_reg = pred_dict['reg']
        batch_height = pred_dict['height']
        # pdb.set_trace()
        if not self.no_log:
            batch_dim = torch.exp(pred_dict['dim'])
        else:
            batch_dim = pred_dict['dim']

        batch_hm = batch_hm.mean(dim=1)
        batch_height = batch_height.mean(dim=1)
        batch_dim = batch_dim.mean(dim=1)

        # y = -y reg_y = 1-reg_y
        # reg是以左下角为原点的全局坐标系，预测x,y的offset,正常应该在0-1之间,flip之后应该用1去减
        batch_reg[:, 1, 1] = 1 - batch_reg[:, 1, 1]

        batch_reg[:, 2, 0] = 1 - batch_reg[:, 2, 0]

        batch_reg[:, 3, 0] = 1 - batch_reg[:, 3, 0]
        batch_reg[:, 3, 1] = 1 - batch_reg[:, 3, 1]
        batch_reg = batch_reg.mean(dim=1)

        batch_rots = pred_dict['rot'][:, :, 0:1]
        batch_rotc = pred_dict['rot'][:, :, 1:2]

        # first yflip
        # y = -y theta = pi -theta
        # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
        # batch_rots[:, 1] the same
        batch_rotc[:, 1] = -batch_rotc[:, 1]

        # then xflip x = -x theta = 2pi - theta
        # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
        # batch_rots[:, 2] the same
        batch_rots[:, 2] = -batch_rots[:, 2]

        # double flip
        batch_rots[:, 3] = -batch_rots[:, 3]
        batch_rotc[:, 3] = -batch_rotc[:, 3]

        batch_rotc = batch_rotc.mean(dim=1)
        batch_rots = batch_rots.mean(dim=1)

        batch_vel = None
        if 'vel' in pred_dict:
            batch_vel = pred_dict['vel']
            # flip vy
            # if flip along x axis, y = -y, vy = -vy
            batch_vel[:, 1, 1] = - batch_vel[:, 1, 1]
            # flip vx
            batch_vel[:, 2, 0] = - batch_vel[:, 2, 0]

            batch_vel[:, 3] = - batch_vel[:, 3]

            batch_vel = batch_vel.mean(dim=1)

        return batch_hm, batch_rots, batch_rotc, batch_height, batch_dim, batch_vel, batch_reg

    @numba.jit(nopython=True)
    def _circle_nms(self, dets, thresh, post_max_size=83):
        """Circular NMS.

        An object is only counted as positive if no other center
        with a higher confidence exists within a radius r using a
        bird-eye view distance metric.

        Args:
            dets (torch.Tensor): Detection results with the shape of [N, 3].
            thresh (float): Value of threshold.
            post_max_size (int): Max number of prediction to be kept. Defaults
                to 83

        Returns:
            torch.Tensor: Indexes of the detections to be kept.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        scores = dets[:, 2]
        order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int32)
        keep = []
        for _i in range(ndets):
            i = order[_i]  # start with highest score box
            if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
                continue
            keep.append(i)
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                # calculate center distance between i and j box
                dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

                # ovr = inter / areas[j]
                if dist <= thresh:
                    suppressed[j] = 1
        return keep[:post_max_size]

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
