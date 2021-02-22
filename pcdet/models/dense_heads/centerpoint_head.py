import copy
import numpy as np
import torch
from torch import nn
from ...utils import loss_utils
from .target_assigner.center_assigner import CenterAssigner
import numba
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

    def __init__(self, in_channels, heads, head_conv=64, final_kernel=1, init_bias=-2.19, bn=False, **kwargs):
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
        self.no_log = self.model_cfg.NO_LOG

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
                    SepHead(share_conv_channel, heads, final_kernel=3, bn=True, init_bias=self.init_bias))

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

    def init_weights(self):
        """Initialize weights."""
        for task_head in self.task_heads:
            task_head.init_weights()

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

            hm_loss = self.crit(pred_dict['hm'], self.forward_ret_dict['heatmap'][task_id])

            target_box_encoding = self.forward_ret_dict['box_encoding'][task_id]
            if self.dataset == 'nuscenes':
                # nuscenes encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['hei'],
                    pred_dict['dim'],
                    pred_dict['rot'],
                    pred_dict['vel']
                ], dim=1).contiguous()  # (B, 10, H, W)
            elif self.dataset == 'waymo':
                pred_box_encoding = torch.cat([
                    pred_dict['reg'],
                    pred_dict['hei'],
                    pred_dict['dim'],
                    pred_dict['rot']
                ], dim=1).contiguous()  # (B, 8, H, W)
            else:
                raise NotImplementedError("Only Support KITTI and nuScene for Now!")

            self.forward_ret_dict['pred_box_encoding'][task_id] = pred_box_encoding

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
            'crit', loss_utils.CenterNetFocalLoss()
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
        double_flip = not self.training and self.post_cfg.get('double_flip', False)  # type: bool
        pred_dicts = self.forward_ret_dict['multi_head_features']  # output of forward func.
        post_center_range = self.post_cfg.post_center_limit_range
        batch_size = data_dict['batch_size']

        task_preds = {}
        task_preds['bboxes'] = {}
        task_preds['scores'] = {}
        task_preds['labels'] = {}

        for task_id, pred_dict in enumerate(pred_dicts):
            # TODO: double flip
            batch_hm = pred_dict['hm'].sigmoid()
            batch_reg = pred_dict['reg']
            batch_hei = pred_dict['hei']
            if not self.no_log:
                batch_dim = torch.exp(pred_dict['dim'])
                # clamp for good init, otherwise it will goes inf with exp
                batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
            else:
                batch_dim = pred_dict['dim']
            batch_rot_sine = pred_dict['rot'][:, 0].unsqueeze(1)
            batch_rot_cosine = pred_dict['rot'][:, 1].unsqueeze(1)

            bat_vel = pred_dict['vel'] if self.dataset == 'nuscenes' else None

            # decode
            boxes = self.proposal_layer(batch_hm, batch_rot_sine, batch_rot_cosine, batch_hei, batch_dim, bat_vel,
                                        reg=batch_reg, cfg=self.post_cfg, task_id=task_id)
            task_preds['bboxes'][task_id] = [box['bboxes'] for box in boxes]
            task_preds['scores'][task_id] = [box['scores'] for box in boxes]
            task_preds['labels'][task_id] = [box['labels'] for box in boxes]  # labels are local here

        pred_dicts = []
        nms_cfg = self.post_cfg.nms
        num_rois = nms_cfg.nms_pre_max_size * self.num_class
        for batch_idx in range(batch_size):
            final_bboxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                offset = 1
                final_bboxes.append(task_preds['bboxes'][batch_idx])
                final_scores.append(task_preds['scores'][batch_idx])
                # convert to global labels
                final_labels.append(task_preds['labels'][batch_idx] + offset)
                # predict class in local categories
                offset += len(class_name)

            final_bboxes = torch.cat(final_bboxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)

            record_dict = {
                'pred_boxes': final_bboxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }

        data_dict['pre_dicts'] = pred_dicts

        return data_dict

    @torch.no_grad()
    def proposal_layer(self, heat, rot_sine, rot_cosine, hei, dim, vel=None, reg=None,
                       cfg=None, raw_rot=False, task_id=-1):
        """Decode bboxes.

                Args:
                    heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
                    rot_sine (torch.Tensor): Sine of rotation with the shape of
                        [B, 1, W, H].
                    rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                        [B, 1, W, H].
                    hei (torch.Tensor): Height of the boxes with the shape
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
        batch, cat, _, _ = heat.size()
        # nms_cfg = cfg.nms.train if self.training else cfg.nms.test
        nms_cfg = cfg.nms
        self.pc_range = cfg.pc_range
        self.score_threshold = cfg.score_threshold
        self.post_center_range = cfg.post_center_range
        self.out_size_factor = cfg.out_size_factor
        K = nms_cfg.nms_pre_max_size
        scores, inds, clses, ys, xs = self._topk(heat, K)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        # rotation value and direction label
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, K, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, K, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, K, 1)

        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, K, 3)

        # class label
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)

        xs = xs.view(
            batch, K,
            1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(
            batch, K,
            1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, K, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts

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

    @numba.jit(nopython=True)
    def circle_nms(dets, thresh, post_max_size=83):
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

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the \
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the \
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the \
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in birdeye view

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_maxsize=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
