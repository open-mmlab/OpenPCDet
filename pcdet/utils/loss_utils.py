import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def neg_loss_sparse(pred, gt):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x n)
        gt: (batch x c x n)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class FocalLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossSparse, self).__init__()
        self.neg_loss = neg_loss_sparse

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegLossSparse(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossSparse, self).__init__()

    def forward(self, output, mask, ind=None, target=None, batch_index=None):
        """
        Args:
            output: (N x dim)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """

        pred = []
        batch_size = mask.shape[0]
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred.append(output[batch_inds][ind[bs_idx]])
        pred = torch.stack(pred)

        loss = _reg_loss(pred, target, mask)
        return loss


class IouLossSparse(nn.Module):
    '''IouLoss loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(IouLossSparse, self).__init__()

    def forward(self, iou_pred, mask, ind, box_pred, box_gt, batch_index):
        if mask.sum() == 0:
            return iou_pred.new_zeros((1))
        batch_size = mask.shape[0]
        mask = mask.bool()

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred = iou_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            pred_box = box_pred[batch_inds][ind[bs_idx]][mask[bs_idx]]
            target = iou3d_nms_utils.boxes_aligned_iou3d_gpu(pred_box, box_gt[bs_idx])
            target = 2 * target - 1
            loss += F.l1_loss(pred, target, reduction='sum')

        loss = loss / (mask.sum() + 1e-4)
        return loss

class IouRegLossSparse(nn.Module):
    '''Distance IoU loss for output boxes
        Arguments:
            output (batch x dim x h x w)
            mask (batch x max_objects)
            ind (batch x max_objects)
            target (batch x max_objects x dim)
    '''

    def __init__(self, type="DIoU"):
        super(IouRegLossSparse, self).__init__()

    def center_to_corner2d(self, center, dim):
        corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                    dtype=torch.float32, device=dim.device)
        corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
        corners = corners + center.view(-1, 1, 2)
        return corners

    def bbox3d_iou_func(self, pred_boxes, gt_boxes):
        assert pred_boxes.shape[0] == gt_boxes.shape[0]

        qcorners = self.center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
        gcorners = self.center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

        inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
        inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
        out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
        out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

        # calculate area
        volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
        volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

        inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
                torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
        inter_h = torch.clamp(inter_h, min=0)

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
        volume_inter = inter[:, 0] * inter[:, 1] * inter_h
        volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

        # boxes_iou3d_gpu(pred_boxes, gt_boxes)
        inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

        outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
                torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
        outer_h = torch.clamp(outer_h, min=0)
        outer = torch.clamp((out_max_xy - out_min_xy), min=0)
        outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

        dious = volume_inter / volume_union - inter_diag / outer_diag
        dious = torch.clamp(dious, min=-1.0, max=1.0)

        return dious

    def forward(self, box_pred, mask, ind, box_gt, batch_index):
        if mask.sum() == 0:
            return box_pred.new_zeros((1))
        mask = mask.bool()
        batch_size = mask.shape[0]

        loss = 0
        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            pred_box = box_pred[batch_inds][ind[bs_idx]]
            iou = self.bbox3d_iou_func(pred_box[mask[bs_idx]], box_gt[bs_idx])
            loss += (1. - iou).sum()

        loss =  loss / (mask.sum() + 1e-4)
        return loss