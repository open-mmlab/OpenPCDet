import torch
import torch.nn as nn

from pcdet.utils import loss_utils


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss: (B, H, W), Pixel-wise loss
            gt_boxes2d: (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                             shape=loss.shape,
                                             downsample_factor=self.downsample_factor,
                                             device=loss.device)
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict = {"balancer_loss": loss.item(), "fg_loss": fg_loss.item(), "bg_loss": bg_loss.item()}
        return loss, tb_dict
