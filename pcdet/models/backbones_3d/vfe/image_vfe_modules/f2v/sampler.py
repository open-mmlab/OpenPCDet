import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler(nn.Module):

    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: (B, C, D, H, W), Input frustum features
            grid: (B, X, Y, Z, 3), Sampling grids for input features
        Returns
            output_features: (B, C, X, Y, Z) Output voxel features
        """
        # Sample from grid
        output = F.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output
