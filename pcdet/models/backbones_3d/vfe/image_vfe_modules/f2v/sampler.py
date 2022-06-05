from functools import partial

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

        if torch.__version__ >= '1.3':
            self.grid_sample = partial(F.grid_sample, align_corners=True)
        else:
            self.grid_sample = F.grid_sample

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
        output = self.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output
