import torch
import torch.nn as nn


class FCNHead(nn.Module):
    def __init__(self, in_channels):
        super(FCNHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

    def forward(self, batch_dict):
        range_feature = batch_dict['range_features']
        output = self.conv1(range_feature)
