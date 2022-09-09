import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        if 'tcount' in batch_dict: # Backbone is gonna use sbnet
            encoded_spconv_tensor.spatial_shape = encoded_spconv_tensor.spatial_shape[::-1]
            inds = encoded_spconv_tensor.indices
            inds_z = inds[..., 1].contiguous()
            inds[..., 1] = inds[...,-1]
            inds[...,-1] = inds_z
            encoded_spconv_tensor.indices = inds
            spatial_features = encoded_spconv_tensor.dense(channels_first=False).contiguous()
            N, H, W, D, C = spatial_features.shape
            spatial_features = spatial_features.view(N, H, W, D * C)
        else:
            spatial_features = encoded_spconv_tensor.dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
