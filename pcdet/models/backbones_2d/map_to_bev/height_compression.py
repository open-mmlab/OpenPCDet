import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sparse_shape = grid_size[::-1]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if 'encoded_spconv_tensor' in batch_dict.keys():
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
        else:
            import spconv
            batch_size = batch_dict['batch_size']
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
            input_sp_tensor = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            spatial_features = input_sp_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_2d'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict.get('encoded_spconv_tensor_stride', 1)
        return batch_dict
