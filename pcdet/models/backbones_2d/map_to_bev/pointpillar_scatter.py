import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        channels_first = 'tcount' not in batch_dict
        dim1 = self.num_bev_features if channels_first else self.nz * self.nx * self.ny
        dim2 = self.nz * self.nx * self.ny if channels_first else self.num_bev_features
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(dim1, dim2, dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            if channels_first:
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
            else:
                spatial_feature[indices, :] = pillars

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        if channels_first:
            batch_spatial_features = batch_spatial_features.view(\
                    batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        else:
            batch_spatial_features = batch_spatial_features.view(\
                    batch_size, self.ny, self.nx, self.num_bev_features * self.nz)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
