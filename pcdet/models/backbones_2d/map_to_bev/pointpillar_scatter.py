import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def slow_forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
    
    def forward(self, batch_dict, **kwargs):
        # coords -> (N, 4) [batch_idx, grid_z_idx, grid_y_idx, grid_x_idx]
        # pillar_features -> (N, C): N == num total voxels, C == channel features
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1

        spatial_features = torch.zeros(self.num_bev_features, batch_size*self.nz*self.ny*self.nx, dtype=pillar_features.dtype, device=pillar_features.device)

        coors_unique_idx = coords[:, 0] * self.nx * self.ny + coords[:, 2] * self.nx + coords[:, 3] # (N)
        coors_unique_idx_expand = coors_unique_idx.unsqueeze(0).expand(self.num_bev_features, -1) # (C,N)

        feature_values = pillar_features.t() # (C,N)

        spatial_features.scatter_(1, coors_unique_idx_expand.type(torch.long), feature_values)
        spatial_features = spatial_features.view(self.num_bev_features, batch_size, self.ny, self.nx).permute(1, 0, 2, 3)

        # Add spatial features back to batch_dict
        batch_dict['spatial_features'] = spatial_features
        return batch_dict


class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def slow_forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
    
    def forward(self, batch_dict, **kwargs):
        # coords -> (N, 4) [batch_idx, grid_z_idx, grid_y_idx, grid_x_idx]
        # pillar_features -> (N, C): N == num total voxels, C == channel features
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1

        spatial_features = torch.zeros(self.num_bev_features_before_compression, batch_size*self.nz*self.ny*self.nx, dtype=pillar_features.dtype, device=pillar_features.device)

        coors_unique_idx = coords[:, 0] * self.nz * self.ny * self.nx + coords[:, 1] * self.ny * self.nx + coords[:, 2] * self.nx + coords[:, 3] # (N)
        coors_unique_idx_expand = coors_unique_idx.unsqueeze(0).expand(self.num_bev_features_before_compression, -1) # (C,N)

        feature_values = pillar_features.t() # (C,N)

        spatial_features.scatter_(1, coors_unique_idx_expand.type(torch.long), feature_values)
        spatial_features = spatial_features.view(self.num_bev_features_before_compression, batch_size, self.nz, self.ny, self.nx).permute(1, 0, 2, 3, 4)
        batch_spatial_features = spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)

        # Add spatial features back to batch_dict
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


if __name__ == '__main__':
    # To test vector operation forward approach vs slow for loop forward
    import numpy as np
    from munch import Munch

    def generate_sample_data_coors(nx, ny, nz, batch_size, channels, num_voxels_each_sample):
        N = sum(num_voxels_each_sample)
        coors = [] # (N,4) // b_idx,z,y,x
        for i in range(batch_size):
            random_coords = set()
            while len(random_coords) < num_voxels_each_sample[i]:
                x = np.random.randint(0, nx)
                y = np.random.randint(0, ny)
                z = np.random.randint(0, nz)
                random_coords.add((x, y, z))
            random_coords = list(random_coords)
            for j in range(num_voxels_each_sample[i]):
                coors.append([i, random_coords[j][2], random_coords[j][1], random_coords[j][0]])
        # Stack coors vertically as torch tensor
        coors = torch.tensor(coors, dtype=torch.int64)
        return coors
    
    def generate_sample_data_features(nx, ny, nz, batch_size, channels, num_voxels_each_sample):
        N = sum(num_voxels_each_sample)
        voxel_features = [] # (N,C)
        for i in range(N):
            voxel_features.append(np.random.normal(size=channels))
        voxel_features = torch.tensor(voxel_features)
        return voxel_features


    nx = 24 * 8
    ny = 8 * 8
    nz = 1 # For pillar based scatter
    bs = 24
    channels = 64
    num_voxels_each_sample = [np.random.randint(1, nx*ny*nz) for _ in range(bs)]

    coors = generate_sample_data_coors(nx, ny, nz, bs, channels, num_voxels_each_sample)
    features = generate_sample_data_features(nx, ny, nz, bs, channels, num_voxels_each_sample)

    fake_dict = {
        'pillar_features': features,
        'voxel_coords': coors
    }
    fake_model_cfg = Munch()
    fake_model_cfg.NUM_BEV_FEATURES = channels
    fake_model_cfg.INPUT_SHAPE = (nx, ny, nz)

    def test_pillar_scatter_vectorized():
        pillar_scatter = PointPillarScatter(fake_model_cfg, (nx, ny, nz))
        slow_forward = pillar_scatter.slow_forward(fake_dict)
        fast_forward = pillar_scatter.forward(fake_dict)
        assert torch.all(torch.isclose(slow_forward['spatial_features'], fast_forward['spatial_features']))
    
    def test_pillar_scatter3d_vectorized():
        pillar_scatter3d = PointPillarScatter3d(fake_model_cfg, (nx, ny, nz))
        slow_forward = pillar_scatter3d.slow_forward(fake_dict)
        fast_forward = pillar_scatter3d.forward(fake_dict)
        assert torch.all(torch.isclose(slow_forward['spatial_features'], fast_forward['spatial_features']))
    
    test_pillar_scatter_vectorized()
    test_pillar_scatter3d_vectorized()
