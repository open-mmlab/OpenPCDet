import torch
import torch.nn as nn

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size,
                                                   pc_range=pc_range,
                                                   disc_cfg=disc_cfg)
        self.sampler = Sampler(**model_cfg.SAMPLER)

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features: (B, C, D, H_image, W_image), Image frustum features
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(lidar_to_cam=batch_dict["trans_lidar_to_cam"],
                                   cam_to_img=batch_dict["trans_cam_to_img"],
                                   image_shape=batch_dict["image_shape"])  # (B, X, Y, Z, 3)

        # Sample frustum volume to generate voxel volume
        voxel_features = self.sampler(input_features=batch_dict["frustum_features"],
                                      grid=grid)  # (B, C, X, Y, Z)

        # (B, C, X, Y, Z) -> (B, C, Z, Y, X)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        batch_dict["voxel_features"] = voxel_features
        return batch_dict
