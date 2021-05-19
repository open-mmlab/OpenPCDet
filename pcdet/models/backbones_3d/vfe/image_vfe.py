import torch

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v


class ImageVFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, depth_downsample_factor, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.downsample_factor = depth_downsample_factor
        self.module_topology = [
            'ffn', 'f2v'
        ]
        self.build_modules()

    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_ffn(self):
        """
        Builds frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

    def build_f2v(self):
        """
        Builds frustum to voxel transformation
        """
        f2v_module = f2v.__all__[self.model_cfg.F2V.NAME](
            model_cfg=self.model_cfg.F2V,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg
        )
        return f2v_module

    def get_output_feature_dim(self):
        return self.ffn.get_output_feature_dim()

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_dict = self.ffn(batch_dict)
        batch_dict = self.f2v(batch_dict)
        return batch_dict

    def get_loss(self):
        loss, tb_dict = self.ffn.get_loss()
        return loss, tb_dict
