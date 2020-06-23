import torch.nn as nn


class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError
