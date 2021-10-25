import numpy as np
import torch
import torch.nn as nn

from .base_bev_backbone import BaseBEVBackbone

class BaseBEVBackboneImprecise(BaseBEVBackbone):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg, input_channels)

        # Overriding it here so dense head can make three different heads
        self.num_bev_features = self.model_cfg.NUM_UPSAMPLE_FILTERS

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                stages and ups
        Returns:
        """
        #spatial_features = data_dict['spatial_features']  # make this stage0
        cur_stg = data_dict["stages_executed"]

        if cur_stg < len(self.deblocks):
            data_dict[f"stage{cur_stg+1}"] = self.blocks[cur_stg](
                    data_dict[f"stage{cur_stg}"])
            data_dict[f"up{cur_stg+1}"] = self.deblocks[cur_stg](
                    data_dict[f"stage{cur_stg+1}"])
            # leave the cat operation to dense head since we might do
            # an unnecessary cat here if there are more stages to
            # be executed
        #    data_dict[f"backbone_out_list"] = [data_dict[f"up{i}"] for i in range(1, cur_stg+2)]
        else:
            # cat
            x = torch.cat([data_dict[f"up{i}"] for i in range(1, cur_stg+2)], dim=1)
            data_dict[f"up{cur_stg+1}"] = self.deblocks[-1](x)
            #data_dict['spatial_features_2d'] = x
        #    data_dict[f"backbone_out_list"] = [data_dict[f"up{cur_stg+1}"]]

        data_dict["stages_executed"] += 1
        #data_dict['spatial_features_2d'] = x
        return data_dict
