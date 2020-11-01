import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vfe_template import VFETemplate

class PointNetPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.merge_interval = self.model_cfg.MERGE_INTERVAL
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        # Set up network here

    def get_output_feature_dim(self):
        return 1024

    def merge_sweeps(self, delays, sweeps):
        '''
        merge_sweeps merge all sweeps that are close enough in terms of time delays. It seems
        that the nuscenes sample sweeps at frequecy of 0.05 (unknown units), and the points
        of a sweep can be sampled at slightly different time (e.g. 50000 at 0.049 and 50000 at 0.051).
        The acceptable range of sampling window is defined at self.merge_interval.
        '''
        merged_sweeps = []
        i = 0
        while True:
            if i >= len(delays):
                break
            merged_sweep = sweeps[i]
            j = i+1
            while j < len(delays):
                if delays[j] > delays[i] + self.merge_interval:
                    break
                merged_sweep = torch.cat((merged_sweep, sweeps[j]), 0)
                j += 1

            merged_sweeps.append(merged_sweep)
            i = j

        return merged_sweeps
        

    def recurrent_feature_encoder(self, delays, sweeps):
        T_x = len(sweeps)

        print(len(sweeps))
        for i in range(len(sweeps)):
            sweep = sweeps[i]
            delay = delays[i]
            print(delay, sweep.shape)


    def forward(self, batch_dict, **kwargs):
        # Remove bacth IDs.
        points = batch_dict['points']
        points = points[:, 1:]

        # Find unique time delays. Delays will be sorted in an ascending order (starting from 0).
        points[:, -1] = (1000 * points[:, -1]).int()
        unique_delays = torch.unique(torch.tensor(points[:, -1], dtype=torch.int32), sorted=True)
        
        # Split sweeps by time delays.
        raw_sweeps = []
        for i in unique_delays:
            raw_sweeps.append(points[points[:, -1] == i])

        # Merge close time delays.
        sweeps = self.merge_sweeps(unique_delays, raw_sweeps)
        point_cnt = 0
        for s in sweeps: 
            print(s.shape)
            point_cnt += s.shape[0]
        assert point_cnt==len(points)

        # TODO(yuhaohe): please note that the encoder is a placeholder and the model won't work beyond this point. 
        features = self.recurrent_feature_encoder(unique_delays, sweeps)
        batch_dict['pillar_features'] = features

        return batch_dict
