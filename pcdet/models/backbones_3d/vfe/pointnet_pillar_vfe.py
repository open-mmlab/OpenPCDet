import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vfe_template import VFETemplate

# Structure from https://www.qwertee.io/blog/deep-learning-with-point-clouds/
class TransformationNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x

class BasePointNet(nn.Module):
    def __init__(self, point_dimension, return_local_features=False):
        super(BasePointNet, self).__init__()
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        num_points = x.shape[1]

        input_transform = self.input_transform(x)

        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = x.transpose(2, 1)

        feature_transform = self.feature_transform(x)

        x = torch.bmm(x, feature_transform)
        local_point_features = x

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024)

        if self.return_local_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return x, feature_transform

# TODO: change the name to RecurrentPointNet.
class PointNetPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.merge_interval = self.model_cfg.MERGE_INTERVAL
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ

        self.sweep_batch_lower_bound = self.model_cfg.SWEEP_BATCH_LOWER_BOUND
        self.sweep_batch_upper_bound = self.model_cfg.SWEEP_BATCH_UPPER_BOUND
        # num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        # Point dimension should exclude the time info and batch IDs.
        self.output_dim = self.model_cfg.OUTPUT_DIM
        self.point_net = BasePointNet(3)
        self.rnn = nn.LSTM(1024, self.output_dim, 1, batch_first=True)

    def get_output_feature_dim(self):
        return 1024

    def merge_sweeps(self, delays, sweeps):
        '''
        merge_sweeps merge all sweeps that are close enough in terms of time delays. It seems
        that the nuscenes sample sweeps at frequecy of 0.05 (unknown units), and the points
        of a sweep can be sampled at slightly different time (e.g. 50000 at 0.049 and 50000 at 0.051).
        The acceptable range of sampling window is defined at self.merge_interval.

        Args:
            delays: list of integers sorted in ascending order. Each integer stands for the time delay x 1000 
                after the first sampling starts.
            sweeps: list of sets of points where sweeps[i] is sampled at delays[i].
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
        
    def split_batches_in_sweeps(self, points, sweeps): 
        '''
        split_batches_in_sweeps breaks points of each sweep into batches. If a sweep have shape like [num_points, 6], 
        the output shape of the sweep will be [num_batch, batch_size, 5] (we will trim the first element as it stores
        the batch id). However, the number of points in a batch of a sweep can be variable. Since pytorch expect 
        fixed-size tensors, we will have to apply padding/downsizing to keep batch sizes the same.

        Args:
            points: list of points where each point is in the format of <batch_id, x, y, z, reflectivity, time delay>
            sweeps: list of sweep of points, and each sweep is a list as well. Each sweep of points stands for the points 
                in that sweep, and the points are in the same format as those in "points".
        Returns:
            list:
                batches_in_sweeps: <batch_size, num_points_in_batch, 5> where 5 is x, y, z, reflectivity, time delay. 
        '''
        unique_batch_ids = torch.unique(torch.tensor(points[:, 0], dtype=torch.int32), sorted=True)
        res = []

        # Each sweep contains points from different batches. 
        for sweep in sweeps:
            batches_in_sweep = []
            max_len = 0
            min_len = len(points)
            for id in unique_batch_ids:
                batch = sweep[sweep[:, 0] == id]
                batches_in_sweep.append(batch)

                if max_len < len(batch):
                    max_len = len(batch)
                if min_len > len(batch):
                    min_len = len(batch)
            
            target_batch_size = max_len
            if target_batch_size > self.sweep_batch_upper_bound:
                target_batch_size = self.sweep_batch_upper_bound
            if min_len * 2 < target_batch_size:
                # Our assumption is that this should be rare. Log to see frequency.
                print("********** UNBALANCED **********", max_len, min_len)
                target_batch_size = min_len

            # Create split batches beforehand; remove batch id from each point's features.
            split_batches = torch.empty(size=(len(unique_batch_ids), target_batch_size, points.shape[-1]-1))

            assert len(unique_batch_ids) == len(batches_in_sweep)
            for i in range(len(batches_in_sweep)): 
                batch = batches_in_sweep[i]

                # Apply downsizing/padding to make sure the sizes are equivalent and trim batch ID's when we pad the points. 
                # Given that these points are already shuffled, we will just repeat the first "max_len-len(batch)"
                # points.
                if len(batch) < target_batch_size:
                    # Padding
                    split_batches[i] = torch.cat((batch[:, 1:], batch[:target_batch_size-len(batch), 1:]))
                else:
                    # Downsizing
                    split_batches[i] = batch[:target_batch_size, 1:]
            res.append(split_batches)
        return res

    def forward(self, batch_dict, **kwargs):
        print(batch_dict.keys())
        points = batch_dict['points']

        # Find unique time delays. Delays will be sorted in an ascending order (starting from 0).
        points[:, -1] = (1000 * points[:, -1]).int()
        unique_delays = torch.unique(torch.tensor(points[:, -1], dtype=torch.int32), sorted=True)
        
        # Split sweeps by time delays.
        raw_sweeps = []
        for i in unique_delays:
            raw_sweeps.append(points[points[:, -1] == i])
        # print("*********** raw sweeps:", ",".join(['(%d,%d)' % (unique_delays[i], raw_sweeps[i].shape[0]) for i in range(len(unique_delays))]))

        # Merge close time delays.
        sweeps = self.merge_sweeps(unique_delays, raw_sweeps)
        point_cnt = 0
        for s in sweeps: 
            point_cnt += s.shape[0]
        assert point_cnt==len(points)
        # print("*********** merged sweeps:", ",".join([str(sweep.shape[0]) for sweep in sweeps]))

        # Filter out small sweeps.
        sweeps[:] = [sweep for sweep in sweeps if len(sweep) > self.sweep_batch_lower_bound]

        # Split sweeps into batches.
        sweeps_of_batches = self.split_batches_in_sweeps(points, sweeps)
        assert len(sweeps_of_batches) > 0
        batch_size = sweeps_of_batches[0].shape[0]

        # TODO: memory issues. 
        # Output features by batches.
        output = torch.empty(size=(batch_size, 1, self.output_dim), device=torch.device('cuda:0'))
        hidden = torch.empty(size=(1, batch_size, self.output_dim), device=torch.device('cuda:0'))
        mem = torch.empty(size=(1, batch_size, self.output_dim), device=torch.device('cuda:0'))
        for sweep in sweeps_of_batches: 
            # Input sweeps into PointNet with only x, y, z.
            print(self.get_gpu_memory_map())
            global_features, _ = self.point_net(sweep[:, :, :3].cuda())
            print(self.get_gpu_memory_map())
            global_features = global_features.unsqueeze(1)
            output, (hidden, mem) = self.rnn(global_features, (hidden, mem))

        print("out!", output.shape)
        return output
        
    def get_gpu_memory_map(self):   
        import subprocess
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
        
        return float(result)
