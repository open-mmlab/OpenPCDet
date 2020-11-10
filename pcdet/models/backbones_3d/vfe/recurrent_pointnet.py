import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vfe_template import VFETemplate

# (Truncated) PointNet. Structure from https://www.qwertee.io/blog/deep-learning-with-point-clouds/
class TransformationNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 512, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(512)
        self.bn_3 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(512, 256)
        self.fc_2 = nn.Linear(256, self.output_dim*self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 512)

        x = F.relu(self.bn_3(self.fc_1(x)))
        x = self.fc_2(x)

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

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)

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

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 128)

        return x, feature_transform

# TODO: change the name to RecurrentPointNet.
class RecurrentPointNet(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.merge_interval = self.model_cfg.MERGE_INTERVAL
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ

        self.min_sweep_batch = self.model_cfg.MIN_SWEEP_BATCH
        self.max_sweep_batch = self.model_cfg.MAX_SWEEP_BATCH
        if self.with_distance:
            num_point_features += 1

        # Point dimension should exclude the time info and batch IDs.
        self.output_dim = self.model_cfg.OUTPUT_DIM
        self.hidden_dim = self.model_cfg.HIDDEN_DIM
        self.point_net = BasePointNet(3)

        self.use_lstm = self.model_cfg.USE_LSTM
        if self.use_lstm:
            self.rnn = nn.LSTM(self.model_cfg.RNN_INPUT_FEATURES, self.output_dim, 1, batch_first=True)
        else:
            self.rnn = nn.GRU(self.model_cfg.RNN_INPUT_FEATURES, self.output_dim, 1, batch_first=True)
        self.unique_batch_ids = None

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
        
    def split_batches_in_sweeps(self, unique_batch_ids, sweeps): 
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
        res = []
        last_available_sweep = [0 for i in range(len(unique_batch_ids))]

        for step in range(len(sweeps)):
            sweep = sweeps[step]
            batches_in_sweep = []
            for id in unique_batch_ids:
                batch = sweep[sweep[:, 0] == id]
                if len(batch) < self.min_sweep_batch:
                    if step == 0:
                        print("Too few points in the first step")
                        for next in range(step+1, len(sweeps)): 
                            curr_sweep = sweeps[next]
                            if len(curr_sweep[curr_sweep[:, 0] == id]) > 25500: # > self.min_sweep_batch:
                                # Find the next non-empty batch.
                                batch = curr_sweep[curr_sweep[:, 0] == id]
                                break
                        print("This batch still contains too few points")
                        batches_in_sweep.append(batch)
                        continue
                        
                    # This batch is too small and we should sample from the last available sweep.
                    last_available = res[last_available_sweep[id]][id]
                    sample_size = min(self.max_sweep_batch, last_available.shape[0])
                    random_indices = torch.randperm(last_available.shape[0])
                    # Randomly sample points from the last available sweep in the corresponding batch.
                    batches_in_sweep.append(last_available[random_indices][:sample_size])
                else: 
                    batches_in_sweep.append(batch)
                    last_available_sweep[id] = step
            res.append(batches_in_sweep) 

        # Find the batches that are valid (large enough)
        valid_batch_ids = []
        invalid_batch_ids = []
        for id in self.unique_batch_ids:
            batch_too_small = True
            for sweep in res:
                if sweep[id].shape[0] > self.min_sweep_batch:
                    batch_too_small = False
            if not batch_too_small:
                valid_batch_ids.append(id)
            else:
                invalid_batch_ids.append(id)
        if len(invalid_batch_ids) > 0:
            print("valid batches:", valid_batch_ids, "; invalid:", invalid_batch_ids)

        # Apply padding or cropping only on valid batches.
        for s in range(len(res)):
            min_len = min([res[s][id].shape[0] for id in valid_batch_ids])
            max_len = max([res[s][id].shape[0] for id in valid_batch_ids])
            target_batch_size = max_len
            if target_batch_size > self.max_sweep_batch:
                target_batch_size = self.max_sweep_batch
            if min_len * 2 < target_batch_size:
                # Unbalanced if impossible to pad.
                # Our assumption is that this should be rare. Log to see frequency.
                print("********** UNBALANCED **********", max_len, min_len)
                target_batch_size = min_len
            if min_len == 0:
                print("********** Empty batch found at step", s)
                continue

            for b in range(len(res[s])):
                rand_indices = torch.randperm(res[s][b].shape[0])
                if res[s][b].shape[0] >= target_batch_size:
                    # Crop points if exceeds the sample size.
                    res[s][b] = res[s][b][rand_indices][:target_batch_size]
                else:
                    # padding
                    padding_size = target_batch_size-res[s][b].shape[0]
                    res[s][b] = torch.cat((res[s][b], res[s][b][:padding_size]), 0)

        '''
        # Log #points in batches
        for sweep in res:
            bs = [str(b.shape[0]) for b in sweep]
            print(", ".join(bs))
        '''

        return res, valid_batch_ids

    def forward(self, batch_dict, **kwargs):
        # Simple tests for split_batches_in_sweeps
        '''
        tests = [torch.zeros(100, 6), torch.zeros(2, 6), torch.zeros(1, 6), torch.zeros(20, 6), torch.zeros(20000, 6)]
        self.split_batches_in_sweeps([0], tests)

        tests = [torch.zeros(100, 6), torch.zeros(2, 6), torch.zeros(1, 6), torch.zeros(20, 6), torch.zeros(200, 6)]
        self.split_batches_in_sweeps([0], tests)

        batchOne = torch.zeros(10000, 6)
        batchOne[0][0] = 1
        tests = [batchOne, torch.zeros(2, 6)]
        self.split_batches_in_sweeps([0, 1], tests)
        '''

        points = batch_dict['points']
        # Set unique batch IDs.
        if self.unique_batch_ids == None:
            self.unique_batch_ids = torch.unique(points[:, 0].int(), sorted=True)

        # Find unique time delays. Delays will be sorted in an ascending order (starting from 0).
        points[:, -1] = (1000 * points[:, -1]).int()
        unique_delays = torch.unique(points[:, -1], sorted=True)
        
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
        sweeps[:] = [sweep for sweep in sweeps if len(sweep) > self.min_sweep_batch]

        # Split sweeps into batches.
        sweeps_of_batches, valid_batch_ids = self.split_batches_in_sweeps(self.unique_batch_ids, sweeps)
        assert len(sweeps_of_batches) > 0 and len(valid_batch_ids) > 0
        batch_size = len(valid_batch_ids)

        # Output features by batches.
        output = torch.empty(size=(batch_size, 1, self.output_dim), device=torch.device('cuda:0'))
        if self.use_lstm:
            hidden = torch.empty(size=(1, batch_size, self.output_dim), device=torch.device('cuda:0'))
            mem = torch.empty(size=(1, batch_size, self.output_dim), device=torch.device('cuda:0'))
            for sweep in sweeps_of_batches: 
                # Each sweep contains batches of points.
                # Input sweeps into PointNet with only x, y, z.
                global_features, _ = self.point_net(torch.stack([sweep[bid] for bid in valid_batch_ids])[:, :, :3])
                global_features = global_features.unsqueeze(1)
                output, (hidden, mem) = self.rnn(global_features, (hidden, mem))
                # print("-------- current memory usage (LSTM):", self.get_gpu_memory_map())
        else:
            # Use GRU
            hidden = torch.empty(size=(1, batch_size, self.output_dim), device=torch.device('cuda:0'))
            for sweep in sweeps_of_batches: 
                # Each sweep contains batches of points.
                # Input sweeps into PointNet with only x, y, z.
                global_features, _ = self.point_net(torch.stack([sweep[bid] for bid in valid_batch_ids], dim=0)[:, :, :3])
                global_features = global_features.unsqueeze(1)
                output, hidden = self.rnn(global_features, hidden)
                # print("-------- current memory usage (GRU):", self.get_gpu_memory_map())

        output = output.squeeze()
        return output
        
    def get_gpu_memory_map(self):   
        import subprocess
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
        
        return float(result)
