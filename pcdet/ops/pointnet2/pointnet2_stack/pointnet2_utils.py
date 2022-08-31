import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_stack_cuda as pointnet2


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0

        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(empty_ball_mask)

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.farthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample = furthest_point_sample = FarthestPointSampling.apply


class StackFarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, xyz_batch_cnt, npoint):
        """
        Args:
            ctx:
            xyz: (N1 + N2 + ..., 3) where N > npoint
            xyz_batch_cnt: [N1, N2, ...]
            npoint: int, number of features in the sampled set

        Returns:
            output: (npoint.sum()) tensor containing the set,
            npoint: (M1, M2, ...)
        """
        assert xyz.is_contiguous() and xyz.shape[1] == 3

        batch_size = xyz_batch_cnt.__len__()
        if not isinstance(npoint, torch.Tensor):
            if not isinstance(npoint, list):
                npoint = [npoint for i in range(batch_size)]
            npoint = torch.tensor(npoint, device=xyz.device).int()

        N, _ = xyz.size()
        temp = torch.cuda.FloatTensor(N).fill_(1e10)
        output = torch.cuda.IntTensor(npoint.sum().item())

        pointnet2.stack_farthest_point_sampling_wrapper(xyz, temp, xyz_batch_cnt, output, npoint)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


stack_farthest_point_sample = StackFarthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, unknown_batch_cnt, known, known_batch_cnt):
        """
        Args:
            ctx:
            unknown: (N1 + N2..., 3)
            unknown_batch_cnt: (batch_size), [N1, N2, ...]
            known: (M1 + M2..., 3)
            known_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
            idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
        """
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = unknown.new_zeros(unknown.shape)
        idx = unknown_batch_cnt.new_zeros(unknown.shape).int()

        pointnet2.three_nn_wrapper(
            unknown.contiguous(), unknown_batch_cnt.contiguous(),
            known.contiguous(), known_batch_cnt.contiguous(), dist2, idx
        )
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        pointnet2.three_interpolate_wrapper(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(
            grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class ThreeNNForVectorPoolByTwoStep(Function):
    @staticmethod
    def forward(ctx, support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                max_neighbour_distance, nsample, neighbor_type, avg_length_of_neighbor_idxs, num_total_grids,
                neighbor_distance_multiplier):
        """
        Args:
            ctx:
            // support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            // xyz_batch_cnt: (batch_size), [N1, N2, ...]
            // new_xyz: (M1 + M2 ..., 3) centers of the ball query
            // new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            // new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            // nsample: find all (-1), find limited number(>0)
            // neighbor_type: 1: ball, others: cube
            // neighbor_distance_multiplier: query_distance = neighbor_distance_multiplier * max_neighbour_distance

        Returns:
            // new_xyz_grid_idxs: (M1 + M2 ..., num_total_grids, 3) three-nn
            // new_xyz_grid_dist2: (M1 + M2 ..., num_total_grids, 3) square of dist of three-nn
        """
        num_new_xyz = new_xyz.shape[0]
        new_xyz_grid_dist2 = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape)
        new_xyz_grid_idxs = new_xyz_grid_centers.new_zeros(new_xyz_grid_centers.shape).int().fill_(-1)

        while True:
            num_max_sum_points = avg_length_of_neighbor_idxs * num_new_xyz
            stack_neighbor_idxs = new_xyz_grid_idxs.new_zeros(num_max_sum_points)
            start_len = new_xyz_grid_idxs.new_zeros(num_new_xyz, 2).int()
            cumsum = new_xyz_grid_idxs.new_zeros(1)

            pointnet2.query_stacked_local_neighbor_idxs_wrapper_stack(
                support_xyz.contiguous(), xyz_batch_cnt.contiguous(),
                new_xyz.contiguous(), new_xyz_batch_cnt.contiguous(),
                stack_neighbor_idxs.contiguous(), start_len.contiguous(), cumsum,
                avg_length_of_neighbor_idxs, max_neighbour_distance * neighbor_distance_multiplier,
                nsample, neighbor_type
            )
            avg_length_of_neighbor_idxs = cumsum[0].item() // num_new_xyz + int(cumsum[0].item() % num_new_xyz > 0)

            if cumsum[0] <= num_max_sum_points:
                break

        stack_neighbor_idxs = stack_neighbor_idxs[:cumsum[0]]
        pointnet2.query_three_nn_by_stacked_local_idxs_wrapper_stack(
            support_xyz, new_xyz, new_xyz_grid_centers, new_xyz_grid_idxs, new_xyz_grid_dist2,
            stack_neighbor_idxs, start_len, num_new_xyz, num_total_grids
        )

        return torch.sqrt(new_xyz_grid_dist2), new_xyz_grid_idxs, torch.tensor(avg_length_of_neighbor_idxs)


three_nn_for_vector_pool_by_two_step = ThreeNNForVectorPoolByTwoStep.apply


class VectorPoolWithVoxelQuery(Function):
    @staticmethod
    def forward(ctx, support_xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor, support_features: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor, num_grid_x, num_grid_y, num_grid_z,
                max_neighbour_distance, num_c_out_each_grid, use_xyz,
                num_mean_points_per_grid=100, nsample=-1, neighbor_type=0, pooling_type=0):
        """
        Args:
            ctx:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            support_features: (N1 + N2 ..., C)
            new_xyz: (M1 + M2 ..., 3) centers of new positions
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            num_grid_x: number of grids in each local area centered at new_xyz
            num_grid_y:
            num_grid_z:
            max_neighbour_distance:
            num_c_out_each_grid:
            use_xyz:
            neighbor_type: 1: ball, others: cube:
            pooling_type: 0: avg_pool, 1: random choice
        Returns:
            new_features: (M1 + M2 ..., num_c_out)
        """
        assert support_xyz.is_contiguous()
        assert support_features.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        num_total_grids = num_grid_x * num_grid_y * num_grid_z
        num_c_out = num_c_out_each_grid * num_total_grids
        N, num_c_in = support_features.shape
        M = new_xyz.shape[0]

        assert num_c_in % num_c_out_each_grid == 0, \
            f'the input channels ({num_c_in}) should be an integral multiple of num_c_out_each_grid({num_c_out_each_grid})'

        while True:
            new_features = support_features.new_zeros((M, num_c_out))
            new_local_xyz = support_features.new_zeros((M, 3 * num_total_grids))
            point_cnt_of_grid = xyz_batch_cnt.new_zeros((M, num_total_grids))

            num_max_sum_points = num_mean_points_per_grid * M
            grouped_idxs = xyz_batch_cnt.new_zeros((num_max_sum_points, 3))

            num_cum_sum = pointnet2.vector_pool_wrapper(
                support_xyz, xyz_batch_cnt, support_features, new_xyz, new_xyz_batch_cnt,
                new_features, new_local_xyz, point_cnt_of_grid, grouped_idxs,
                num_grid_x, num_grid_y, num_grid_z, max_neighbour_distance, use_xyz,
                num_max_sum_points, nsample, neighbor_type, pooling_type
            )
            num_mean_points_per_grid = num_cum_sum // M + int(num_cum_sum % M > 0)
            if num_cum_sum <= num_max_sum_points:
                break

        grouped_idxs = grouped_idxs[:num_cum_sum]

        normalizer = torch.clamp_min(point_cnt_of_grid[:, :, None].float(), min=1e-6)
        new_features = (new_features.view(-1, num_total_grids, num_c_out_each_grid) / normalizer).view(-1, num_c_out)

        if use_xyz:
            new_local_xyz = (new_local_xyz.view(-1, num_total_grids, 3) / normalizer).view(-1, num_total_grids * 3)

        num_mean_points_per_grid = torch.Tensor([num_mean_points_per_grid]).int()
        nsample = torch.Tensor([nsample]).int()
        ctx.vector_pool_for_backward = (point_cnt_of_grid, grouped_idxs, N, num_c_in)
        ctx.mark_non_differentiable(new_local_xyz, num_mean_points_per_grid, nsample, point_cnt_of_grid)
        return new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid

    @staticmethod
    def backward(ctx, grad_new_features: torch.Tensor, grad_local_xyz: torch.Tensor, grad_num_cum_sum, grad_point_cnt_of_grid):
        """
        Args:
            ctx:
            grad_new_features: (M1 + M2 ..., num_c_out), num_c_out = num_c_out_each_grid * num_total_grids

        Returns:
            grad_support_features: (N1 + N2 ..., C_in)
        """
        point_cnt_of_grid, grouped_idxs, N, num_c_in = ctx.vector_pool_for_backward
        grad_support_features = grad_new_features.new_zeros((N, num_c_in))

        if grouped_idxs.shape[0] > 0:
            pointnet2.vector_pool_grad_wrapper(
                grad_new_features.contiguous(), point_cnt_of_grid, grouped_idxs,
                grad_support_features
            )

        return None, None, grad_support_features, None, None, None, None, None, None, None, None, None, None, None, None


vector_pool_with_voxel_query_op = VectorPoolWithVoxelQuery.apply


if __name__ == '__main__':
    pass
