import torch
from pcdet.ops.cuda_point_tile_mask import cuda_point_tile_mask
import time

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class DynamicFilterMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        #print(f'grid_size {grid_size}') # [1024 1024 40]
        #print(f'voxel_size {voxel_size}') # [0.1 0.1 0.2]
        #print(f'point_cloud_range {point_cloud_range}') # [-51.2 -51.2 -5. 51.2 51.2 3.]
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.tile_begin_idx=0

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
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
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        # point_coords are cell indexes
        # NOTE I turned .int() to .long(), might have an negative impact
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).long()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)

        points = points[mask]
        point_coords = point_coords[mask]
        ################################################################################
        # We have been given a number of tiles we can execute, let's call it K
        # We need to determine which tiles has voxels in them (nonempty tiles). Within
        # nontempty tiles we will select the K highest priority tiles. If the number K is
        # already higher than the number of nonempty tiles, we go ahead and select all of them
        # and do not filter any point. Otherwise, we will filter the points that fall outside of
        # selected tiles. Finally, we will update points and point_coords according to new filter
        # if needed. This part appears to have 3-4 ms overhead

        tcount, tile_prios, num_tiles_to_process, total_num_tiles = \
                batch_dict['tcount'], batch_dict['tile_prios'], \
                batch_dict['num_tiles_to_process'], batch_dict['total_num_tiles']

        tile_size_voxels = (self.grid_size[:2] / tcount).long() # [64, 64]

        # points[...,0] batch id
        tile_coords = torch.div(point_coords[..., :2], tile_size_voxels, rounding_mode='trunc')
        tile_coords = points[:, 0].long() * total_num_tiles + \
                tile_coords[...,0] * tcount[1] + tile_coords[...,1]

        # TODO There could be a way to avoid this unique, duplicate indexes appears working
        nonempty_tile_coords = torch.unique(tile_coords, sorted=True)

        if not self.training and nonempty_tile_coords.size(0) > num_tiles_to_process * batch_size:
            # Point filtering is needed
            # supress empty tiles by temporarily increasing the priority of nonempty tiles
            #tile_prios[nonempty_tile_coords] += total_num_tiles
            #highest_prios, chosen_tile_coords = \
            #        torch.topk(tile_prios, num_tiles_to_process, sorted=False)
            #tile_prios[nonempty_tile_coords] -= total_num_tiles

            #TODO this adhoc algorithm assumes the size nonempty tiles doesn't change much
            ntc = torch.cat((nonempty_tile_coords, nonempty_tile_coords))
            tl_end = self.tile_begin_idx+num_tiles_to_process
            chosen_tile_coords = ntc[self.tile_begin_idx:tl_end]
            self.tile_begin_idx = tl_end % nonempty_tile_coords.size(0)

            tile_filter = cuda_point_tile_mask.point_tile_mask(tile_coords, chosen_tile_coords)
            #TODO this masking could be done together with the initial masking
            points = points[tile_filter]
            point_coords = point_coords[tile_filter]
            batch_dict['chosen_tile_coords'] = chosen_tile_coords
        else:
            # No filtering, process all nonempty tiles
            self.tile_begin_idx=0
            batch_dict['num_tiles_to_process'] = nonempty_tile_coords.size(0)
            batch_dict['chosen_tile_coords'] = nonempty_tile_coords
            #print('points and coords aft filtering', points.size(), point_coords.size())
        ################################################################################

        merge_coords = points[:, 0].long() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.long()
        voxel_coords = torch.stack((torch.div(unq_coords, self.scale_xyz, rounding_mode='trunc'),
                torch.div((unq_coords % self.scale_xyz), self.scale_yz, rounding_mode='trunc'),
                torch.div((unq_coords % self.scale_yz), self.scale_z, rounding_mode='trunc'),
                unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict
