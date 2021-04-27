from ....datasets.waymo.waymo_utils import plot_pointcloud, plot_pointcloud_with_gt_boxes
import torch


def plot_pc(this_points):
    import mayavi.mlab as mlab
    this_points_np = this_points[:, 1:].cpu().numpy()
    plot_pointcloud(this_points_np)
    mlab.show()


def plot_pc_with_gt(this_points, batch_idx, batch_dict):
    gt_np = batch_dict['gt_boxes'][batch_idx].cpu().numpy()
    this_points_np = this_points[:, 1:].cpu().numpy()
    plot_pointcloud_with_gt_boxes(this_points_np, gt_np)


def map_plot_with_gt(batch_idx, batch_dict):
    seg_mask = batch_dict['range_mask']
    batch_size, height, width = seg_mask.shape
    points = batch_dict['points']
    ri_indices = batch_dict['ri_indices']
    cur_seg_mask = seg_mask[batch_idx]
    cur_seg_mask = torch.flatten(cur_seg_mask)

    # points
    batch_points_mask = points[:, 0] == batch_idx
    this_points = points[batch_points_mask, :]
    this_ri_indices = ri_indices[batch_points_mask, :]
    this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
    this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
    this_points = this_points[this_points_mask]
    plot_pc_with_gt(this_points, batch_idx, batch_dict)


def plot_pc_with_gt_threshold(batch_idx, batch_dict, threshold=0.1):
    seg_mask = batch_dict['seg_pred'] >= threshold
    batch_size, height, width = seg_mask.shape
    points = batch_dict['points']
    ri_indices = batch_dict['ri_indices']
    cur_seg_mask = seg_mask[batch_idx]
    cur_seg_mask = torch.flatten(cur_seg_mask)

    # points
    batch_points_mask = points[:, 0] == batch_idx
    this_points = points[batch_points_mask, :]
    this_ri_indices = ri_indices[batch_points_mask, :]
    this_ri_indexes = (this_ri_indices[:, 1] * width + this_ri_indices[:, 2]).long()
    this_points_mask = torch.gather(cur_seg_mask, dim=0, index=this_ri_indexes).bool()
    this_points = this_points[this_points_mask]
    plot_pc_with_gt(this_points, batch_idx, batch_dict)
