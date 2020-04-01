import numpy as np
import torch
import logging
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess
import random


def rotate_pc_along_z(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the LiDAR coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, 0:2] = np.dot(pc[:, 0:2], rotmat)
    return pc


def rotate_pc_along_z_torch(pc, rot_angle, inplace=True):
    """
    :param pc: (N, num_points, 3 + C) in the LiDAR coordinate
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)  # (N, 1)
    sina = torch.sin(rot_angle).view(-1, 1)  # (N, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)  # (N, 2)
    raw_2 = torch.cat([sina, cosa], dim=1)  # (N, 2)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, :, 0:2]  # (N, 512, 2)

    if inplace:
        pc[:, :, 0:2] = torch.matmul(pc_temp, R)  # (N, 512, 2)
    else:
        xy_rotated = torch.matmul(pc_temp, R)  # (N, 512, 2)
        pc = torch.cat((xy_rotated, pc[:, :, 2:]), dim=2)
    return pc


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    points = points[mask]
    return points


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 2] -= extra_width  # bugfixed: here should be minus, not add in LiDAR, 20190508
    return large_boxes3d


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def limit_period_torch(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period


def dict_select(dict_src, inds):
    for key, val in dict_src.items():
        if isinstance(val, dict):
            dict_select(val, inds)
        else:
            dict_src[key] = val[inds]


def create_logger(log_file, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(log_level if rank == 0 else 'ERROR')
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def init_dist_pytorch(batch_size, tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    assert batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, num_gpus)
    batch_size_each_gpu = batch_size // num_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def init_dist_slurm(batch_size, tcp_port, local_rank=None, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    :param batch_size:
    :param tcp_port:
    :param local_rank:
    :param backend:
    :return:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert batch_size % total_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, total_gpus)
    batch_size_each_gpu = batch_size // total_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
