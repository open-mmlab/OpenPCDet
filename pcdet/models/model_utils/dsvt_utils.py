import torch
import torch.nn as nn
import numpy as np
from pcdet.ops.ingroup_inds.ingroup_inds_op import ingroup_inds


get_inner_win_inds_cuda = ingroup_inds


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift, shift_list=None, return_win_coors=False):

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        if shift_list is not None:
            shift_x, shift_y, shift_z = shift_list[0], shift_list[1], shift_list[2]
        else:
            shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z
    
    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z

    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z

    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
    if return_win_coors:
        batch_win_coords = torch.stack([win_coors_z, win_coors_y, win_coors_x], dim=-1)
        return batch_win_inds, coors_in_win, batch_win_coords
    
    return batch_win_inds, coors_in_win


def get_pooling_index(coors, sparse_shape, window_shape):
    win_shape_x, win_shape_y, win_shape_z = window_shape
    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    
    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)))
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y))) 
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z))) 
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    coors_x = coors[:, 3] 
    coors_y = coors[:, 2]
    coors_z = coors[:, 1]

    win_coors_x = coors_x // win_shape_x
    win_coors_y = coors_y // win_shape_y
    win_coors_z = coors_z // win_shape_z

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = coors_x % win_shape_x
    coors_in_win_y = coors_y % win_shape_y
    coors_in_win_z = coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    index_in_win = coors_in_win_x * win_shape_y * win_shape_z + \
                    coors_in_win_y * win_shape_z + \
                    coors_in_win_z

    batch_win_coords = torch.stack([coors[:, 0], win_coors_z, win_coors_y, win_coors_x], dim=-1)
    return batch_win_inds, coors_in_win, index_in_win, batch_win_coords


def get_continous_inds(setnum_per_win):
    '''
    Args:
        setnum_per_win (Tensor[int]): Number of sets assigned to each window with shape (win_num).
    Returns:
        set_win_inds (Tensor[int]): Window indexs of each set with shape (set_num).
        set_inds_in_win (Tensor[int]): Set indexs inner window with shape (set_num).

    Examples:
        setnum_per_win = torch.tensor([1, 2, 1, 3])
        set_inds_in_win = get_continous_inds(setnum_per_win)
        # we can get: set_inds_in_win = tensor([0, 0, 1, 0, 0, 1, 2])
    '''
    set_num = setnum_per_win.sum().item() # set_num = 7
    setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1] # [1, 3, 4]
    set_win_inds = torch.full((set_num,), 0, device=setnum_per_win.device)
    set_win_inds[setnum_per_win_cumsum] = 1 # [0, 1, 0, 1, 1, 0, 0]
    set_win_inds = torch.cumsum(set_win_inds, dim=0) # [0, 1, 1, 2, 3, 3, 3]
    
    roll_set_win_inds_left = torch.roll(set_win_inds, -1)  # [1, 1, 2, 3, 3, 3, 0]
    diff = set_win_inds - roll_set_win_inds_left # [-1, 0, -1, -1, 0, 0, 3]
    end_pos_mask = diff != 0
    template = torch.ones_like(set_win_inds)
    template[end_pos_mask] = (setnum_per_win - 1) * -1  # [ 0, 1, -1, 0, 1, 1, -2]
    set_inds_in_win = torch.cumsum(template,dim=0) # [0, 1, 0, 0, 1, 2, 0]
    set_inds_in_win[end_pos_mask] = setnum_per_win # [1, 1, 2, 1, 1, 2, 3]
    set_inds_in_win = set_inds_in_win - 1 # [0, 0, 1, 0, 0, 1, 2]
    
    return set_win_inds, set_inds_in_win