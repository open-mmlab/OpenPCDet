import torch

def get_hilbert_index_3d_mamba_lite(template, coors, batch_size, z_dim, hilbert_spatial_size, shift=(0, 0, 0), debug=True):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    hil_size_z, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[2]
    y = coors[:, 2] + shift[1]
    z = coors[:, 1] + shift[0]

    flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info



def get_hilbert_index_2d_mamba_lite(template, coors, batch_size, hilbert_spatial_size, shift=(0, 0), debug=True):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    _, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[1]
    y = coors[:, 2] + shift[0]
    # z = coors[:, 1] + shift[0]

    # flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    flat_coors = (y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info