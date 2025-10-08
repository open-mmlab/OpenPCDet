"""edited:
~change N into current at #flag1
pep-8, variable name
assert
~deepcopy the input tensor "xy" (otherwise xy will be cleared!!)
~check: print(xy.sum() == 0) in the function "convert"
draw a curve
"""

import torch
import pickle
from ipdb import set_trace


def convert_2d(xy:torch.Tensor, n:int) -> torch.Tensor:   
    """xy = Tensor([(x1, y1), (x2, y2), ...]), the size of the image is 2^n,
      the output is idx = Tensor([idx1, idx2, ...]) 
    """
    N = 2 ** n
    assert 0 <= xy.max() < N and isinstance(xy.sum().item(), int), "wrong input!"
    xy = xy.clone().detach()
    current_order = N >> 1 # the current order to be dealt with
    idx = torch.zeros(xy.size()[0], dtype=torch.int)

    while(current_order > 0): # orders go from high to low

        xy_bit = xy & current_order # get the bit of x, y at the current order
        y_cur_order = xy_bit[:, 1] > 0
        x_cur_order = xy_bit[:, 0] > 0 # the coordinate in the current order, denoted by the bool type.

        idx += current_order * current_order * (1 * x_cur_order + 1 * y_cur_order + 2 * (x_cur_order & (~y_cur_order)))
        # the formula in "()": transforms (0,0) to 0, ..., (1,0) to 3

        judge_reverse = (~y_cur_order) & x_cur_order # if the coordinate of the current order is (1, 0), reverse it 
        judge_rotate = ~y_cur_order # if the coordinate y of the current order is 1, rotate it

        xy[:, 0][judge_reverse] = current_order*2 - 1 - xy[:, 0][judge_reverse]  #flag1
        xy[:, 1][judge_reverse] = current_order*2 - 1 - xy[:, 1][judge_reverse]
        xy[:, 0][judge_rotate], xy[:,1][judge_rotate]  = xy[:, 1][judge_rotate], xy[:, 0][judge_rotate]

        xy[:, 0][x_cur_order] -= current_order
        xy[:, 1][y_cur_order] -= current_order  # or :  & ~(1 << current_order)
        current_order >>= 1 # go to the next order

    return idx
      
def get_bit_data(point_locs:torch.Tensor, bit:int) -> torch.Tensor:
    """point_locs_dim.size(): (input_size, num_dims) 
    """
    bit_data = (point_locs >> bit) & 1
    return bit_data

def convert_to_index(point_locs:torch.Tensor, num_dims:int, num_bits:int) -> torch.Tensor:
    """Decode a tensor of locations in a cube into a Hilbert integer.
    Params:
    -------
    point_locs - Locations in a cube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  
            The last dimension of the input has size num_dims.

    num_dims - The dimensionality of the cube. 

    num_bits - The number of bits for each dimension.

    Returns:
    --------
    The output is an tensor of int64 integers with the same shape as the
    input, excluding the last dimension, which needs to be num_dims.
    """

    # check that the locations are valid.
    if point_locs.shape[-1] != num_dims:
        raise ValueError(
            '''
            The shape of locs was surprising in that the last dimension was of size
            %d, but num_dims=%d.  These need to be equal.
            ''' % (point_locs.shape[-1], num_dims)
        )
    
    if num_dims*num_bits >= 64:
        raise ValueError(
            '''
            num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
            into an int64.  Are you sure you need that many points on your Hilbert
            curve?
            ''' % (num_dims, num_bits, num_dims*num_bits)
        )
    
    # follow the device on which the input is located
    if point_locs.device.type=='cuda':
        device = point_locs.device
    else:
        device = torch.device('cpu')

    # deepcopy the input
    point_locs = point_locs.clone().detach()

    # As num_dims*num_bits < 64, coordinates can be denoted by int32.
    point_locs = point_locs.type(torch.int32)

    fig_size = 1 << num_bits
    # Iterate forwards through the bits.
    bit_pow = fig_size >> 1
    while bit_pow > 1:

        # Iterate forwards through the dimensions.
        mask = bit_pow -1
        for dim in range(num_dims):

            judge_invert = (point_locs[:, dim] & bit_pow) > 0
            # Where this bit is on, invert the 0 dimension for lower bits.
            point_locs[:, 0][judge_invert] ^= mask

            judge_exchange = ~judge_invert 
            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = (point_locs[:, 0] ^ point_locs[:, dim]) & mask
            point_locs[:, 0][judge_exchange] ^= to_flip[judge_exchange]
            point_locs[:, dim][judge_exchange] ^= to_flip[judge_exchange]

        bit_pow >>= 1

    # Combine dims into one Gray code
    gray_code = torch.zeros(point_locs.size(0), dtype=torch.int64).to(device)
    for bit_current in range(num_bits):

        bit_data = get_bit_data(point_locs, bit_current)

        for dim in range(num_dims):
            # send bit_data to the correct position
            dim_shift = num_dims - 1 - dim # lower dim more significant
            gray_code += bit_data[:, dim] << (bit_current * num_dims + dim_shift)


    # Convert Gray code back to binary form of the index.
    shift = 2 ** (int(torch.ceil(torch.log2(torch.tensor(num_dims*num_bits)))) - 1)
    while shift > 0:
        gray_code ^= (gray_code >> shift)
        shift >>= 1
    
    point_indices = gray_code
    return point_indices

    
if __name__ == '__main__':


    # Generate Hilbert Curves 
    # (bit=9, z_max=41) / (bit=8, z_max=17) / (bit=7, z_max=9)
    dim = 3  # voxel-based dim is 3, pillar-based dim is 2
    # bit = 7  
    # N = 2 ** bit   # N must larger than BEV resolution
    device = torch.device('cuda')

    # z_max = 33 # for our setting, downstride = 1/2/4 | z_max = 33/17/9 for Waymo, z_max = 41/10/5 for nuScene
    # z_max = 9 # for our setting, downstride = 1/2/4 | z_max = 41/17/9 
    # use_size = N * N * z_max # Truncate the curve, z_max must be larger than the Z-axis resolution

    # generate for original resolution
    bit = 9  
    N = 2 ** bit   # N must larger than BEV resolution
    z_max = 41
    use_size = N * N * z_max

    if dim == 3:
        point_locs = torch.tensor([(z, y, x) for z in range(N) for y in range(N) for x in range(N)]).to(device)
    elif dim == 2:
        point_locs = torch.tensor([(y, x) for y in range(N) for x in range(N)]).to(device)
    else: 
        raise ValueError(
        '''
        The space dimension can only be 2 or 3 in the real world!
        '''
        )
    
    curve_index = convert_to_index(point_locs, dim, bit).cpu()
    curve_index_used = curve_index[:use_size]
    torch.save(curve_index_used, f'./data/hilbert/curve_template_3d_rank_{bit}.pth')

    # generate for downstride = 2
    bit = 8  
    N = 2 ** bit   # N must larger than BEV resolution
    z_max = 17
    use_size = N * N * z_max

    if dim == 3:
        point_locs = torch.tensor([(z, y, x) for z in range(N) for y in range(N) for x in range(N)]).to(device)
    elif dim == 2:
        point_locs = torch.tensor([(y, x) for y in range(N) for x in range(N)]).to(device)
    else: 
        raise ValueError(
        '''
        The space dimension can only be 2 or 3 in the real world!
        '''
        )
    
    curve_index = convert_to_index(point_locs, dim, bit).cpu()
    curve_index_used = curve_index[:use_size]
    torch.save(curve_index_used, f'./data/hilbert/curve_template_3d_rank_{bit}.pth')

    # generate for downstride = 4
    bit = 7  
    N = 2 ** bit   # N must larger than BEV resolution
    z_max = 9
    use_size = N * N * z_max

    if dim == 3:
        point_locs = torch.tensor([(z, y, x) for z in range(N) for y in range(N) for x in range(N)]).to(device)
    elif dim == 2:
        point_locs = torch.tensor([(y, x) for y in range(N) for x in range(N)]).to(device)
    else: 
        raise ValueError(
        '''
        The space dimension can only be 2 or 3 in the real world!
        '''
        )
    
    curve_index = convert_to_index(point_locs, dim, bit).cpu()
    curve_index_used = curve_index[:use_size]
    torch.save(curve_index_used, f'./data/hilbert/curve_template_3d_rank_{bit}.pth')