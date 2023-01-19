import torch
from pcdet.ops.cuda_point_tile_mask import cuda_point_tile_mask
import time

# Sanity check
def do_sanity_check():
    for i in range(100):
        tile_coords = torch.randint(0, 16*16, (200000,), dtype=torch.long, device='cuda')
        chosen_tile_coords = torch.randint(0, 16*16, (150,), dtype=torch.long, device='cuda')
        chosen_tile_coords = torch.unique(chosen_tile_coords, sorted=False)

        mask1 = cuda_point_tile_mask.point_tile_mask(tile_coords, chosen_tile_coords)

        mask2 = torch.zeros(tile_coords.size(0), dtype=torch.bool, device='cuda')
        for tc in chosen_tile_coords:
            mask2 |= tc == tile_coords

        print(chosen_tile_coords.size(), torch.equal(mask1,mask2))

def do_speed_test():
    tile_coords = torch.randint(0, 16*16, (200000,), dtype=torch.long, device='cuda')
    chosen_tile_coords = torch.randint(0, 16*16, (150,), dtype=torch.long, device='cuda')
    chosen_tile_coords = torch.unique(chosen_tile_coords, sorted=False)

    torch.cuda.synchronize()
    t1=time.time()
    for i in range(100):
        mask1 = cuda_point_tile_mask.point_tile_mask(tile_coords, chosen_tile_coords)
    torch.cuda.synchronize()
    t2=time.time()
    
    print('tile_coords size:', tile_coords.size())
    print('chosen_tile_coords size:', chosen_tile_coords.size())
    print('time:', round((t2-t1)*1000/100, 3), 'ms')

    #mask2 = torch.zeros(tile_coords.size(0), dtype=torch.bool, device='cuda')
    #for tc in chosen_tile_coords:
    #    mask2 |= tc == tile_coords


do_sanity_check()
do_speed_test()
