import torch
from cuda_slicer import slice_and_batch_nhwc
import time

torch.backends.cudnn.benchmark = False

batch_size=2
channels=64
slice_size=5
num_slices=500
H, W = 400, 400
inp = torch.rand((batch_size, H, W, channels)).cuda()
print('Input size:', inp.size())
outp = torch.zeros((num_slices, channels, slice_size, slice_size)).cuda()
print('Sliced output size:', outp.size())

rand_weight1 = torch.rand((64, 64, 3, 3)).cuda()
rand_weight2 = torch.rand((3, 64, 3, 3)).cuda()

slice_times_ms=[]
reg_conv_times_ms=[]
slice_conv_times_ms=[]
for i in range(11):
    inds = [torch.randint(0, m, (num_slices,), dtype=torch.short).cuda() \
            for m in (batch_size, H-slice_size, W-slice_size)]
    inds = torch.stack(inds, dim=1)
    # change pos from center to corner in padded
    torch.cuda.synchronize()
    t1 = time.time()
    for j in range(10):
        outp = slice_and_batch_nhwc(inp, inds, slice_size)
    torch.cuda.synchronize()
    t2 = time.time()
    slice_times_ms.append((t2-t1)*1000/10)

    # Sanity check to make sure slicing happened as expected
    for i, ind in enumerate(inds):
        h = ind[1]
        w = ind[2]
        slice1 = inp[ind[0], h:(h+slice_size), w:(w+slice_size), :].permute(2,0,1).contiguous()
        slice2 = outp[i]
        if not torch.equal(slice1, slice2):
            print('Slicing error!', slice1.size(), slice2.size())

#    torch.cuda.synchronize()
#    t1 = time.time()
#    x = torch.nn.functional.conv2d(inp.unsqueeze(0), rand_weight1, stride=1, padding=1)
#    x = torch.nn.functional.conv2d(x, rand_weight2, stride=1, padding=1)
#    torch.cuda.synchronize()
#    t2 = time.time()
#    reg_conv_times_ms.append((t2-t1)*1000)
#
#    torch.cuda.synchronize()
#    t1 = time.time()
#    x = torch.nn.functional.conv2d(outp, rand_weight1, stride=1, padding=0)
#    x = torch.nn.functional.conv2d(x, rand_weight2, stride=1, padding=0)
#    torch.cuda.synchronize()
#    t2 = time.time()
#    slice_conv_times_ms.append((t2-t1)*1000)

def avrg_time_ms(times):
    times = times[1:]
    return round(sum(times)/len(times), 3)

print(f'Average slicing time for {num_slices} slices:', avrg_time_ms(slice_times_ms), 'ms')
#print(f'Average regular convolution time:', avrg_time_ms(reg_conv_times_ms), 'ms')
#print(f'Average sliced convolution time for {num_slices} slices:', avrg_time_ms(slice_conv_times_ms), 'ms')
