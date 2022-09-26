import torch
import slice_and_batch_cuda
import time

torch.backends.cudnn.benchmark = False

channels=64
slice_size=5
num_slices=500
H, W = 400, 352
inp = torch.rand((channels, H, W)).cuda()
print('Input size:', inp.size())
outp = torch.zeros((num_slices, channels, slice_size, slice_size)).cuda()
print('Sliced output size:', outp.size())

rand_weight1 = torch.rand((64, 64, 3, 3)).cuda()
rand_weight2 = torch.rand((3, 64, 3, 3)).cuda()

slice_times_ms=[]
reg_conv_times_ms=[]
slice_conv_times_ms=[]
hss = slice_size//2
for i in range(11):
    slice_indices = torch.randint(0, H * W, (num_slices,), dtype=torch.long).cuda()
    inp_padded = torch.nn.functional.pad(inp, (hss, hss, hss, hss))
    # change pos from center to corner in padded
    slice_indices += torch.div(slice_indices, W, rounding_mode='trunc') * hss * 2
    torch.cuda.synchronize()
    t1 = time.time()
    slice_and_batch_cuda.slice_and_batch_cuda(inp_padded, slice_indices, slice_size, outp)
    torch.cuda.synchronize()
    t2 = time.time()
    slice_times_ms.append((t2-t1)*1000)

    # Sanity check to make sure slicing happened as expected
    for i, ind in enumerate(slice_indices):
        # Also check if the first slice is correctly obtained
        h = torch.div(ind, inp_padded.size(2), rounding_mode='trunc')
        w = ind % inp_padded.size(2)

        slice1 = inp_padded[..., h:(h+slice_size), w:(w+slice_size)]
        slice2 = outp[i]
        if not torch.equal(slice1, slice2):
            print('Slicing error!', ind, w, h, slice1.size(), slice2.size())

    torch.cuda.synchronize()
    t1 = time.time()
    x = torch.nn.functional.conv2d(inp.unsqueeze(0), rand_weight1, stride=1, padding=1)
    x = torch.nn.functional.conv2d(x, rand_weight2, stride=1, padding=1)
    torch.cuda.synchronize()
    t2 = time.time()
    reg_conv_times_ms.append((t2-t1)*1000)

    torch.cuda.synchronize()
    t1 = time.time()
    x = torch.nn.functional.conv2d(outp, rand_weight1, stride=1, padding=0)
    x = torch.nn.functional.conv2d(x, rand_weight2, stride=1, padding=0)
    torch.cuda.synchronize()
    t2 = time.time()
    slice_conv_times_ms.append((t2-t1)*1000)

def avrg_time_ms(times):
    times = times[1:]
    return round(sum(times)/len(times), 3)

print(f'Average slicing time for {num_slices} slices:', avrg_time_ms(slice_times_ms), 'ms')
print(f'Average regular convolution time:', avrg_time_ms(reg_conv_times_ms), 'ms')
print(f'Average sliced convolution time for {num_slices} slices:', avrg_time_ms(slice_conv_times_ms), 'ms')
