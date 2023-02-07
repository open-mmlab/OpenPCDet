import torch
import timeit
import csv

torch.backends.cudnn.benchmarking = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

num_channels=64

def conv_bn_relu_conv(outp_ch):
    global num_channels
    return torch.nn.Sequential(
        torch.nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1)),
        torch.nn.BatchNorm2d(num_channels, eps=1e-05, momentum=0.1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_channels, outp_ch, (3, 3), padding=(1, 1)))

# These are for single heada
def det_head(num_cls):
    return torch.nn.ModuleList([
#        conv_bn_relu_conv(num_cls),
        conv_bn_relu_conv(2),
        conv_bn_relu_conv(1),
        conv_bn_relu_conv(3),
        conv_bn_relu_conv(2),
        conv_bn_relu_conv(2)])


#single head
sep_convs = det_head(1).cuda()
num_convs = len(sep_convs)
group_conv = torch.nn.Sequential(
        torch.nn.Conv2d(num_channels, num_channels*num_convs, (3, 3), padding=(1, 1)),
        torch.nn.GroupNorm(num_convs, num_channels*num_convs, eps=1e-05),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_channels*num_convs, 3*num_convs, (3, 3), padding=(1, 1), 
            groups=num_convs)).cuda()

#multi-head
#cls_per_head = [1,2,2,1,2,2]
#num_heads = len(cls_per_head)
#sep_convs = torch.nn.ModuleList([det_head(c) for c in cls_per_head]).cuda()
#group_conv = torch.nn.Sequential(
#        torch.nn.Conv2d(num_channels, num_channels*num_heads*6, (3, 3), padding=(1, 1)),
#        torch.nn.GroupNorm(num_heads*6, num_channels*num_heads*6, eps=1e-05),
#        torch.nn.ReLU(),
#        torch.nn.Conv2d(num_channels*num_heads*6, 3*num_heads*6, (3, 3),
#            padding=(1, 1), groups=6*num_heads)).cuda()

def run_sep_convs(convs, inp):
    ret = [c(inp) for c in convs] # single head
    #ret = [c(inp) for h in convs for c in h] # multi head
    torch.cuda.synchronize()
    return ret

def run_group_conv(gconv, inp):
    ret = gconv(inp)
    torch.cuda.synchronize()
    return ret

# Group convolution is faster until 32 64x5x5 slices, then its slower than sep convolution

for inp_size in range(1, 257):
    inp = torch.rand((inp_size, num_channels, 5, 5)).cuda()
    out = run_sep_convs(sep_convs, inp)
    out = run_group_conv(group_conv, inp)

    t1 = timeit.Timer(
        stmt='run_group_conv(group_conv, inp)',
        setup='from __main__ import run_group_conv',
        globals={'group_conv': group_conv, 'inp':inp})

    t0 = timeit.Timer(
        stmt='run_sep_convs(sep_convs, inp)',
        setup='from __main__ import run_sep_convs',
        globals={'sep_convs': sep_convs, 'inp': inp})

    print('Timings for input size of', inp.size())
    print(f'sep conv:  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
    print(f'grp conv:  {t1.timeit(100) / 100 * 1e6:>5.1f} us')


def run_sep_convs(convs, inp):
    ret = [c(inp) for c in  convs]
    torch.cuda.synchronize()
    return ret

def run_group_conv(gconv, inp):
    ret = gconv(inp)
    torch.cuda.synchronize()
    return ret


with open('timing.csv', 'w', newline='') as csvfile:
    time_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for inp_size in range(1, 257):
        inp = torch.rand((inp_size, num_channels, 5, 5)).cuda()
        out = run_sep_convs(sep_convs, inp)
        out = run_group_conv(group_conv, inp)

        t1 = timeit.Timer(
            stmt='run_group_conv(group_conv, inp)',
            setup='from __main__ import run_group_conv',
            globals={'group_conv': group_conv, 'inp':inp})

        t0 = timeit.Timer(
            stmt='run_sep_convs(sep_convs, inp)',
            setup='from __main__ import run_sep_convs',
            globals={'sep_convs': sep_convs, 'inp': inp})

        sep_time_us = t0.timeit(100) / 100 * 1e6
        grp_time_us = t1.timeit(100) / 100 * 1e6
        print('Timings for input size of', inp.size())
        print(f'sep conv:  {sep_time_us} us')
        print(f'grp conv:  {grp_time_us} us')

        time_writer.writerow([inp_size, sep_time_us, grp_time_us])

