import torch
import timeit

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

sep_convs = torch.nn.ModuleList([
        conv_bn_relu_conv(2),
        conv_bn_relu_conv(2),
        conv_bn_relu_conv(2),
        conv_bn_relu_conv(3),
        conv_bn_relu_conv(1),
        conv_bn_relu_conv(2)]).cuda()

group_conv = torch.nn.Sequential(
        torch.nn.Conv2d(num_channels, num_channels*6, (3, 3), padding=(1, 1)),
        torch.nn.GroupNorm(6, num_channels*6, eps=1e-05),
        torch.nn.ReLU(),
        torch.nn.Conv2d(num_channels*6, 3*6, (3, 3), padding=(1, 1), groups=6)).cuda()


def run_sep_convs(convs, inp):
    ret = [c(inp) for c in  convs]
    torch.cuda.synchronize()
    return ret

def run_group_conv(gconv, inp):
    ret = gconv(inp)
    torch.cuda.synchronize()
    return ret


for inp_size in (16, 32, 64, 128, 256):
    inp = torch.rand((1, num_channels, inp_size, inp_size)).cuda()
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

