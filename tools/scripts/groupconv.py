import torch
import timeit

torch.backends.cudnn.benchmarking = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

group_conv = torch.nn.Conv2d(64*5, 15, (3, 3), padding=(1, 1), dilation=1, groups=5).cuda()
sep_convs = (torch.nn.Conv2d(64, 2, (3, 3), padding=(1, 1)).cuda(),
        torch.nn.Conv2d(64, 1, (3, 3), padding=(1, 1)).cuda(),
        torch.nn.Conv2d(64, 3, (3, 3), padding=(1, 1)).cuda(),
        torch.nn.Conv2d(64, 2, (3, 3), padding=(1, 1)).cuda(),
        torch.nn.Conv2d(64, 2, (3, 3), padding=(1, 1)).cuda())

inp_group = torch.rand((1, 64*5, 128, 128)).cuda()
inp_sep   = (torch.rand((1, 64, 128, 128)).cuda(),
        torch.rand((1, 64, 128, 128)).cuda(),
        torch.rand((1, 64, 128, 128)).cuda(),
        torch.rand((1, 64, 128, 128)).cuda(),
        torch.rand((1, 64, 128, 128)).cuda())

def run_sep_convs(convs, inps):
    return [c(i) for i, c in zip (inps, convs)]

def run_group_conv(gconv, inp):
    return gconv(inp)

out = run_sep_convs(sep_convs, inp_sep)
print('Seperate conv outputs:')
for o in out:
    print(o.size(), end=' ')
print()

out = run_group_conv(group_conv, inp_group)
print('group conv outputs:')
print(out.size())

torch.cuda.synchronize()

t1 = timeit.Timer(
    stmt='run_group_conv(group_conv, inp_group)',
    setup='from __main__ import run_group_conv',
    globals={'group_conv': group_conv, 'inp_group':inp_group})

torch.cuda.synchronize()

t0 = timeit.Timer(
    stmt='run_sep_convs(sep_convs, inp_sep)',
    setup='from __main__ import run_sep_convs',
    globals={'sep_convs': sep_convs, 'inp_sep': inp_sep})

torch.cuda.synchronize()

print(f'sep conv:  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'grp conv:  {t1.timeit(100) / 100 * 1e6:>5.1f} us')

