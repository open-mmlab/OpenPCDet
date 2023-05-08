# This file is modified from https://github.com/traveller59/second.pytorch

import math
from functools import partial

import numpy as np
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        # if not isinstance(fai_optimizer, OptimWrapper):
        #     raise TypeError('{} is not a fastai OptimWrapper'.format(
        #         type(fai_optimizer).__name__))
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step, epoch=None):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    # print(pct, start, end)
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch=None):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def linear_warmup(end, lr_max, pct):
    k = (1 - pct / end) * (1 - 0.33333333)
    warmup_lr = lr_max * (1 - k)
    return warmup_lr


class CosineAnnealing(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, total_epoch, lr_max, moms, pct_start, warmup_iter):
        self.lr_max = lr_max
        self.moms = moms
        self.pct_start = pct_start

        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = lr_max, self.moms[0]

        self.optimizer = fai_optimizer
        self.total_step = total_step 
        self.warmup_iter = warmup_iter
        self.total_epoch = total_epoch

        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0
    
    def step(self, step, epoch):
        # update lr
        if step < self.warmup_iter:
            self.optimizer.lr = linear_warmup(self.warmup_iter, self.lr_max, step)
        else:
            target_lr = self.lr_max * 0.001
            cos_lr = annealing_cos(self.lr_max, target_lr, epoch / self.total_epoch)
            self.optimizer.lr = cos_lr
        # update mom
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


class FakeOptim:
    def __init__(self):
        self.lr = 0
        self.mom = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    opt = FakeOptim()  # 3e-3, wd=0.4, div_factor=10
    schd = OneCycle(opt, 100, 3e-3, (0.95, 0.85), 10.0, 0.1)

    lrs = []
    moms = []
    for i in range(100):
        schd.step(i)
        lrs.append(opt.lr)
        moms.append(opt.mom)
    plt.plot(lrs)
    # plt.plot(moms)
    plt.show()
    plt.plot(moms)
    plt.show()
