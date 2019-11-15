import math

import torch
import torch.optim as optim
import torch.distributed as dist

# inherit Adam optim from torch library
class DistributedAdam(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, comm=None):
        # import all the basic params in Adam class
        super(DistributedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        # MPI comm
        self.comm = comm

    def step(self): # use super(SharedAdam, self).step
        size = float(dist.get_world_size())
        tensorlist = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.data /= size
                tensorlist.append(p.grad.data)

        dist.all_reduce_multigpu(tensorlist, op=dist.ReduceOp.SUM)
        return super(DistributedAdam, self).step()