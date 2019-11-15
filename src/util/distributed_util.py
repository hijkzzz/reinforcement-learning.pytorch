import torch.distributed as dist

def init(backend, rank, world_size):
    dist.init_process_group(backend=backend,
                            init_method="tcp://127.0.0.1:7788",
                            rank=rank,
                            world_size=world_size)

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.grad.data, src=0)