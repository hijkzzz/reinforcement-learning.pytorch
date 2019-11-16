import torch.distributed as dist

def init(backend, init_method, rank, world_size):
    dist.init_process_group(backend=backend,
                            init_method=init_method,
                            rank=rank,
                            world_size=world_size)

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)