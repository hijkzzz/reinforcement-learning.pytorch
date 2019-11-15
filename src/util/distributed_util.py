import torch.distributed as dist

def init(backend):
    dist.init_process_group(backend=backend,
                            init_method="tcp://127.0.0.1:7788")

def sync_model(model):
    dist.broadcast_multigpu(list(model.parameters()), src=0)