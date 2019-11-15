import torch.distributed as dist

def init(backend="nccl"):
    dist.init_process_group(backend=backend,
                            init_method="tcp://127.0.0.1:35555")

def sync_model(model):
    dist.broadcast_multigpu(list(model.parameters()), src=0)