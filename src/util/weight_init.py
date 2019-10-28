from torch import nn

def init_orthogonal_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        nn.init.uniform_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GRUCell):
        nn.init.constant_(m.b_ih, 0)
        nn.init.constant_(m.b_hh, 0)