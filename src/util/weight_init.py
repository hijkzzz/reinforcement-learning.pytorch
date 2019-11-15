from torch import nn

def init_orthogonal_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

def init_xavier_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
    elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform(param.data)
            elif 'bias' in name:
                param.data.fill_(0)