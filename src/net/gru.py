import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.zt_w = nn.Linear(input_size, 1)
        self.zt_u = nn.Linear(hidden_size, 1)

        self.rt_w = nn.Linear(input_size, 1)
        self.rt_u = nn.Linear(hidden_size, 1)

        self.h_w = nn.Linear(input_size, hidden_size)
        self.h_u = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hidden):
        T = input.shape[0]
        B = input.shape[1]
        output = torch.empty((T, B, self.hidden_size),
                              dtype=input.dtype, device=input.device)

        for i in range(T):
            zt = torch.sigmoid(self.zt_w(input[i]) + self.zt_u(hidden))
            rt = torch.sigmoid(self.rt_w(input[i]) + self.rt_u(hidden))
            nt = torch.tanh(self.h_w(input[i]) + self.h_u(hidden) * rt)
            output[i, :] = hidden = (1 - zt) * nt + zt * hidden

        return output, hidden
