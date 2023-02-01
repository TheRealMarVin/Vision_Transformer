import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, out_size):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, out_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)

        return x
