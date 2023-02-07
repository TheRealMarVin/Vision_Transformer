import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, embedding_dim, out_size):
        super(Classifier, self).__init__()
        self.fc_1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc_2 = nn.Linear(embedding_dim, out_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)

        return x
