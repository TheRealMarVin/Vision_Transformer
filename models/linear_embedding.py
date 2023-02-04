import numpy as np
import torch.nn as nn


class LinearEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_size):
        super(LinearEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size

        self.fc_1 = nn.Linear(np.product(patch_size), embedding_size)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc_1(x)
        return x
