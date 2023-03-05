import numpy as np
import torch.nn as nn

from models.embeddings.to_patch import ToPatch


class LinearEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embedding_size):
        super(LinearEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size

        self.patch_layer = ToPatch(img_size, patch_size)
        self.fc_1 = nn.Linear(np.product(patch_size), embedding_size)

    def forward(self, x):
        x = self.patch_layer(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc_1(x)
        return x
