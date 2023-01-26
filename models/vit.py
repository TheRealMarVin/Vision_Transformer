import torch
import torch.nn as nn
import torch.nn.functional as  F

import numpy as np

import matplotlib.pyplot as plt


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, patch_hidden_size, nb_output, group_channel):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = self._make_dimension_as_tuple(patch_size)
        self.patch_embedding_size = patch_hidden_size
        self.group_channel_in_patch = group_channel

        # determine patch count and the patch dimension
        patch_count = (self.img_size[1] * self.img_size[2]) / (self.patch_size[0] * self.patch_size[1])
        if self.group_channel_in_patch:
            patch_dimension = np.product(self.patch_size) * self.img_size[0]
        else:
            patch_count = patch_count * self.img_size[0]
            patch_dimension = np.product(self.patch_size)

        self.patch_count = int(patch_count)
        self.patch_dimension = patch_dimension

        # Patch embeddings layer and position embeddings
        self.patch_fc = nn.Linear(self.patch_dimension, self.patch_embedding_size)
        self.register_buffer('positional_embeddings', self._create_positional_embedding(), persistent=False)

        # v_class parameter
        self.v_class = nn.Parameter(torch.rand(1, patch_hidden_size))

        # temp
        self.fc = nn.Linear(400, nb_output) # 400 is just to have something that will run

    def forward(self, x):
        # Create Patches
        patches = self._create_patch(x)

        # Convert Patches to embeddings
        embedding = self.patch_fc(patches)

        # Add the v_class to the tokes
        embedding = torch.stack([torch.vstack((self.v_class, t)) for t in embedding])

        # Add positional embeddings
        embedding = embedding + self.positional_embeddings.repeat(x.shape[0], 1, 1)

        # just send to an FC and
        out = embedding.view(embedding.shape[0], -1)
        out = self.fc(out)

        return out

    def _make_dimension_as_tuple(self, val):
        if type(val) is tuple:
            res = val
        else:
            res = (val, val)

        return res

    def _create_positional_embedding(self):
        results = torch.ones(self.patch_count + 1, self.patch_embedding_size)
        for i in range(self.patch_count + 1):
            for j in range(self.patch_embedding_size):
                if j % 2 == 0:
                    embedding = np.sin(i / 10000 ** (j / self.patch_embedding_size))
                else:
                    embedding = np.cos(i / 10000 ** ((j-1) / self.patch_embedding_size))

                results[i][j] = embedding
        return results

    def _create_patch(self, x):
        # unfold channels
        x = x.data.unfold(dimension=1, size=self.img_size[0], step=self.img_size[0])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        if self.group_channel_in_patch:
            x = x.reshape(x.shape[0], self.patch_count, self.img_size[0] * self.patch_size[0] * self.patch_size[1])
        else:
            x = x.reshape(x.shape[0], self.patch_count, self.patch_size[0] * self.patch_size[1])

        return x
