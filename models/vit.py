import torch
import torch.nn as nn
import torch.nn.functional as  F

import matplotlib.pyplot as plt

class ViT(nn.Module):
    def __init__(self, patch_size):
        super(ViT, self).__init__()

        self.patch_size = self._make_tuple(patch_size)
        # self.image_input_size = self._make_tuple(image_input_size)

        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self._create_patch(x)
        return x

    def _make_tuple(self, val):
        if type(val) is tuple:
            res = val
        else:
            res = (val, val)

        return res

    def _create_patch(self, x):
        # a = x[0].permute(1, 2, 0)
        # plt.imshow(a)
        # plt.show()

        shape = x.shape
        patch_count = (shape[-1] * shape[-2]  * shape[-3]) / (self.patch_size[0] * self.patch_size[1])

        # unfold channels
        x = x.data.unfold(dimension=1, size=shape[1], step=shape[1])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        # reshape to [batch_size, patch_count, patch_size*patch_size]
        x = x.reshape(shape[0], int(patch_count), self.patch_size[0] * self.patch_size[1])

        return x
