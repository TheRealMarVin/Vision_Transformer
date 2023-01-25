import torch
import torch.nn as nn
import torch.nn.functional as  F

import numpy as np

import matplotlib.pyplot as plt

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, patch_hidden_size, nb_output, group_channel):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = self._make_tuple(patch_size)
        self.patch_hidden_size = patch_hidden_size
        self.group_channel_in_patch = group_channel

        patch_count = (self.img_size[1] * self.img_size[2]) / (self.patch_size[0] * self.patch_size[1])
        if self.group_channel_in_patch:
            patch_dimension = np.product(self.patch_size) * self.img_size[0]
        else:
            patch_count = patch_count * self.img_size[0]
            patch_dimension = np.product(self.patch_size)

        self.patch_count = int(patch_count)
        self.patch_dimension = patch_dimension

        self.patch_fc = nn.Linear(self.patch_dimension, self.patch_hidden_size)
        self.v_class = nn.Parameter(torch.rand(1, patch_hidden_size))

        self.fc = nn.Linear(400, nb_output) # 392 is just to have something that will run

    def forward(self, x):
        patches = self._create_patch(x)
        out = self.patch_fc(patches)
        out = torch.stack([torch.vstack((self.v_class, t)) for t in out])
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

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

        # unfold channels
        x = x.data.unfold(dimension=1, size=self.img_size[0], step=self.img_size[0])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        if self.group_channel_in_patch:
            x = x.reshape(x.shape[0], self.patch_count, self.img_size[0] * self.patch_size[0] * self.patch_size[1])
            # part = np.array([x[0][0][0].numpy(), x[0][0][1].numpy(), x[0][0][2].numpy()]).transpose(1,2,0)
            # plt.imshow(part)
            # plt.show()
        else:
            x = x.reshape(x.shape[0], self.patch_count, self.patch_size[0] * self.patch_size[1])

        return x
