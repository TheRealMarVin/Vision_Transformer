import torch
import torch.nn as nn
import torch.nn.functional as  F

import matplotlib.pyplot as plt

class ViT(nn.Module):
    def __init__(self, img_size, patch_size):
        super(ViT, self).__init__()

        self.img_size = img_size
        self.patch_size = self._make_tuple(patch_size)
        self.group_channel_in_patch = True
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

        patch_count = (self.img_size[1] * self.img_size[2]) / (self.patch_size[0] * self.patch_size[1])
        if not self.group_channel_in_patch:
            patch_count = patch_count * self.img_size[0]

        # unfold channels
        x = x.data.unfold(dimension=1, size=self.img_size[0], step=self.img_size[0])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        if self.group_channel_in_patch:
            x = x.reshape(x.shape[0], int(patch_count), self.img_size[0], self.patch_size[0], self.patch_size[1])
            # import numpy as np
            # part = np.array([x[0][0][0].numpy(), x[0][0][1].numpy(), x[0][0][2].numpy()]).transpose(1,2,0)
            # plt.imshow(part)
            # plt.show()
        else:
            x = x.reshape(x.shape[0], int(patch_count), self.patch_size[0] * self.patch_size[1])

        return x
