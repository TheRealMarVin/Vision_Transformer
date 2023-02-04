import torch.nn as nn


class ToPatch(nn.Module):
    def __init__(self, img_size, patch_size):
        super(ToPatch, self).__init__()

        self.img_size = img_size
        self.patch_size = self._make_dimension_as_tuple(patch_size)

        # determine patch count and the patch dimension
        patch_count = (self.img_size[1] * self.img_size[2]) / (self.patch_size[0] * self.patch_size[1])
        self.patch_count = int(patch_count)

    def forward(self, x):
        # unfold channels
        x = x.data.unfold(dimension=1, size=self.img_size[0], step=self.img_size[0])
        # unfold width
        x = x.data.unfold(dimension=2, size=self.patch_size[0], step=self.patch_size[0])
        # unfold height
        x = x.data.unfold(dimension=3, size=self.patch_size[1], step=self.patch_size[1])

        x = x.reshape(x.shape[0], self.patch_count, self.img_size[0] , self.patch_size[0] , self.patch_size[1])

        return x

    def _make_dimension_as_tuple(self, val):
        if type(val) is tuple:
            res = val
        else:
            res = (val, val)

        return res
