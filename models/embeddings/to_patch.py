import torch.nn as nn


class ToPatch(nn.Module):
    def __init__(self, img_size, patch_size):
        super(ToPatch, self).__init__()

        if len(img_size) != 3:
            raise ValueError(f"img_size must be (C, H, W), got {img_size}")

        self.img_size = img_size
        self.patch_size = self._make_dimension_as_tuple(patch_size)

        if self.img_size[1] % self.patch_size[0] != 0 or self.img_size[2] % self.patch_size[1] != 0:
            raise ValueError(f"Image size ({self.img_size}) must be divisible by patch size {self.patch_size}")

        self.grid_size = (self.img_size[1] // self.patch_size[0], self.img_size[2] // self.patch_size[1])
        self.patch_count = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        patch_h, patch_w = self.patch_size

        x = x.unfold(2, patch_h, patch_h)
        x = x.unfold(3, patch_w, patch_w)

        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

        x = x.view(x.shape[0], self.patch_count, self.img_size[0], patch_h, patch_w)

        return x

    def _make_dimension_as_tuple(self, val):
        if type(val) is tuple:
            if len(val) != 2:
                raise ValueError(f"patch_size tuple must be length 2, got {len(val)}")
            res = val
        else:
            res = (val, val)

        return res
