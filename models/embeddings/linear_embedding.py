import numpy as np
import torch.nn as nn

from helpers.patch_helpers import test_patch_order, debug_show_image_and_patches
from models.embeddings.to_patch import ToPatch


class LinearEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embedding_size):
        super(LinearEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size

        self.patch_layer = ToPatch(img_size, patch_size)
        self.fc_1 = nn.Linear(img_size[0] * np.prod(patch_size), embedding_size)

    def forward(self, img):
        x = self.patch_layer(img)

        # test_patch_order(img=img[0], patches=x[0], patch_size=self.patch_layer.patch_size)
        # debug_show_image_and_patches(img, x, self.patch_size, title="patch debug")

        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc_1(x)
        return x
