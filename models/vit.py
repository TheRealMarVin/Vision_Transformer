import numpy as np
import torch
import torch.nn as nn

from models.encoder_block import EncoderBlock
from models.to_patch import ToPatch


class ViT(nn.Module):
    def __init__(self, embedding_layer,
                 img_size,
                 nb_output,
                 nb_encoder_blocks,
                 nb_heads):
        super(ViT, self).__init__()

        self.img_size = img_size

        # create the patch and embedding layer
        self.patch_layer = ToPatch(self.img_size, embedding_layer.patch_size)
        self.embedding_layer = embedding_layer

        # determine patch count
        patch_count = (self.img_size[1] * self.img_size[2]) / (embedding_layer.patch_size[0] * embedding_layer.patch_size[1])
        self.patch_count = int(patch_count)

        # position embeddings
        self.register_buffer('positional_embeddings', self._positional_encoding(self.embedding_layer.embedding_size, self.patch_count + 1), persistent=False)

        # v_class parameters
        self.v_class = nn.Parameter(torch.rand(1, self.embedding_layer.embedding_size))

        # Create encoder blocks
        encoder_list = []
        for _ in range(nb_encoder_blocks):
            encoder_list.append(EncoderBlock(embedding_dim=self.embedding_layer.embedding_size,
                                             nb_embeddings=self.patch_count + 1,
                                             nb_heads=nb_heads,
                                             hidden_size=self.embedding_layer.embedding_size * 4))

        self.encoder_block = nn.ModuleList(encoder_list)
        self.fc = nn.Linear(((self.patch_count + 1) * self.embedding_layer.embedding_size), nb_output)  # 400 is just to have something that will run

    def forward(self, x):
        # Create Patches
        patches = self.patch_layer(x)

        # Convert Patches to embeddings
        embedding = self.embedding_layer(patches)

        # Add the v_class to the tokes
        embedding = torch.stack([torch.vstack((self.v_class, t)) for t in embedding])

        # Add positional embeddings
        encoder = embedding + self.positional_embeddings.repeat(x.shape[0], 1, 1)

        # Encode blocks
        for encoder_block in self.encoder_block:
            encoder = encoder_block(encoder)

        # just send to an FC and
        out = encoder.view(encoder.shape[0], -1)
        out = self.fc(out)

        return out

    def _make_dimension_as_tuple(self, val):
        if type(val) is tuple:
            res = val
        else:
            res = (val, val)

        return res

    def _create_positional_embedding(self):
        # comes from the tutorial, but I don't think it is right
        results = torch.ones(self.patch_count + 1, self.patch_embedding_size)
        for i in range(self.patch_count + 1):
            for j in range(self.patch_embedding_size):
                if j % 2 == 0:
                    embedding = np.sin(i / 10000 ** (j / self.patch_embedding_size))
                else:
                    embedding = np.cos(i / 10000 ** ((j-1) / self.patch_embedding_size))

                results[i][j] = embedding
        return results

    def _positional_encoding(self, patch_embedding_size, patch_count):
        half_patch_count = patch_count / 2

        positions = np.arange(patch_embedding_size)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(half_patch_count)[np.newaxis, :] / half_patch_count  # (1, depth)

        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return torch.Tensor(pos_encoding.T[:patch_count, :])
