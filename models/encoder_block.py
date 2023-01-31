import torch
import torch.nn as nn
import torch.nn.functional as  F

import numpy as np

import matplotlib.pyplot as plt

from models.multi_head_self_attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, nb_embeddings, nb_heads, hidden_size):
        super(EncoderBlock, self).__init__()

        # Input layer normalization
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        # Multi heads self attention
        self.attention = MultiHeadSelfAttention(input_dim=embedding_dim,
                                                embedding_dim=embedding_dim,
                                                nb_heads=nb_heads)

        # second normalization layer
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.fc_1 = nn.Linear(embedding_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.layer_norm_1(x)
        x = self.attention(x) + identity

        x = self.layer_norm_2(x)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)

        return x
