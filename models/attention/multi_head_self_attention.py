import numpy as np
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, nb_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = embedding_dim
        self.num_heads = nb_heads
        self.query = nn.Linear(input_dim, embedding_dim)
        self.key = nn.Linear(input_dim, embedding_dim)
        self.value = nn.Linear(input_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(embedding_dim * nb_heads, embedding_dim)

    def forward(self, input):
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        # Split into num_heads heads
        query = torch.chunk(query, self.num_heads, dim=-1)
        key = torch.chunk(key, self.num_heads, dim=-1)
        value = torch.chunk(value, self.num_heads, dim=-1)

        attention_outputs = []
        all_attention_scores = []
        for q, k, v in zip(query, key, value):
            # Compute dot product between query and key
            dot_product = torch.bmm(q, k.transpose(1,2))

            # Scale dot product by square root of attention dim
            attention_weights = dot_product / (self.attention_dim ** 0.5)

            # Apply softmax to compute attention weights
            attention_scores = self.softmax(attention_weights)
            all_attention_scores.append(attention_scores.detach().cpu().numpy())

            # Compute weighted sum of values
            attention_outputs.append(torch.bmm(attention_scores, v))

        # Concatenate attention outputs and project to attention_dim
        result = torch.cat(attention_outputs, dim=2)
        all_attention_scores = np.array(all_attention_scores)
        all_attention_scores = np.moveaxis(all_attention_scores, 0, 1)
        return result, all_attention_scores
