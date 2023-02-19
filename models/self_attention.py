import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = embedding_dim
        self.query = nn.Linear(input_dim, embedding_dim)
        self.key = nn.Linear(input_dim, embedding_dim)
        self.value = nn.Linear(input_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        # Compute dot product between query and key
        dot_product = torch.bmm(query, key.transpose(1, 2))

        # Scale dot product by square root of attention dim
        attention_weights = dot_product / (self.attention_dim ** 0.5)

        # Apply softmax to compute attention weights
        attention_scores = self.softmax(attention_weights)

        # Compute weighted sum of values
        attention_output = torch.bmm(attention_scores, value)

        return attention_output, attention_scores
