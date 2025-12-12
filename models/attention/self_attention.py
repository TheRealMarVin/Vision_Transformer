import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.attention_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        dot_product = torch.bmm(query, key.transpose(1, 2))
        attention_weights = dot_product / (self.attention_dim ** 0.5)
        attention_scores = self.softmax(attention_weights)
        attention_output = torch.bmm(attention_scores, value)

        return attention_output, attention_scores
