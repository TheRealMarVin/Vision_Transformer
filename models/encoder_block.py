import torch.nn as nn

from models.attention.multi_head_self_attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, nb_heads, hidden_size):
        super(EncoderBlock, self).__init__()

        # Input layer normalization
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        # Multi heads self attention
        self.attention = MultiHeadSelfAttention(input_dim=input_dim,
                                                embedding_dim=input_dim,
                                                nb_heads=nb_heads)
        #self.attention = SelfAttention(input_dim=input_dim,
        #                               embedding_dim=input_dim)

        # second normalization layer
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.fc_1 = nn.Linear(input_dim, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x

        x, _ = self.attention(x)
        x = x + identity
        x = self.layer_norm_1(x)

        identity = x
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x) + identity
        x = self.layer_norm_2(x)

        return x
