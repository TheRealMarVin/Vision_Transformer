import torch.nn as nn
from matplotlib.projections import projection_registry
from torch.nn.functional import embedding

from models.attention.multi_head_self_attention import MultiHeadSelfAttention
from models.drop_path import DropPath


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, nb_heads, linear_expand_ratio=4.0, attention_dropout=0.0,
                 projection_dropout=0.0, linear_dropout=0.0, drop_path=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, nb_heads, attention_dropout, projection_dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)

        mlp_hidden_dim = int(embedding_dim * linear_expand_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(linear_dropout),
            nn.Linear(mlp_hidden_dim, embedding_dim),
            nn.Dropout(linear_dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, return_attention=False):
        attn_out, attn = self.attention(self.norm1(x), return_attn=return_attention)
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)

        if return_attention:
            return x, attn
        return x
