import math

import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

from openstl.modules import Attention, PreNorm


def _align_to_patch(size, patch_size):
    return int(math.ceil(size / patch_size) * patch_size)


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x = self.act(self.fc1_g(x)) * self.fc1_x(x)
        return self.drop2(self.fc2(self.norm(self.drop1(x))))


class GatedTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        for attention, feed_forward, drop_attention, drop_feed_forward in self.layers:
            x = x + drop_attention(attention(x))
            x = x + drop_feed_forward(feed_forward(x))
        return self.norm(x)


def sinusoidal_embedding(length, dim):
    embedding = torch.FloatTensor([
        [position / (10000 ** (2 * (index // 2) / dim)) for index in range(dim)]
        for position in range(length)
    ])
    embedding[:, 0::2] = torch.sin(embedding[:, 0::2])
    embedding[:, 1::2] = torch.cos(embedding[:, 1::2])
    return embedding.unsqueeze(0)


