import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
from openstl.utils import measure_throughput
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from openstl.modules import Attention, CrossAttention, PreNorm, FeedForward
import math

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
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
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for attn, ff, drop_path1,drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')
      
class PredFormer_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        #修改点：加入空间裁剪后的高度和宽度
        # When spatial tiling/cropping is enabled, use the cropped spatial
        # dimensions for patching and positional embeddings so they match the
        # runtime input shape.
        self.patch_size = model_config['patch_size']
        self.valid_height = model_config.get('space_h', model_config['height'])
        self.valid_width = model_config.get('space_w', model_config['width'])
        self.pad_to_patch = bool(model_config.get('pad_to_patch', False))
        if self.pad_to_patch:
            self.image_height = ((self.valid_height + self.patch_size - 1) // self.patch_size) * self.patch_size
            self.image_width = ((self.valid_width + self.patch_size - 1) // self.patch_size) * self.patch_size
        else:
            self.image_height = self.valid_height
            self.image_width = self.valid_width
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = model_config['pre_seq']
        self.dim = model_config['dim']
        self.in_channels = model_config.get('in_channels', model_config.get('num_channels'))
        self.input_channels = model_config.get('input_channels', self.in_channels)
        self.dynamic_channels = model_config.get('dynamic_channels', None)
        self.static_in_channels = model_config.get('static_in_channels', None)
        self.static_out_channels = model_config.get('static_out_channels', None)
        self.static_kernel_size = model_config.get('static_kernel_size', 3)
        self.static_proj = None
        if self.static_in_channels is not None and self.static_out_channels is not None:
            if not isinstance(self.static_kernel_size, int) or self.static_kernel_size < 1:
                raise ValueError(f"static_kernel_size must be a positive int, got {self.static_kernel_size}")
            if self.static_kernel_size % 2 == 0:
                raise ValueError(
                    f"static_kernel_size={self.static_kernel_size} must be odd to keep spatial size with padding."
                )
            if self.dynamic_channels is None:
                self.dynamic_channels = self.input_channels - self.static_in_channels
            expected_in = self.dynamic_channels + self.static_in_channels
            if self.input_channels != expected_in:
                raise ValueError(
                    f"input_channels={self.input_channels} does not match "
                    f"dynamic_channels+static_in_channels={expected_in}."
                )
            projected_in = self.dynamic_channels + self.static_out_channels
            if self.in_channels != projected_in:
                raise ValueError(
                    f"in_channels={self.in_channels} does not match "
                    f"dynamic_channels+static_out_channels={projected_in}."
                )
            self.static_proj = nn.Conv2d(
                self.static_in_channels,
                self.static_out_channels,
                kernel_size=self.static_kernel_size,
                stride=1,
                padding=self.static_kernel_size // 2,
            )
        if self.dynamic_channels is None:
            if self.static_in_channels is not None:
                self.dynamic_channels = self.input_channels - self.static_in_channels
            else:
                self.dynamic_channels = self.in_channels
        self.out_channels = model_config.get('out_channels', self.in_channels)
        self.num_classes = self.out_channels
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.attn_dropout = model_config['attn_dropout']
        self.drop_path = model_config['drop_path']
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']  # Ensure this is defined
        self.depth = model_config['depth']  # Ensure this is defined
        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.in_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )
        self.attn_type = model_config.get('attn_type', model_config.get('pre_attn_type', 'none'))
        if self.attn_type not in {'none', 'pre_cross', 'post_cross'}:
            raise ValueError(f"Unknown attn_type={self.attn_type}")
        self.pre_cross_attn = None
        self.pre_cross_norm_q = None
        self.pre_cross_norm_kv = None
        self.to_patch_embedding_dyn = None
        self.to_patch_embedding_static = None
        if self.attn_type in {'pre_cross', 'post_cross'}:
            static_channels = self.static_out_channels if self.static_proj is not None else self.static_in_channels
            if static_channels is None:
                raise ValueError(f"attn_type='{self.attn_type}' requires static channels.")
            dyn_patch_dim = self.dynamic_channels * self.patch_size ** 2
            sta_patch_dim = static_channels * self.patch_size ** 2
            self.to_patch_embedding_dyn = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
                nn.Linear(dyn_patch_dim, self.dim),
            )
            self.to_patch_embedding_static = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
                nn.Linear(sta_patch_dim, self.dim),
            )
            self.pre_cross_norm_q = nn.LayerNorm(self.dim)
            self.pre_cross_norm_kv = nn.LayerNorm(self.dim)
            self.pre_cross_attn = CrossAttention(
                self.dim, heads=self.heads, dim_head=self.dim_head, dropout=self.attn_dropout
            )
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                               requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.space_transformer = GatedTransformer(self.dim, self.Ndepth, self.heads, self.dim_head, 
                                                self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
        self.temporal_transformer = GatedTransformer(self.dim, self.Ndepth, self.heads, self.dim_head, 
                                                self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_channels * self.patch_size ** 2)
            ) 
                

    def forward(self, x):
        B, T, C, H, W = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected input channels={self.input_channels}, got {C}")
        if H != self.image_height or W != self.image_width:
            raise ValueError(
                f"Expected spatial size {(self.image_height, self.image_width)}, got {(H, W)}. "
                f"Check pad_to_patch={self.pad_to_patch}, patch_size={self.patch_size}, "
                f"and dataloader padding settings."
            )
        dyn = None
        static = None
        if self.static_in_channels is not None:
            dyn = x[:, :, :self.dynamic_channels]
            static = x[:, :, self.dynamic_channels:self.dynamic_channels + self.static_in_channels]
        if self.static_proj is not None:
            static = static.reshape(B * T, self.static_in_channels, H, W)
            static = self.static_proj(static)
            static = static.reshape(B, T, self.static_out_channels, H, W)
            x = torch.cat([dyn, static], dim=2)
        if x.shape[2] != self.in_channels:
            raise ValueError(f"Expected projected channels={self.in_channels}, got {x.shape[2]}")
        
        pos = self.pos_embedding.to(x.device)
        if self.attn_type == 'pre_cross':
            if dyn is None or static is None:
                raise ValueError("attn_type='pre_cross' requires static channels.")
            dyn_tok = self.to_patch_embedding_dyn(dyn)
            static_tok = self.to_patch_embedding_static(static)
            dyn_tok = dyn_tok + pos
            static_tok = static_tok + pos
            b, t, n, _ = dyn_tok.shape
            q = dyn_tok.reshape(B * T, n, self.dim)
            kv = static_tok.reshape(B * T, n, self.dim)
            q = q + self.pre_cross_attn(
                self.pre_cross_norm_q(q),
                self.pre_cross_norm_kv(kv),
            )
            x = q.reshape(B, T, n, self.dim)
        else:
            x = self.to_patch_embedding(x)
            x = x + pos
            b, t, n, _ = x.shape
            # attn_type == 'none' or 'post_cross' falls through without extra attention
        
        # ts-t branch
        x_t = rearrange(x, 'b t n d -> b n t d')
        x_t = rearrange(x_t, 'b n t d -> (b n) t d')
        x_t = self.temporal_transformer(x_t)
        
        # ts-s branch
        x_ts = rearrange(x_t, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        x_ts = self.space_transformer(x_ts)

        if self.attn_type == 'post_cross':
            if dyn is None or static is None:
                raise ValueError("attn_type='post_cross' requires static channels.")
            # Use transformer output as dynamic tokens (query)
            dyn_tok = x_ts.reshape(B, T, n, self.dim)
            static_tok = self.to_patch_embedding_static(static)
            static_tok = static_tok + pos
            q = dyn_tok.reshape(B * T, n, self.dim)
            kv = static_tok.reshape(B * T, n, self.dim)
            q = q + self.pre_cross_attn(
                self.pre_cross_norm_q(q),
                self.pre_cross_norm_kv(kv),
            )
            x_ts = q.reshape(B * T, n, self.dim)
            
        # MLP head        
        x = self.mlp_head(x_ts.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, self.out_channels, H, W)
        
        return x
    
