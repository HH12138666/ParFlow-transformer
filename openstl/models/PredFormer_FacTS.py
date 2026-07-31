import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from .PredFormer_layers import GatedTransformer, sinusoidal_embedding
from torch import nn

from openstl.modules import CrossAttention


def _align_to_patch(size, patch_size):
    return int(math.ceil(size / patch_size) * patch_size)


class PredFormer_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self._init_dimensions(model_config)
        self._init_channels(model_config)
        self._init_hyperparameters(model_config)
        self._init_patch_embeddings()
        self._init_cross_attention()
        self._init_transformers()

    def _init_dimensions(self, config):
        self.patch_size = int(config["patch_size"])
        self.valid_height = int(config.get("space_h") or config["height"])
        self.valid_width = int(config.get("space_w") or config["width"])
        self.image_height = _align_to_patch(self.valid_height, self.patch_size)
        self.image_width = _align_to_patch(self.valid_width, self.patch_size)
        self.pad_h = self.image_height - self.valid_height
        self.pad_w = self.image_width - self.valid_width
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = int(config["pre_seq"])
        requested_frames_out = int(config["after_seq"])
        if requested_frames_out <= 0:
            raise ValueError(f"after_seq must be positive, got {requested_frames_out}")
        # Long horizons are assembled from pre_seq-sized autoregressive blocks.
        self.num_frames_out = min(requested_frames_out, self.num_frames_in)

    def _init_channels(self, config):
        self.in_channels = int(config.get("in_channels", config.get("num_channels")))
        self.input_channels = int(config.get("input_channels", self.in_channels))
        self.static_in_channels = config.get("static_in_channels")
        self.static_out_channels = config.get("static_out_channels")
        self.dynamic_channels = config.get("dynamic_channels")
        if self.dynamic_channels is None:
            static_channels = self.static_in_channels or 0
            self.dynamic_channels = self.input_channels - static_channels
        self.out_channels = int(config.get("out_channels", self.in_channels))
        self.num_classes = self.out_channels
        self.static_kernel_size = int(config.get("static_kernel_size", 3))
        self.static_proj = self._build_static_projection()

    def _build_static_projection(self):
        if self.static_in_channels is None or self.static_out_channels is None:
            self._validate_projected_channels()
            return None
        if self.static_kernel_size < 1 or self.static_kernel_size % 2 == 0:
            raise ValueError("static_kernel_size must be a positive odd integer")
        self._validate_projected_channels()
        return nn.Conv2d(
            self.static_in_channels,
            self.static_out_channels,
            kernel_size=self.static_kernel_size,
            padding=self.static_kernel_size // 2,
        )

    def _validate_projected_channels(self):
        raw_expected = self.dynamic_channels + (self.static_in_channels or 0)
        if self.input_channels != raw_expected:
            raise ValueError(
                f"input_channels={self.input_channels}, expected {raw_expected}"
            )
        projected_static = (
            self.static_out_channels
            if self.static_out_channels is not None
            else (self.static_in_channels or 0)
        )
        projected_expected = self.dynamic_channels + projected_static
        if self.in_channels != projected_expected:
            raise ValueError(f"in_channels={self.in_channels}, expected {projected_expected}")

    def _init_hyperparameters(self, config):
        self.dim = int(config["dim"])
        self.heads = int(config["heads"])
        self.dim_head = int(config["dim_head"])
        self.dropout = float(config["dropout"])
        self.attn_dropout = float(config["attn_dropout"])
        self.drop_path = float(config["drop_path"])
        self.scale_dim = int(config["scale_dim"])
        self.Ndepth = int(config["Ndepth"])
        self.depth = int(config["depth"])
        self.attn_type = config.get("attn_type", config.get("pre_attn_type", "none"))
        if self.attn_type not in {"none", "pre_cross", "post_cross"}:
            raise ValueError(f"Unknown attn_type={self.attn_type}")

    def _patch_embedding(self, channels):
        patch_dim = channels * self.patch_size**2
        return nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )

    def _init_patch_embeddings(self):
        self.patch_dim = self.in_channels * self.patch_size**2
        self.to_patch_embedding = self._patch_embedding(self.in_channels)
        self.to_patch_embedding_dyn = None
        self.to_patch_embedding_static = None

    def _init_cross_attention(self):
        self.pre_cross_attn = None
        self.pre_cross_norm_q = None
        self.pre_cross_norm_kv = None
        if self.attn_type == "none":
            return
        static_channels = self.static_out_channels or self.static_in_channels
        if static_channels is None:
            raise ValueError(f"attn_type={self.attn_type} requires static channels")
        self.to_patch_embedding_dyn = self._patch_embedding(self.dynamic_channels)
        self.to_patch_embedding_static = self._patch_embedding(static_channels)
        self.pre_cross_norm_q = nn.LayerNorm(self.dim)
        self.pre_cross_norm_kv = nn.LayerNorm(self.dim)
        self.pre_cross_attn = CrossAttention(
            self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.attn_dropout,
        )

    def _init_transformers(self):
        embedding = sinusoidal_embedding(
            self.num_frames_in * self.num_patches, self.dim
        )
        self.pos_embedding = nn.Parameter(
            embedding, requires_grad=False
        ).view(1, self.num_frames_in, self.num_patches, self.dim)
        transformer_args = (
            self.dim,
            self.Ndepth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout,
            self.attn_dropout,
            self.drop_path,
        )
        self.space_transformer = GatedTransformer(*transformer_args)
        self.temporal_transformer = GatedTransformer(*transformer_args)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_channels * self.patch_size**2),
        )

    def forward(self, x):
        residual_input = x
        x, dynamic, static, input_shape = self._prepare_input(x)
        tokens = self._embed_input(x, dynamic, static)
        tokens = self._transform(tokens, static)
        prediction = self._decode(tokens, input_shape)
        return self._add_residual(prediction, residual_input)

    def _prepare_input(self, x):
        batch, frames, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected C={self.input_channels}, got {channels}")
        valid = (height, width) == (self.valid_height, self.valid_width)
        padded = (height, width) == (self.image_height, self.image_width)
        if not valid and not padded:
            raise ValueError(f"Unexpected spatial size {(height, width)}")
        if valid and (self.pad_h or self.pad_w):
            x = x.reshape(batch * frames, channels, height, width)
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="replicate")
            x = x.reshape(
                batch, frames, channels, self.image_height, self.image_width
            )
        dynamic, static = self._split_channels(x)
        if self.static_proj is not None:
            static = self._project_static(static, batch, frames)
            x = torch.cat([dynamic, static], dim=2)
        if x.shape[2] != self.in_channels:
            raise ValueError(f"Expected projected C={self.in_channels}, got {x.shape[2]}")
        return x, dynamic, static, (batch, frames, height, width, valid)

    def _split_channels(self, x):
        if self.static_in_channels is None:
            return None, None
        dynamic = x[:, :, : self.dynamic_channels]
        static = x[:, :, self.dynamic_channels :]
        return dynamic, static

    def _project_static(self, static, batch, frames):
        static = self.static_proj(static[:, 0])
        return static.unsqueeze(1).expand(
            batch,
            frames,
            self.static_out_channels,
            self.image_height,
            self.image_width,
        )

    def _embed_input(self, x, dynamic, static):
        pos = self.pos_embedding.to(x.device)
        if self.attn_type != "pre_cross":
            return self.to_patch_embedding(x) + pos
        return self._cross_tokens(dynamic, static, pos)

    def _cross_tokens(self, dynamic, static, pos):
        if dynamic is None or static is None:
            raise ValueError(f"attn_type={self.attn_type} requires static channels")
        dynamic_tokens = self.to_patch_embedding_dyn(dynamic) + pos
        static_tokens = self.to_patch_embedding_static(static) + pos
        batch, frames, patches, _ = dynamic_tokens.shape
        query = dynamic_tokens.reshape(batch * frames, patches, self.dim)
        key_value = static_tokens.reshape(batch * frames, patches, self.dim)
        query = query + self.pre_cross_attn(
            self.pre_cross_norm_q(query),
            self.pre_cross_norm_kv(key_value),
        )
        return query.reshape(batch, frames, patches, self.dim)

    def _transform(self, tokens, static):
        batch, _, patches, _ = tokens.shape
        temporal = rearrange(tokens, "b t n d -> (b n) t d")
        temporal = self.temporal_transformer(temporal)
        temporal = rearrange(
            temporal, "(b n) t d -> b t n d", b=batch, n=patches
        )[:, : self.num_frames_out]
        spatial = rearrange(temporal, "b t n d -> (b t) n d")
        spatial = self.space_transformer(spatial)
        if self.attn_type != "post_cross":
            return spatial
        pos = self.pos_embedding[:, : self.num_frames_out].to(spatial.device)
        static = static[:, : self.num_frames_out]
        static_tokens = self.to_patch_embedding_static(static) + pos
        key_value = rearrange(static_tokens, "b t n d -> (b t) n d")
        return spatial + self.pre_cross_attn(
            self.pre_cross_norm_q(spatial),
            self.pre_cross_norm_kv(key_value),
        )

    def _decode(self, tokens, input_shape):
        batch, _, height, width, valid = input_shape
        frames = self.num_frames_out
        x = self.mlp_head(tokens.reshape(-1, self.dim))
        x = x.view(
            batch,
            frames,
            self.num_patches_h,
            self.num_patches_w,
            self.out_channels,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        x = x.reshape(
            batch,
            frames,
            self.out_channels,
            self.image_height,
            self.image_width,
        )
        if valid and (self.pad_h or self.pad_w):
            return x[:, :, :, :height, :width]
        return x

    def _add_residual(self, prediction, residual_input):
        if residual_input.shape[2] < self.out_channels:
            raise ValueError("Residual input has fewer channels than model output")
        base = residual_input[:, -1, : self.out_channels].unsqueeze(1)
        return base + prediction
