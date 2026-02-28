import torch
import torch.nn as nn

from functools import partial
from einops import rearrange

from refnet.util import checkpoint_wrapper, exists
from refnet.modules.layers import FeedForward, Normalize, zero_module, RMSNorm
from refnet.modules.attention import MemoryEfficientAttention, MultiModalAttention, MultiScaleCausalAttention


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "vanilla": MemoryEfficientAttention,
        "multi-scale": MultiScaleCausalAttention,
        "multi-modal": MultiModalAttention,
    }
    def __init__(
            self,
            dim,
            n_heads = None,
            d_head = 64,
            dropout = 0.,
            context_dim = None,
            gated_ff = True,
            ff_mult = 4,
            checkpoint = True,
            disable_self_attn = False,
            disable_cross_attn = False,
            self_attn_type = "vanilla",
            cross_attn_type = "vanilla",
            rotary_positional_embedding = False,
            context_dim_2 = None,
            casual_self_attn = False,
            casual_cross_attn = False,
            qk_norm = False,
            norm_type = "layer",
    ):
        super().__init__()
        assert self_attn_type in self.ATTENTION_MODES
        assert cross_attn_type in self.ATTENTION_MODES
        self_attn_cls = self.ATTENTION_MODES[self_attn_type]
        crossattn_cls = self.ATTENTION_MODES[cross_attn_type]

        if norm_type == "layer":
            norm_cls = nn.LayerNorm
        elif norm_type == "rms":
            norm_cls = RMSNorm
        else:
            raise NotImplementedError(f"Normalization {norm_type} is not implemented.")

        self.dim = dim
        self.disable_self_attn = disable_self_attn
        self.disable_cross_attn = disable_cross_attn

        self.attn1 = self_attn_cls(
            query_dim = dim,
            heads = n_heads,
            dim_head = d_head,
            dropout = dropout,
            context_dim = context_dim if self.disable_self_attn else None,
            casual = casual_self_attn,
            rope = rotary_positional_embedding,
            qk_norm = qk_norm
        )
        self.attn2 = crossattn_cls(
            query_dim = dim,
            context_dim = context_dim,
            context_dim_2 = context_dim_2,
            heads = n_heads,
            dim_head = d_head,
            dropout = dropout,
            casual = casual_cross_attn
        )  if not disable_cross_attn else None

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=ff_mult)
        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim) if not disable_cross_attn else None
        self.norm3 = norm_cls(dim)
        self.reference_scale = 1
        self.scale_factor = None
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x, context=None, mask=None, emb=None, **kwargs):
        x = self.attn1(self.norm1(x), **kwargs) + x
        if not self.disable_cross_attn:
            x = self.attn2(self.norm2(x), context, mask, self.reference_scale, self.scale_factor) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SelfInjectedTransformerBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank = None
        self.time_proj = None
        self.injection_type = "concat"
        self.forward_without_bank = super().forward

    @checkpoint_wrapper
    def forward(self, x, context=None, mask=None, emb=None, **kwargs):
        if exists(self.bank):
            bank = self.bank
            if bank.shape[0] != x.shape[0]:
                bank = bank.repeat(x.shape[0], 1, 1)
            if exists(self.time_proj) and exists(emb):
                bank = bank + self.time_proj(emb).unsqueeze(1)
            x_in = self.norm1(x)

            self.attn1.mask_threshold = self.attn2.mask_threshold
            x = self.attn1(
                x_in,
                torch.cat([x_in, bank], 1) if self.injection_type == "concat" else x_in + bank,
                mask = mask,
                scale_factor = self.scale_factor,
                **kwargs
            ) + x

            x = self.attn2(
                self.norm2(x),
                context,
                mask = mask,
                scale = self.reference_scale,
                scale_factor = self.scale_factor
            ) + x

            x = self.ff(self.norm3(x)) + x
        else:
            x = self.forward_without_bank(x, context, mask, emb)
        return x


class SelfTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head = 64,
            dropout = 0.,
            mlp_ratio = 4,
            checkpoint = True,
            casual_attn = False,
            reshape = True
    ):
        super().__init__()
        self.attn = MemoryEfficientAttention(query_dim=dim, heads=dim//dim_head, dropout=dropout, casual=casual_attn)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            zero_module(nn.Linear(dim * mlp_ratio, dim))
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.reshape = reshape
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x, context=None):
        b, c, h, w = x.shape
        if self.reshape:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        x = self.attn(self.norm1(x), context if exists(context) else None) + x
        x = self.ff(self.norm2(x)) + x

        if self.reshape:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x


class Transformer(nn.Module):
    transformer_type = {
        "vanilla": BasicTransformerBlock,
        "self-injection": SelfInjectedTransformerBlock,
    }
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_linear=False,
                 use_checkpoint=True, type="vanilla", transformer_config=None, **kwargs):
        super().__init__()
        transformer_block = self.transformer_type[type]
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        if isinstance(context_dim, list):
            if depth != len(context_dim):
                context_dim = depth * [context_dim[0]]

        proj_layer = nn.Linear if use_linear else partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
        inner_dim = n_heads * d_head

        self.in_channels = in_channels
        self.proj_in = proj_layer(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList([
            transformer_block(
                inner_dim,
                n_heads,
                d_head,
                dropout = dropout,
                context_dim = context_dim[d],
                checkpoint = use_checkpoint,
                **(transformer_config or {}),
                **kwargs
            ) for d in range(depth)
        ])
        self.proj_out = zero_module(proj_layer(inner_dim, in_channels))
        self.norm = Normalize(in_channels)
        self.use_linear = use_linear

    def forward(self, x, context=None, mask=None, emb=None, *args, **additional_context):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, mask=mask, emb=emb, grid_size=(h, w), *args, **additional_context)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def SpatialTransformer(*args, **kwargs):
    return Transformer(type="vanilla", *args, **kwargs)

def SelfInjectTransformer(*args, **kwargs):
    return Transformer(type="self-injection", *args, **kwargs)
