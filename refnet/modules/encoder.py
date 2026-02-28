import torch
import torch.nn as nn
import torch.nn.functional as F

from refnet.util import checkpoint_wrapper
from refnet.modules.unet import TimestepEmbedSequential
from refnet.modules.layers import Upsample, zero_module, RMSNorm, FeedForward
from refnet.modules.attention import MemoryEfficientAttention, MultiScaleCausalAttention
from einops import rearrange
from functools import partial



def make_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels or in_channels
    return zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))

def activate_zero_conv(in_channels, out_channels=None):
    out_channels = out_channels or in_channels
    return TimestepEmbedSequential(
        nn.SiLU(),
        zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0))
    )

def sequential_downsample(in_channels, out_channels, sequential_cls=nn.Sequential):
    return sequential_cls(
        nn.Conv2d(in_channels, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 32, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 96, 3, padding=1, stride=2),
        nn.SiLU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(96, 256, 3, padding=1, stride=2),
        nn.SiLU(),
        zero_module(nn.Conv2d(256, out_channels, 3, padding=1))
    )


class SimpleEncoder(nn.Module):
    def __init__(self, c_channels, model_channels):
        super().__init__()
        self.model = sequential_downsample(c_channels, model_channels)

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class MultiEncoder(nn.Module):
    def __init__(self, in_ch, model_channels, ch_mults, checkpoint=True, time_embed=False):
        super().__init__()
        sequential_cls = TimestepEmbedSequential if time_embed else nn.Sequential
        output_chs = [model_channels * mult for mult in ch_mults]
        self.model = sequential_downsample(in_ch, model_channels, sequential_cls)
        self.zero_layer = make_zero_conv(output_chs[0])
        self.output_blocks = nn.ModuleList()
        self.zero_blocks = nn.ModuleList()

        block_num = len(ch_mults)
        prev_ch = output_chs[0]
        for i in range(block_num):
            self.output_blocks.append(sequential_cls(
                nn.SiLU(),
                nn.Conv2d(prev_ch, output_chs[i], 3, padding=1, stride=2 if i != block_num-1 else 1),
                nn.SiLU(),
                nn.Conv2d(output_chs[i], output_chs[i], 3, padding=1)
            ))
            self.zero_blocks.append(
                TimestepEmbedSequential(make_zero_conv(output_chs[i])) if time_embed
                else make_zero_conv(output_chs[i])
            )
            prev_ch = output_chs[i]

        self.checkpoint = checkpoint

    def forward(self, x):
        x = self.model(x)
        hints = [self.zero_layer(x)]
        for layer, zero_layer in zip(self.output_blocks, self.zero_blocks):
            x = layer(x)
            hints.append(zero_layer(x))
        return hints


class MultiScaleAttentionEncoder(nn.Module):
    def __init__(
            self,
            in_ch,
            model_channels,
            ch_mults,
            dim_head = 128,
            transformer_layers = 2,
            checkpoint = True
    ):
        super().__init__()
        conv_proj = partial(nn.Conv2d, kernel_size=1, padding=0)
        output_chs = [model_channels * mult for mult in ch_mults]
        block_num = len(ch_mults)
        attn_ch = output_chs[-1]

        self.model = sequential_downsample(in_ch, output_chs[0])
        self.proj_ins = nn.ModuleList([conv_proj(output_chs[0], attn_ch)])
        self.proj_outs = nn.ModuleList([zero_module(conv_proj(attn_ch, output_chs[0]))])

        prev_ch = output_chs[0]
        self.downsample_layers = nn.ModuleList()
        for i in range(block_num):
            ch = output_chs[i]
            self.downsample_layers.append(nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(prev_ch, ch, 3, padding=1, stride=2 if i != block_num - 1 else 1),
            ))
            self.proj_ins.append(conv_proj(ch, attn_ch))
            self.proj_outs.append(zero_module(conv_proj(attn_ch, ch)))
            prev_ch = ch

        self.proj_ins.append(conv_proj(attn_ch, attn_ch))
        self.attn_layer = MultiScaleCausalAttention(attn_ch, rope=True, qk_norm=True, dim_head=dim_head)
        # self.transformer = nn.ModuleList([
        #     BasicTransformerBlock(
        #         attn_ch,
        #         rotary_positional_embedding = True,
        #         qk_norm = True,
        #         d_head = dim_head,
        #         disable_cross_attn = True,
        #         self_attn_type = "multi-scale",
        #         ff_mult = 2,
        #     )
        # ] * transformer_layers)
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x):
        proj_in_iter = iter(self.proj_ins)
        proj_out_iter = iter(self.proj_outs[::-1])

        x = self.model(x)
        hints = [rearrange(next(proj_in_iter)(x), "b c h w -> b (h w) c")]
        grid_sizes = [(x.shape[2], x.shape[3])]
        token_lens = [(x.shape[2] * x.shape[3])]

        for layer in self.downsample_layers:
            x = layer(x)
            h, w = x.shape[2], x.shape[3]
            grid_sizes.append((h, w))
            token_lens.append(h * w)
            hints.append(rearrange(next(proj_in_iter)(x), "b c h w -> b (h w) c"))

        hints.append(rearrange(
            next(proj_in_iter)(x.mean(dim=[2, 3], keepdim=True)),
            "b c h w -> b (h w) c"
        ))

        hints = hints[::-1]
        grid_sizes = grid_sizes[::-1]
        token_lens = token_lens[::-1]
        hints = torch.cat(hints, 1)
        hints = self.attn_layer(hints, grid_size=grid_sizes, token_lens=token_lens) + hints
        # for layer in self.transformer:
        #     hints = layer(hints, grid_size=grid_sizes, token_lens=token_lens)

        prev_idx = 1
        controls = []
        for gs, token_len in zip(grid_sizes, token_lens):
            control = hints[:, prev_idx: prev_idx + token_len]
            control = rearrange(control, "b (h w) c -> b c h w", h=gs[0], w=gs[1])
            controls.append(next(proj_out_iter)(control))
            prev_idx = prev_idx + token_len
        return controls[::-1]



class Downsampler(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic")


class SpatialConditionEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            dim,
            out_dim,
            patch_size,
            n_layers = 4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.conv = nn.Sequential(nn.SiLU(), nn.Conv2d(dim, dim, kernel_size=3, padding=1))

        self.transformer = nn.ModuleList(
            nn.ModuleList([
                RMSNorm(dim),
                MemoryEfficientAttention(dim, rope=True),
                RMSNorm(dim),
                FeedForward(dim, mult=2)
            ]) for _ in range(n_layers)
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Conv2d(dim, out_dim, kernel_size=1, padding=0))
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.conv(x)

        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        for norm, layer, norm2, ff in self.transformer:
            x = layer(norm(x), grid_size=(h, w)) + x
            x = ff(norm2(x)) + x
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return self.out(x)
