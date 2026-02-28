import torch
import torch.nn as nn

from refnet.modules.layers import zero_module
from refnet.modules.attention import MemoryEfficientAttention
from refnet.modules.transformer import BasicTransformerBlock
from refnet.util import checkpoint_wrapper, exists
from ckpt_util import load_weights


class NormalizedLinear(nn.Module):
    def __init__(self, dim, output_dim, checkpoint=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x):
        return self.layers(x)


class GlobalProjection(nn.Module):
    def __init__(self, input_dim, output_dim, heads, dim_head=128, checkpoint=True):
        super().__init__()
        self.c_dim = output_dim
        self.dim_head = dim_head
        self.head = (heads[0], heads[0] * heads[1])

        self.proj1 = nn.Linear(input_dim, dim_head * heads[0])
        self.proj2 = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(dim_head, output_dim * heads[1])),
        )
        self.norm = nn.LayerNorm(output_dim)
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x):
        x = self.proj1(x).reshape(-1, self.head[0], self.dim_head).contiguous()
        x = self.proj2(x).reshape(-1, self.head[1], self.c_dim).contiguous()
        return self.norm(x)


class ClusterConcat(nn.Module):
    def __init__(self, input_dim, c_dim, output_dim, dim_head=64, token_length=196, checkpoint=True):
        super().__init__()
        self.attn = MemoryEfficientAttention(input_dim, dim_head=dim_head)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim + c_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.token_length = token_length
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x, emb, fgbg=False, *args, **kwargs):
        x = self.attn(x)[:, :self.token_length]
        x = self.norm(x)
        x = torch.cat([x, emb], 2)
        x = self.proj(x)

        if fgbg:
            x = torch.cat(torch.chunk(x, 2), 1)
        return x


class RecoveryClusterConcat(ClusterConcat):
    def __init__(self, input_dim, c_dim, output_dim, dim_head=64, *args, **kwargs):
        super().__init__(input_dim, c_dim, output_dim, dim_head=dim_head, *args, **kwargs)
        self.transformer = BasicTransformerBlock(
            output_dim, output_dim//dim_head, dim_head,
            disable_cross_attn=True, checkpoint=False
        )

    @checkpoint_wrapper
    def forward(self, x, emb, bg=False):
        x = self.attn(x)[:, :self.token_length]
        x = self.norm(x)
        x = torch.cat([x, emb], 2)
        x = self.proj(x)

        if bg:
            x = self.transformer(x)
        return x


class LogitClusterConcat(ClusterConcat):
    def __init__(self, c_dim, mlp_in_dim, mlp_ckpt_path=None, *args, **kwargs):
        super().__init__(c_dim=c_dim, *args, **kwargs)
        self.mlp = AdaptiveMLP(c_dim, mlp_in_dim)
        if exists(mlp_ckpt_path):
            self.mlp.load_state_dict(load_weights(mlp_ckpt_path), strict=True)

    @checkpoint_wrapper
    def forward(self, x, emb, bg=False):
        with torch.no_grad():
            emb = self.mlp(emb).detach()
        return super().forward(x, emb, bg)


class AdaptiveMLP(nn.Module):
    def __init__(self, dim, in_dim, layers=4, checkpoint=True):
        super().__init__()

        model = [nn.Sequential(nn.Linear(in_dim, dim))]
        for i in range(1, layers):
            model += [nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )]
        self.mlp = nn.Sequential(*model)
        self.fusion_layer = nn.Linear(dim * layers, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    @checkpoint_wrapper
    def forward(self, x):
        fx = []

        for layer in self.mlp:
            x = layer(x)
            fx.append(x)

        x = torch.cat(fx, dim=2)
        out = self.fusion_layer(x)
        out = self.norm(out)
        return out


class Concat(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, y, *args, **kwargs):
        return torch.cat([x, y], dim=-1)