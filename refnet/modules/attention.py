from calendar import c
import torch.nn as nn

from einops import rearrange
from refnet.util import exists, default, checkpoint_wrapper
from .layers import RMSNorm
from .attn_utils import *


def create_masked_attention_bias(
        mask: torch.Tensor,
        threshold: float,
        num_heads: int,
        context_len: int
):
    b, seq_len, _ = mask.shape
    half_len = context_len // 2

    if context_len % 8 != 0:
        padded_context_len = ((context_len + 7) // 8) * 8
    else:
        padded_context_len = context_len
    
    fg_bias = torch.zeros(b, seq_len, padded_context_len, device=mask.device, dtype=mask.dtype)
    bg_bias = torch.zeros(b, seq_len, padded_context_len, device=mask.device, dtype=mask.dtype)
    
    fg_bias[:, :, half_len:] = -float('inf')
    bg_bias[:, :, :half_len] = -float('inf')
    attn_bias = torch.where(mask > threshold, fg_bias, bg_bias)
    return attn_bias.unsqueeze(1).repeat_interleave(num_heads, dim=1)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


# Rotary Positional Embeddings implementation
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=1024, theta=10000.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        dim = dim // 2
        self.max_seq_len = max_seq_len
        freqs = torch.outer(
            torch.arange(self.max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freq_h", freqs, persistent=False)
        self.register_buffer("freq_w", freqs, persistent=False)
        
    def forward(self, x, grid_size):
        bs, seq_len, heads = x.shape[:3]
        h, w = grid_size

        x_complex = torch.view_as_complex(
            x.float().reshape(bs, seq_len, heads, -1, 2)
        )
        freqs = torch.cat([
            self.freq_h[:h].view(1, h, 1, -1).expand(bs, h, w, -1),
            self.freq_w[:w].view(1, 1, w, -1).expand(bs, h, w, -1)
        ], dim=-1).reshape(bs, seq_len, 1, -1)

        x_out = x_complex * freqs
        x_out = torch.view_as_real(x_out).flatten(3)
        
        return x_out.type_as(x)


class MemoryEfficientAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
            self,
            query_dim,
            context_dim = None,
            heads = None,
            dim_head = 64,
            dropout = 0.0,
            log = False,
            causal = False,
            rope = False,
            max_seq_len = 1024,
            qk_norm = False,
            **kwargs
    ):
        super().__init__()
        if log:
            print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
                  f"{heads} heads.")

        heads = heads or query_dim // dim_head
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.q_norm = RMSNorm(inner_dim) if qk_norm else Identity()
        self.k_norm = RMSNorm(inner_dim) if qk_norm else Identity()
        self.rope = RotaryPositionalEmbeddings(dim_head, max_seq_len=max_seq_len) if rope else Identity()
        self.attn_ops = causal_ops if causal else {}

        # default setting for split cross-attention
        self.bg_scale = 1.
        self.fg_scale = 1.
        self.merge_scale = 0.
        self.mask_threshold = 0.05

    @checkpoint_wrapper
    def forward(
        self, 
        x, 
        context=None, 
        mask=None, 
        scale=1., 
        scale_factor=None, 
        grid_size=None, 
        **kwargs,
    ):
        context = default(context, x)

        if exists(mask):
            out = self.masked_forward(x, context, mask, scale, scale_factor)
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            out = self.attn_forward(q, k, v, scale, grid_size)

        return self.to_out(out)

    def attn_forward(self, q, k, v, scale=1., grid_size=None, mask=None):
        q, k = map(
            lambda t:
            self.rope(rearrange(t, "b n (h c) -> b n h c", h=self.heads), grid_size),
            (self.q_norm(q), self.k_norm(k))
        )
        v = rearrange(v, "b n (h c) -> b n h c", h=self.heads)
        out = attn_processor(q, k, v, attn_mask=mask, **self.attn_ops) * scale
        out = rearrange(out, "b n h c -> b n (h c)")
        return out

    def masked_forward(self, x, context, mask, scale=1., scale_factor=None):
        # split cross-attention function
        def qkv_forward(x, context):
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
            return q, k, v

        assert exists(scale_factor), "Scale factor must be assigned before masked attention"
        mask = rearrange(
            F.interpolate(mask, scale_factor=scale_factor, mode="bicubic"),
            "b c h w -> b (h w) c"
        ).contiguous()

        if self.merge_scale > 0:
            # split cross-attention with merging scale, need two times forward
            c1, c2 = context.chunk(2, dim=1)

            # Background region cross-attention
            q2, k2, v2 = qkv_forward(x, c2)
            bg_out = self.attn_forward(q2, k2, v2, scale) * self.bg_scale

            # Foreground region cross-attention
            q1, k1, v1 = qkv_forward(x, c1)
            fg_out = self.attn_forward(q1, k1, v1, scale) * self.fg_scale

            fg_out = fg_out * (1 - self.merge_scale) + bg_out * self.merge_scale
            return torch.where(mask < self.mask_threshold, bg_out, fg_out)

        else:
            attn_mask = create_masked_attention_bias(
                mask, self.mask_threshold, self.heads, context.size(1)
            )
            q, k, v = qkv_forward(x, context)
            return self.attn_forward(q, k, v, mask=attn_mask) * scale


class MultiModalAttention(MemoryEfficientAttention):
    def __init__(self, query_dim, context_dim_2, heads=8, dim_head=64, qk_norm=False, *args, **kwargs):
        super().__init__(query_dim, heads=heads, dim_head=dim_head, qk_norm=qk_norm, *args, **kwargs)
        inner_dim = dim_head * heads
        self.to_k_2 = nn.Linear(context_dim_2, inner_dim, bias=False)
        self.to_v_2 = nn.Linear(context_dim_2, inner_dim, bias=False)
        self.k2_norm = RMSNorm(inner_dim) if qk_norm else Identity()

    def forward(self, x, context=None, mask=None, scale=1., grid_size=None):
        if not isinstance(scale, list) and not isinstance(scale, tuple):
            scale = (scale, scale)
        assert len(context.shape) == 4, "Multi-modal attention requires different context inputs to be (b, m, n c)"
        context, context2 = context.chunk(2, dim=1)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        k2 = self.to_k_2(context2)
        v2 = self.to_k_2(context2)

        b, _, _ = q.shape
        q, k, k2 = map(
            lambda t: self.rope(rearrange(t, "b n (h c) -> b n h c", h=self.heads), grid_size),
            (self.q_norm(q), self.k_norm(k), self.k2_norm(k2))
        )
        v, v2 = map(lambda t: rearrange(t, "b n (h c) -> b n h c", h=self.heads), (v, v2))

        out = (attn_processor(q, k, v, **self.attn_ops) * scale[0] +
               attn_processor(q, k2, v2, **self.attn_ops) * scale[1])

        if exists(mask):
            raise NotImplementedError
        out = rearrange(out, "b n h c -> b n (h c)")
        return self.to_out(out)


class MultiScaleCausalAttention(MemoryEfficientAttention):
    def forward(
            self,
            x,
            context=None,
            mask=None,
            scale=1.,
            scale_factor=None,
            grid_size=None,
            token_lens=None
    ):
        context = default(context, x)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        out = self.attn_forward(q, k, v, scale, grid_size=grid_size, token_lens=token_lens)
        return self.to_out(out)

    def attn_forward(self, q, k, v, scale = 1., grid_size = None, token_lens = None):
        q, k, v = map(
            lambda t: rearrange(t, "b n (h c) -> b n h c", h=self.heads),
            (self.q_norm(q), self.k_norm(k), v)
        )

        attn_output = []
        prev_idx = 0
        for idx, (grid, length) in enumerate(zip(grid_size, token_lens)):
            end_idx = prev_idx + length + (idx == 0)
            rope_prev_idx = prev_idx + (idx == 0)
            rope_slice = slice(rope_prev_idx, end_idx)

            q[:, rope_slice] = self.rope(q[:, rope_slice], grid)
            k[:, rope_slice] = self.rope(k[:, rope_slice], grid)
            qs = q[:, prev_idx: end_idx]
            ks, vs = map(lambda t: t[:, :end_idx], (k, v))
            
            attn_output.append(attn_processor(qs.clone(), ks.clone(), vs.clone()) * scale)
            prev_idx = end_idx
        attn_output = rearrange(torch.cat(attn_output, 1), "b n h c -> b n (h c)")
        return attn_output

        # if FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_AVAILABLE:
        #     k_chunks = []
        #     v_chunks = []
        #     kv_token_lens = []
        #     prev_idx = 0
        #     for idx, (grid, length) in enumerate(zip(grid_size, token_lens)):
        #         end_idx = prev_idx + length + (idx == 0)
        #         rope_prev_idx = prev_idx + (idx == 0)

                # rope_slice = slice(rope_prev_idx, end_idx)
                # q[:, rope_slice], k[:, rope_slice], v[:, rope_slice] = map(
                #     lambda t: self.rope(t[:, rope_slice], grid),
                #     (q, k, v)
                # )
                # kv_token_lens.append(end_idx+1)
                # k_chunks.append(k[:, :end_idx])
                # v_chunks.append(v[:, :end_idx])
                # prev_idx = end_idx
            # k = torch.cat(k_chunks, 1)
            # v = torch.cat(v_chunks, 1)
            # B, N, H, C = q.shape
            # token_lens = torch.tensor(token_lens, device=q.device, dtype=torch.int32)
            # kv_token_lens = torch.tensor(kv_token_lens, device=q.device, dtype=torch.int32)
            # token_lens[0] = token_lens[0] + 1
            #
            # cu_seqlens_q, cu_seqlens_kv = map(lambda t:
            #     torch.cat([t.new_zeros([1]), t]).cumsum(0, dtype=torch.int32),
            #     (token_lens, kv_token_lens)
            # )
            # max_seqlen_q, max_seqlen_kv = map(lambda t: int(t.max()), (token_lens, kv_token_lens))
            #
            # q_flat = q.reshape(-1, H, C).contiguous()
            # k_flat = k.reshape(-1, H, C).contiguous()
            # v_flat = v.reshape(-1, H, C).contiguous()
            # out_flat = flash_attn_varlen_func(
            #     q=q_flat, k=k_flat, v=v_flat,
            #     cu_seqlens_q=cu_seqlens_q,
            #     cu_seqlens_k=cu_seqlens_kv,
            #     max_seqlen_q=max_seqlen_q,
            #     max_seqlen_k=max_seqlen_kv,
            #     causal=True,
            # )
            #
            # out = rearrange(out_flat, "(b n) h c -> b n (h c)", b=B, n=N)
            # return out * scale
