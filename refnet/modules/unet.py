import torch
import torch.nn as nn

from functools import partial
from refnet.modules.attention import MemoryEfficientAttention
from refnet.util import exists
from refnet.modules.transformer import (
    SelfTransformerBlock,
    Transformer,
    SpatialTransformer,
    SelfInjectTransformer,
)
from refnet.ldm.openaimodel import (
    timestep_embedding,
    conv_nd,
    TimestepBlock,
    zero_module,
    ResBlock,
    linear,
    Downsample,
    Upsample,
    normalization,
)


def hack_inference_forward(model):
    model.forward = InferenceForward.__get__(model, model.__class__)


def InferenceForward(self, x, timesteps=None, y=None, *args, **kwargs):
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb).to(self.dtype)
    assert (y is not None) == (
            self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y.to(emb.device))
    emb = emb.to(self.dtype)
    h = self._forward(x, emb, *args, **kwargs)
    return self.out(h.to(x.dtype))


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    # Dispatch constants
    _D_TIMESTEP = 0
    _D_TRANSFORMER = 1
    _D_OTHER = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache dispatch types at init (before FSDP wrapping), so forward()
        # needs no isinstance checks and is immune to FSDP wrapper breakage.
        self._dispatch = tuple(
            self._D_TIMESTEP if isinstance(layer, TimestepBlock) else
            self._D_TRANSFORMER if isinstance(layer, Transformer) else
            self._D_OTHER
            for layer in self
        )

    def forward(self, x, emb=None, context=None, mask=None, **additional_context):
        for layer, d in zip(self, self._dispatch):
            if d == self._D_TIMESTEP:
                x = layer(x, emb)
            elif d == self._D_TRANSFORMER:
                x = layer(x, context, mask, emb, **additional_context)
            else:
                x = layer(x)
        return x



class UNetModel(nn.Module):
    transformers = {
        "vanilla": SpatialTransformer,
        "selfinj": SelfInjectTransformer,
    }
    def __init__(
            self,
            in_channels,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            out_channels = 4,
            dropout = 0,
            channel_mult = (1, 2, 4, 8),
            conv_resample = True,
            dims = 2,
            num_classes = None,
            use_checkpoint = False,
            num_heads = -1,
            num_head_channels = -1,
            use_scale_shift_norm = False,
            resblock_updown = False,
            use_spatial_transformer = False,  # custom transformer support
            transformer_depth = 1,  # custom transformer support
            context_dim = None,  # custom transformer support
            disable_self_attentions = None,
            disable_cross_attentions = False,
            num_attention_blocks = None,
            use_linear_in_transformer = False,
            adm_in_channels = None,
            transformer_type = "vanilla",
            map_module = False,
            warp_module = False,
            style_modulation = False,
            discard_final_layers = False,   # for reference net
            additional_transformer_config = None,
            in_channels_fg = None,
            in_channels_bg = None,
    ):
        super().__init__()
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        assert num_heads > -1 or num_head_channels > -1, 'Either num_heads or num_head_channels has to be set'
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.num_classes = num_classes
        self.model_channels = model_channels
        self.dtype = torch.float32

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth[-1]
        time_embed_dim = model_channels * 4
        resblock = partial(
            ResBlock,
            emb_channels = time_embed_dim,
            dropout = dropout,
            dims = dims,
            use_checkpoint = use_checkpoint,
            use_scale_shift_norm = use_scale_shift_norm,
        )
        transformer = partial(
            self.transformers[transformer_type],
            context_dim = context_dim,
            use_linear = use_linear_in_transformer,
            use_checkpoint = use_checkpoint,
            disable_self_attn = disable_self_attentions,
            disable_cross_attn = disable_cross_attentions,
            transformer_config = additional_transformer_config
        )

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [resblock(ch, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels > -1:
                        current_num_heads = ch // num_head_channels
                        current_head_dim = num_head_channels
                    else:
                        current_num_heads = num_heads
                        current_head_dim = ch // num_heads

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, current_head_dim)
                            if not use_spatial_transformer
                            else transformer(
                                ch, current_num_heads, current_head_dim,
                                depth=transformer_depth[level],
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    resblock(ch, out_channels=out_ch, down=True) if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels > -1:
            current_num_heads = ch // num_head_channels
            current_head_dim = num_head_channels
        else:
            current_num_heads = num_heads
            current_head_dim = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            resblock(ch),
            SelfTransformerBlock(ch, current_head_dim) if not use_spatial_transformer
            else transformer(ch, current_num_heads, current_head_dim, depth=transformer_depth_middle),
            resblock(ch),
        )

        self.output_blocks = nn.ModuleList([])
        self.map_modules = nn.ModuleList([])
        self.warp_modules = nn.ModuleList([])
        self.style_modules = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [resblock(ch + ich, out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels > -1:
                        current_num_heads = ch // num_head_channels
                        current_head_dim = num_head_channels
                    else:
                        current_num_heads = num_heads
                        current_head_dim = ch // num_heads

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            SelfTransformerBlock(ch, current_head_dim) if not use_spatial_transformer
                            else transformer(
                                ch, current_num_heads, current_head_dim, depth=transformer_depth[level]
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        resblock(ch, up=True) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                if level == 0 and discard_final_layers:
                    break

                if map_module:
                    self.map_modules.append(nn.ModuleList([
                        MemoryEfficientAttention(
                            ich,
                            heads = ich // num_head_channels,
                            dim_head = num_head_channels
                        ),
                        nn.Linear(time_embed_dim, ich)
                    ]))

                if warp_module:
                    self.warp_modules.append(nn.ModuleList([
                        MemoryEfficientAttention(
                            ich,
                            heads = ich // num_head_channels,
                            dim_head = num_head_channels
                        ),
                        nn.Linear(time_embed_dim, ich)
                    ]))

                    # self.warp_modules.append(nn.ModuleList([
                    #     SpatialTransformer(ich, ich//num_head_channels, num_head_channels),
                    #     nn.Linear(time_embed_dim, ich)
                    # ]))

                if style_modulation:
                    self.style_modules.append(nn.ModuleList([
                        nn.LayerNorm(ch*2),
                        nn.Linear(time_embed_dim, ch*2),
                        zero_module(nn.Linear(ch*2, ch*2))
                    ]))

        if not discard_final_layers:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )

        self.conv_fg = zero_module(
            conv_nd(dims, in_channels_fg, model_channels, 3, padding=1)
        ) if exists(in_channels_fg) else None
        self.conv_bg = zero_module(
            conv_nd(dims, in_channels_bg, model_channels, 3, padding=1)
        ) if exists(in_channels_bg) else None

    def forward(self, x, timesteps=None, y=None, *args, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y.to(self.dtype))

        h = self._forward(x, emb, *args, **kwargs)
        return self.out(h).to(x.dtype)

    def _forward(
            self,
            x,
            emb,
            control = None,
            context = None,
            mask = None,
            **additional_context
    ):
        hs = []
        h = x.to(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb, context, mask, **additional_context)
            hs.append(h)

        h = self.middle_block(h, emb, context, mask, **additional_context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, mask, **additional_context)
        return h


class DualCondUNetXL(UNetModel):
    def __init__(
            self,
            hint_encoder_index = (0, 3, 6, 8),
            hint_decoder_index = (),
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hint_encoder_index = hint_encoder_index
        self.hint_decoder_index = hint_decoder_index

    def _forward(self, x, emb, concat=None, control=None, context=None, mask=None, **additional_context):
        h = x.to(self.dtype)
        hs = []

        if exists(concat):
            h = torch.cat([h, concat], 1)

        control_iter = iter(control)
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb, context, mask, **additional_context)

            if idx in self.hint_encoder_index:
                h += next(control_iter)
            hs.append(h)

        h = self.middle_block(h, emb, context, mask, **additional_context)

        for idx, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, mask, **additional_context)

            if idx in self.hint_decoder_index:
                h += next(control_iter)

        return h


class ReferenceNet(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(discard_final_layers=True, *args, **kwargs)

    def forward(self, x, timesteps=None, y=None, *args, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y.to(self.dtype))
        self._forward(x, emb, *args, **kwargs)

    def _forward(self, *args, **kwargs):
        super()._forward(*args, **kwargs)
        return None