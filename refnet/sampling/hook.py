import torch
import torch.nn as nn

from refnet.modules.transformer import BasicTransformerBlock, SelfInjectedTransformerBlock
from refnet.util import checkpoint_wrapper

"""
    This implementation refers to Multi-ControlNet, thanks for the authors
    Paper: Adding Conditional Control to Text-to-Image Diffusion Models
    Link: https://github.com/Mikubill/sd-webui-controlnet
"""

def exists(v):
    return v is not None

def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

class AutoMachine():
    Read = "read"
    Write = "write"


"""
    This class controls the attentions of reference unet and denoising unet
"""
class ReferenceAttentionControl:
    writer_modules = []
    reader_modules = []
    def __init__(
            self,
            reader_module,
            writer_module,
            time_embed_ch = 0,
            only_decoder = True,
            *args,
            **kwargs
    ):
        self.time_embed_ch = time_embed_ch
        self.trainable_layers = []
        self.only_decoder = only_decoder
        self.hooked = False

        self.register("read", reader_module)
        self.register("write", writer_module)

        if time_embed_ch > 0:
            self.insert_time_emb_proj(reader_module)

    def insert_time_emb_proj(self, unet):
        for module in torch_dfs(unet.output_blocks if self.only_decoder else unet):
            if isinstance(module, BasicTransformerBlock):
                module.time_proj = nn.Linear(self.time_embed_ch, module.dim)
                self.trainable_layers.append(module.time_proj)

    def register(self, mode, unet):
        @checkpoint_wrapper
        def transformer_forward_write(self, x, context=None, mask=None, emb=None, **kwargs):
            x_in = self.norm1(x)
            x = self.attn1(x_in) + x

            if not self.disable_cross_attn:
                x = self.attn2(self.norm2(x), context) + x
            x = self.ff(self.norm3(x)) + x

            self.bank = x_in
            return x

        @checkpoint_wrapper
        def transformer_forward_read(self, x, context=None, mask=None, emb=None, **kwargs):
            if exists(self.bank):
                bank = self.bank
                if bank.shape[0] != x.shape[0]:
                    bank = bank.repeat(x.shape[0], 1, 1)
                if hasattr(self, "time_proj"):
                    bank = bank + self.time_proj(emb).unsqueeze(1)
                x_in = self.norm1(x)

                x = self.attn1(
                    x = x_in,
                    context = torch.cat([x_in, bank], 1),
                    mask = mask,
                    scale_factor = self.scale_factor,
                    **kwargs
                ) + x

                x = self.attn2(
                    x = self.norm2(x),
                    context = context,
                    mask = mask,
                    scale = self.reference_scale,
                    scale_factor = self.scale_factor
                ) + x

                x = self.ff(self.norm3(x)) + x
            else:
                x = self.original_forward(x, context, mask, emb)
            return x

        assert mode in ["write", "read"]

        if mode == "read":
            self.hooked = True
        for module in torch_dfs(unet.output_blocks if self.only_decoder else unet):
            if isinstance(module, BasicTransformerBlock):
                if mode == "write":
                    module.original_forward = module.forward
                    module.forward = transformer_forward_write.__get__(module, BasicTransformerBlock)
                    self.writer_modules.append(module)
                else:
                    if not isinstance(module, SelfInjectedTransformerBlock):
                        print(f"Hooking transformer block {module.__class__.__name__} for read mode")
                        module.original_forward = module.forward
                        module.forward = transformer_forward_read.__get__(module, BasicTransformerBlock)
                    self.reader_modules.append(module)

    def update(self):
        for idx in range(len(self.writer_modules)):
            self.reader_modules[idx].bank = self.writer_modules[idx].bank
            
    def restore(self):
        for idx in range(len(self.writer_modules)):
            self.writer_modules[idx].forward = self.writer_modules[idx].original_forward
            self.reader_modules[idx].forward = self.reader_modules[idx].original_forward
            self.reader_modules[idx].bank = None
        self.hooked = False

    def clean(self):
        for idx in range(len(self.reader_modules)):
            self.reader_modules[idx].bank = None
        for idx in range(len(self.writer_modules)):
            self.writer_modules[idx].bank = None
        self.hooked = False

    def reader_restore(self):
        for idx in range(len(self.reader_modules)):
            self.reader_modules[idx].forward = self.reader_modules[idx].original_forward
            self.reader_modules[idx].bank = None
        self.hooked = False

    def get_trainable_layers(self):
        return self.trainable_layers


"""
    This class is for self-injection inside the denoising unet 
"""
class UnetHook:
    def __init__(self):
        super().__init__()
        self.attention_auto_machine = AutoMachine.Read

    def enhance_reference(
            self,
            model,
            ldm,
            bs,
            s,
            r,
            style_cfg=0.5,
            control_cfg=0,
            gr_indice=None,
            injection=False,
            start_step=0,
    ):
        def forward(self, x, t, control, context, **kwargs):
            if 1 - t[0] / (ldm.num_timesteps - 1) >= outer.start_step:
                # Write
                outer.attention_auto_machine = AutoMachine.Write

                rx = ldm.add_noise(outer.r.cpu(), torch.round(t.float()).long().cpu()).cuda().to(x.dtype)
                self.original_forward(rx, t, control=outer.s, context=context, **kwargs)

                # Read
                outer.attention_auto_machine = AutoMachine.Read
            return self.original_forward(x, t, control=control, context=context, **kwargs)

        def hacked_basic_transformer_inner_forward(self, x, context=None, mask=None, emb=None, **kwargs):
            x_norm1 = self.norm1(x)
            self_attn1 = None
            if self.disable_self_attn:
                # Do not use self-attention
                self_attn1 = self.attn1(x_norm1, context=context, **kwargs)

            else:
                # Use self-attention
                self_attention_context = x_norm1
                if outer.attention_auto_machine == AutoMachine.Write:
                    self.bank.append(self_attention_context.detach().clone())
                    self.style_cfgs.append(outer.current_style_fidelity)
                if outer.attention_auto_machine == AutoMachine.Read:
                    if len(self.bank) > 0:
                        style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                        self_attn1_uc = self.attn1(
                            x_norm1,
                            context=torch.cat([self_attention_context] + self.bank, dim=1),
                            **kwargs
                        )
                        self_attn1_c = self_attn1_uc.clone()
                        if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                            self_attn1_c[outer.current_uc_indices] = self.attn1(
                                x_norm1[outer.current_uc_indices],
                                context=self_attention_context[outer.current_uc_indices],
                                **kwargs
                            )
                        self_attn1 = style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                    self.bank = []
                    self.style_cfgs = []
                if self_attn1 is None:
                    self_attn1 = self.attn1(x_norm1, context=self_attention_context)

            x = self_attn1.to(x.dtype) + x
            x = self.attn2(self.norm2(x), context, mask, self.reference_scale, self.scale_factor, **kwargs) + x
            x = self.ff(self.norm3(x)) + x
            return x

        self.s = [s.repeat(bs, 1, 1, 1) * control_cfg for s in ldm.control_encoder(s)]
        self.r = r
        self.injection = injection
        self.start_step = start_step
        self.current_uc_indices = gr_indice
        self.current_style_fidelity = style_cfg

        outer = self
        model = model.diffusion_model
        model.original_forward = model.forward
        # TODO: change the class name to target
        model.forward = forward.__get__(model, model.__class__)
        all_modules = torch_dfs(model)

        for module in all_modules:
            if isinstance(module, BasicTransformerBlock):
                module._unet_hook_original_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.style_cfgs = []


    def restore(self, model):
        model = model.diffusion_model
        if hasattr(model, "original_forward"):
            model.forward = model.original_forward
            del model.original_forward

        all_modules = torch_dfs(model)
        for module in all_modules:
            if isinstance(module, BasicTransformerBlock):
                if hasattr(module, "_unet_hook_original_forward"):
                    module.forward = module._unet_hook_original_forward
                    del module._unet_hook_original_forward
                if hasattr(module, "bank"):
                    module.bank = None
                if hasattr(module, "style_cfgs"):
                    del module.style_cfgs