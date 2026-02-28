"""SDXL inference wrapper for ColorizeDiffusion.
InferenceWrapper handles conditional generation with reference image guidance,
sketch encoding, foreground/background separation, and attention injection.
"""

import torch
import torch.nn.functional as F

from ..modules.reference_net import hack_inference_forward
from ..models.basemodel import CustomizedColorizer, CustomizedWrapper
from ..modules.lora import LoraModules
from ..util import exists, expand_to_batch_size, instantiate_from_config, get_crop_scale, resize_and_crop



class InferenceWrapper(CustomizedWrapper, CustomizedColorizer):
    def __init__(
            self,
            scalar_embedder_config,
            img_embedder_config,
            lora_config = None,
            logits_embed = False,
            *args,
            **kwargs
    ):
        CustomizedColorizer.__init__(self, version="sdxl", *args, **kwargs)
        CustomizedWrapper.__init__(self)

        self.scalar_embedder = instantiate_from_config(scalar_embedder_config)
        self.img_embedder = instantiate_from_config(img_embedder_config)
        self.loras = LoraModules(self, **lora_config) if exists(lora_config) else None
        self.logits_embed = logits_embed

        new_model_list = {
            "scalar_embedder": self.scalar_embedder,
            "img_embedder": self.img_embedder,
            # "style_encoder": self.style_encoder,
        }
        self.switch_cond_modules += list(new_model_list.keys())
        self.model_list.update(new_model_list)

    def retrieve_attn_modules(self):
        scale_factor_levels = {"high": 0.5, "low": 0.25, "bottom": 0.25}

        from refnet.modules.transformer import BasicTransformerBlock
        from refnet.sampling import torch_dfs

        attn_modules = []
        for module in torch_dfs(self.model.diffusion_model):
            if isinstance(module, BasicTransformerBlock):
                attn_modules.append(module)

        self.attn_modules = {
            "high": [0, 1, 2, 3] + [64, 65, 66, 67, 68, 69],
            "low": [i for i in range(4, 24)] + [i for i in range(34, 64)],
            "bottom": [i for i in range(24, 34)],
            "encoder": [i for i in range(24)],
            "decoder": [i for i in range(34, len(attn_modules))]
        }
        self.attn_modules["modules"] = attn_modules

        for k in ["high", "low", "bottom"]:
            scale_factor = scale_factor_levels[k]
            for attn in self.attn_modules[k]:
                attn_modules[attn].scale_factor = scale_factor

    def adjust_reference_scale(self, scale_kwargs):
        for module in self.attn_modules["modules"]:
            module.reference_scale = scale_kwargs["scales"]["encoder"]

    def adjust_masked_attn(self, scale, mask_threshold, merge_scale):
        for layer in self.attn_layers:
            layer.mask_scale = scale
            layer.mask_threshold = mask_threshold
            layer.merge_scale = merge_scale

    def rescale_size(self, x: torch.Tensor, height, width):
        oh, ow = x.shape[2:]
        if oh < height or ow < width:
            dh, dw = height - oh, width - ow
            if dh > dw:
                iw = ow + int(dh * ow/oh)
                ih = height
            else:
                ih = oh + int(dw * oh/ow)
                iw = width
        else:
            ih, iw = oh, ow
        return torch.Tensor([ih]), torch.Tensor([iw])

    def get_learned_embedding(self, c, bg=False, mapping=False, sketch=None, *args, **kwargs):
        clip_emb = self.cond_stage_model.encode(c, "full").detach()
        wd_emb, logits = self.img_embedder.encode(c, pooled=False, return_logits=True)
        cls_emb, local_emb = clip_emb[:, :1], clip_emb[:, 1:]

        if mapping:
            _, sketch_logits = self.img_embedder.encode(-sketch, pooled=False, return_logits=True)
            sketch_logits.mean(dim=1, keepdim=True)
            logits = self.img_embedder.geometry_update(logits, sketch_logits)
        emb = self.proj(clip_emb, logits if self.logits_embed else wd_emb, bg)
        return emb, cls_emb

    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            height,
            width,
            control_scale = (1., 1., 1., 1.),
            merge_scale = 0,
            mask_scale = 1.,
            fg_scale = 1.,
            bg_scale = 1.,
            smask = None,
            rmask = None,
            mask_threshold_ref = 0.,
            mask_threshold_sketch = 0.,
            style_enhance = False,
            fg_enhance = False,
            bg_enhance = False,
            background = None,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            geometry_map = False,
            latent_inpaint = False,
            low_vram = False,
            *args,
            **kwargs
    ):
        # prepare reference embedding
        # manipulate = self.check_manipulate(target_scales)
        c = {}
        uc = [{}, {}]

        if exists(reference):
            emb, cls_emb = self.get_learned_embedding(reference, sketch=sketch, mapping=geometry_map)
        else:
            emb, cls_emb = map(lambda t: torch.zeros_like(t), self.get_learned_embedding(sketch))

        h, w, score = torch.Tensor([height]), torch.Tensor([width]), torch.Tensor([7.])
        y = torch.cat(self.scalar_embedder(torch.cat([(h*w)**0.5, score])).cuda().chunk(2), 1)

        if bg_enhance:
            assert exists(rmask) and exists(smask)

            if low_vram:
                self.low_vram_shift(["first", "cond", "img_embedder", "proj"])
            
            if latent_inpaint and exists(background):
                bgh, bgw = background.shape[2:]
                ch, cw = get_crop_scale(torch.tensor([height]), torch.tensor([width]), bgh, bgw)
                hs_bg = self.get_first_stage_encoding(resize_and_crop(background, ch, cw, height, width).to(self.first_stage_model.dtype))
                bg_emb, _ = self.get_learned_embedding(background, bg=True)
                hs_bg = expand_to_batch_size(hs_bg, bs)
                c.update({"inpaint_bg": hs_bg})
            else:
                if exists(background):
                    bg_emb, _ = self.get_learned_embedding(background, bg=True)
                else:
                    bg_emb, _ = self.get_learned_embedding(
                        torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference)),
                        True
                    )
            emb = torch.cat([emb, bg_emb], 1)

        if fg_enhance and exists(self.loras):
            self.loras.switch_lora(True, "foreground")
            if not bg_enhance:
                emb = emb.repeat(1, 2, 1)

        if fg_enhance or bg_enhance:
            # sketch mask for cross-attention
            smask = expand_to_batch_size(smask.to(self.dtype), bs)
            for d in [c] + uc:
                d.update({"mask": F.interpolate(smask, scale_factor=0.125)})
        elif exists(self.loras):
            self.loras.switch_lora(False)

        sketch = sketch.to(self.dtype)
        context = expand_to_batch_size(emb, bs).to(self.dtype)
        y = expand_to_batch_size(y, bs)
        uc_context = torch.zeros_like(context)

        control = []
        uc_control = []
        if low_vram:
            self.low_vram_shift(["control_encoder"])
        encoded_sketch = self.control_encoder(
            torch.cat([sketch, -torch.ones_like(sketch)], 0)
        )
        for idx, es in enumerate(encoded_sketch):
            es = es * control_scale[idx]
            ec, uec = es.chunk(2)
            control.append(expand_to_batch_size(ec, bs))
            uc_control.append(expand_to_batch_size(uec, bs))

        c.update({"control": control, "context": [context], "y": [y]})
        uc[0].update({"control": control, "context": [uc_context], "y": [y]})
        uc[1].update({"control": uc_control, "context": [context], "y": [y]})
        return c, uc