"""SDXL v2 inference wrapper with adapter-based foreground/background separation.
InferenceWrapperXL extends the base colorizer with ReferenceNet-based background encoding,
foreground encoder, LoRA-based background adaptation, and style modulation support.
Used with configs/inference/xlv2.yaml.
"""

from refnet.models.basemodel import CustomizedColorizer, CustomizedWrapper
from refnet.util import *
from refnet.modules.lora import LoraModules
from refnet.modules.reference_net import hack_unet_forward, hack_inference_forward
from refnet.sampling.hook import ReferenceAttentionControl


class InferenceWrapperXL(CustomizedWrapper, CustomizedColorizer):
    def __init__(
            self,
            scalar_embedder_config,
            img_embedder_config,
            fg_encoder_config = None,
            bg_encoder_config = None,
            style_encoder_config = None,
            lora_config = None,
            logits_embed = False,
            controller = False,
            *args,
            **kwargs
    ):
        CustomizedColorizer.__init__(self, version="sdxl", *args, **kwargs)
        CustomizedWrapper.__init__(self)

        self.logits_embed = logits_embed

        (
            self.scalar_embedder,
            self.img_embedder,
            self.fg_encoder,
            self.bg_encoder,
            self.style_encoder
        ) = map(
            lambda t: instantiate_from_config(t) if exists(t) else None,
            (
                scalar_embedder_config,
                img_embedder_config,
                fg_encoder_config,
                bg_encoder_config,
                style_encoder_config
            )
        )
        self.loras = LoraModules(self, **lora_config)

        if controller:
            self.controller = ReferenceAttentionControl(
                reader_module = self.model.diffusion_model,
                writer_module = self.bg_encoder,
            )
        else:
            self.controller = None

        new_model_list = {
            "scalar_embedder": self.scalar_embedder,
            "img_embedder": self.img_embedder,
        }

        hack_unet_forward(self.model.diffusion_model)
        if exists(self.fg_encoder):
            hack_inference_forward(self.fg_encoder)
            new_model_list["fg_encoder"] = self.fg_encoder
        if exists(self.bg_encoder):
            hack_inference_forward(self.bg_encoder)
            new_model_list["bg_encoder"] = self.bg_encoder

        self.switch_cond_modules += list(new_model_list.keys())
        self.model_list.update(new_model_list)


    def switch_to_fp16(self):
        super().switch_to_fp16()
        self.model.diffusion_model.map_modules.to(self.half_precision_dtype)
        self.model.diffusion_model.warp_modules.to(self.half_precision_dtype)
        self.model.diffusion_model.style_modules.to(self.half_precision_dtype)
        self.model.diffusion_model.conv_fg.to(self.half_precision_dtype)

        if exists(self.fg_encoder):
            self.fg_encoder.to(self.half_precision_dtype)
            self.fg_encoder.dtype = self.half_precision_dtype
            self.fg_encoder.time_embed.float()
        if exists(self.bg_encoder):
            self.bg_encoder.to(self.half_precision_dtype)
            self.bg_encoder.dtype = self.half_precision_dtype
            self.bg_encoder.time_embed.float()

    def switch_to_fp32(self):
        super().switch_to_fp32()
        self.model.diffusion_model.map_modules.float()
        self.model.diffusion_model.warp_modules.float()
        self.model.diffusion_model.style_modules.float()

        self.fg_encoder.float()
        self.bg_encoder.float()

        self.fg_encoder.dtype = torch.float32
        self.bg_encoder.dtype = torch.float32

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
        return torch.tensor([ih]), torch.tensor([iw])

    def rescale_background_size(self, x, height, width):
        oh, ow = x.shape[2:]
        if oh < height or ow < width:
            mind = max(height, width)
            ih = oh + mind
            iw = ow / oh * ih
        else:
            ih, iw = oh, ow
        return torch.tensor([ih]), torch.tensor([iw])

    def get_learned_embedding(self, c, bg=False, sketch=None, mapping=False, *args, **kwargs):
        clip_emb = self.cond_stage_model.encode(c, "full").detach()
        wd_emb, logits = self.img_embedder.encode(c, pooled=False, return_logits=True)
        cls_emb, local_emb = clip_emb[:, :1], clip_emb[:, 1:]

        if self.logits_embed and exists(sketch) and mapping:
            _, sketch_logits = self.img_embedder.encode(-sketch, pooled=True, return_logits=True)
            logits = self.img_embedder.geometry_update(logits, sketch_logits)

        if self.logits_embed:
            emb = self.proj(clip_emb, logits, bg)[0]
        else:
            emb = self.proj(clip_emb, wd_emb, bg)
        return emb.to(self.dtype), cls_emb.to(self.dtype)

    def prepare_conditions(
            self,
            bs,
            sketch,
            reference,
            height,
            width,
            control_scale = 1,
            mask_scale = 1,
            merge_scale = 0.,
            cond_aug = 0.,
            background = None,
            smask = None,
            rmask = None,
            mask_threshold_ref = 0.,
            mask_threshold_sketch = 0.,
            style_enhance = False,
            fg_enhance = False,
            bg_enhance = False,
            latent_inpaint = False,
            fg_disentangle_scale = 1.,
            targets = None,
            anchors = None,
            controls = None,
            target_scales = None,
            enhances = None,
            thresholds_list = None,
            low_vram = False,
            *args,
            **kwargs
    ):
        def prepare_style_modulations(y):
            z_ref = self.get_first_stage_encoding(warp_resize(reference, (height, width)))
            if exists(background) and merge_scale > 0:
                rh, rw = self.rescale_size(background, height, width)
                z_bg = self.get_first_stage_encoding(warp_resize(background, (height, width)))
                bg_emb, bg_cls_emb = self.get_learned_embedding(background)
                scalar_embed = torch.cat(
                    self.scalar_embedder(torch.cat([rh, rw, ct, cl, h, w])).chunk(6), 1
                ).to(bg_emb.device)
                bgy = torch.cat([bg_cls_emb.squeeze(1), scalar_embed], 1).to(self.dtype)

                style_modulations = self.style_encoder(
                    torch.cat([z_ref, z_bg]),
                    timesteps = torch.zeros((2,), dtype=torch.long, device=z_ref.device),
                    context = torch.cat([emb, bg_emb]),
                    y = torch.cat([y, bgy])
                )

                for idx, m in enumerate(style_modulations):
                    fg, bg = m.chunk(2)
                    m = fg * (1-merge_scale) + merge_scale * bg
                    style_modulations[idx] = expand_to_batch_size(m, bs).to(self.dtype)

            else:
                z_bg = None
                bg_emb = None
                bgy = None
                style_modulations = self.style_encoder(
                    z_ref,
                    timesteps = torch.zeros((1,), dtype=torch.long, device=z_ref.device),
                    context = emb,
                    y = y,
                )
                style_modulations = [expand_to_batch_size(m, bs).to(self.dtype) for m in style_modulations]

            return style_modulations, z_bg, bg_emb, bgy


        def prepare_background_latents(z_bg, bg_emb, bgy):
            bgh, bgw = background.shape[2:] if exists(background) else reference.shape[2:]
            ch, cw = get_crop_scale(h, w, bgh, bgw)

            if low_vram:
                self.low_vram_shift(["first", "cond", "img_embedder"])
            if latent_inpaint and exists(background):
                hs_bg = self.get_first_stage_encoding(resize_and_crop(background, ch, cw, height, width))
                bg_emb, cls_emb = self.get_learned_embedding(background)

            else:
                if not exists(z_bg):
                    bgy = torch.cat(
                        self.scalar_embedder(torch.tensor([ct, cl, ch, cw])).chunk(4), 1
                    ).to(self.dtype).cuda()

                    if exists(background):
                        z_bg = self.get_first_stage_encoding(warp_resize(background, (height, width)))
                        bg_emb, cls_emb = self.get_learned_embedding(background)
                    else:
                        xbg = torch.where(rmask < mask_threshold_ref, reference, torch.ones_like(reference))
                        z_bg = self.get_first_stage_encoding(warp_resize(xbg, (height, width)))
                        bg_emb, cls_emb = self.get_learned_embedding(xbg)

                if low_vram:
                    self.low_vram_shift(["bg_encoder"])
                hs_bg = self.bg_encoder(
                    x = torch.cat([
                        z_bg,
                        F.interpolate(warp_resize(smask, (height, width)), scale_factor=0.125),
                        F.interpolate(warp_resize(rmask, (height, width)), scale_factor=0.125)
                    ], 1),
                    timesteps = torch.zeros((1,), dtype=torch.long, device=z_bg.device),
                    y = bgy.to(self.dtype),
                )
            return hs_bg, bg_emb

        self.loras.recover_lora()
        c = {}
        uc = [{}, {}]
        self.loras.switch_lora(False)

        if exists(reference):
            emb, cls_emb = self.get_learned_embedding(reference, sketch=sketch)
        else:
            emb, cls_emb = map(lambda t: torch.zeros_like(t), self.get_learned_embedding(sketch))

        ct, cl = torch.Tensor([0]), torch.Tensor([0])
        h, w, score = torch.Tensor([height]), torch.Tensor([width]), torch.Tensor([7.])
        y = torch.cat(self.scalar_embedder(torch.cat([(h * w) ** 0.5, score])).cuda().chunk(2), 1)

        z_bg, bg_emb, bgy = None, None, None

        # Style modulation injection
        if style_enhance:
            style_modulations, z_bg, bg_emb, bgy = prepare_style_modulations(y)
            for d in [c] + uc:
                d.update({"style_modulations": style_modulations})

        # Foreground enhancement with dedicated encoder
        if fg_enhance:
            assert exists(smask) and exists(rmask)
            self.loras.switch_lora(True, "foreground")
            if low_vram:
                self.low_vram_shift(["first"])
            z_fg = self.get_first_stage_encoding(warp_resize(
                torch.where(rmask >= mask_threshold_ref, reference, torch.ones_like(reference)),
                (height, width)
            )) * fg_disentangle_scale
            self.loras.adjust_lora_scales(fg_disentangle_scale, "foreground")
            if low_vram:
                self.low_vram_shift(["fg_encoder"])
            hs_fg = self.fg_encoder(
                z_fg,
                timesteps = torch.zeros((1,), dtype=torch.long, device=z_fg.device),
            )
            hs_fg = expand_to_batch_size(hs_fg, bs)
            for d in [c] + uc:
                d.update({
                    "hs_fg": hs_fg,
                    "inject_mask": expand_to_batch_size(smask, bs),
                })

        # Background enhancement with ReferenceNet encoder
        if bg_enhance:
            assert exists(rmask) and exists(smask)
            hs_bg, bg_emb = prepare_background_latents(z_bg, bg_emb, default(bgy, y))
            self.loras.switch_lora(True, "background")
            if latent_inpaint and exists(background):
                hs_bg = expand_to_batch_size(hs_bg, bs)
                c.update({"inpaint_bg": hs_bg})
            elif exists(self.controller):
                self.controller.update()
            else:
                hs_bg = expand_to_batch_size(hs_bg, bs)
                for d in [c] + uc:
                    d.update({"hs_bg": hs_bg})

        elif exists(self.controller):
            self.controller.clean()

        if fg_enhance or bg_enhance:
            # Activate mask-guided split cross-attention
            emb = torch.cat([emb, default(bg_emb, emb)], 1)
            smask = expand_to_batch_size(smask.to(self.dtype), bs)
            for d in [c] + uc:
                d.update({"mask": F.interpolate(smask, scale_factor=0.125), "threshold": mask_threshold_sketch})

        sketch = sketch.to(self.dtype)
        context = expand_to_batch_size(emb, bs).to(self.dtype)
        y = expand_to_batch_size(y, bs).float()
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

        self.loras.merge_lora()
        c.update({"control": control, "context": [context], "y": [y]})
        uc[0].update({"control": control, "context": [uc_context], "y": [y]})
        uc[1].update({"control": uc_control, "context": [context], "y": [y]})
        return c, uc
