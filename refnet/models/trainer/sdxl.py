"""SDXL training modules for ColorizeDiffusion.
- ColorizerXLTrainer: Adapter-based training with ReferenceNet (sdxl-adapter.yaml).
- GramTrainer: Base model training with gram-style cross-attention guidance (sdxl-base.yaml).
"""

from refnet.util import *
from refnet.modules.lora import LoraModules
from refnet.modules.reference_net import hack_unet_forward
from refnet.models.trainer.trainer import ColorizerTrainer
from refnet.sampling.hook import ReferenceAttentionControl
from typing import Union


class ColorizerXLTrainer(ColorizerTrainer):
    def __init__(
            self,
            score_key = None,
            smask_required = False,
            img_embedder_config = None,
            txt_embedder_config = None,
            scalar_embedder_config = None,
            pooled_wd_emb = False,
            return_logits = False,
            ucg_wd_rate = 0.,
            ucg_clip_rate = 0.,
            ucg_mask = 0.,
            stage = 0,
            thresh_max = 0.99,
            thresh_min = 0.01,
            merge_offset = 4,
            inpaint_p = 0.,
            masked_p = 0.,
            fgbg_p = 0.,
            preserve_lambda = 1.,
            lora_config = None,
            concat_mask = False,
            bg_encoder_config = None,
            fg_encoder_config = None,
            *args,
            **kwargs
    ):
        super().__init__(version="sdxl", *args, **kwargs)
        self.img_embedder, self.txt_embedder, self.scalar_embedder = map(
            lambda t: instantiate_from_config(t).eval().requires_grad_(False) if exists(t) else None,
            (img_embedder_config, txt_embedder_config, scalar_embedder_config)
        )
        self.score_key = score_key
        self.smask_required = smask_required
        self.ucg_wd_rate = ucg_wd_rate
        self.ucg_clip_rate = ucg_clip_rate
        self.ucg_mask = ucg_mask

        self.pooled_wd_emb = pooled_wd_emb
        self.return_logits = return_logits
        self.thresh_min = thresh_min
        self.thresh_rate = thresh_max - thresh_min
        self.merge_offset = merge_offset
        self.masked_p = masked_p
        self.inpaint_p = inpaint_p
        self.fgbg_p = fgbg_p
        self.preserve_lambda = preserve_lambda
        self.concat_mask = concat_mask

        self.stage = stage
        self.setup_training(lora_config, bg_encoder_config, fg_encoder_config)
        self.setup_training_params()


    def setup_training(self, lora_config, bg_encoder_config, fg_encoder_config):
        self.loras = LoraModules(self, **lora_config) if exists(lora_config) else None

        if self.stage > 0:
            hack_unet_forward(self.model.diffusion_model)

        if exists(fg_encoder_config):
            self.fg_encoder = instantiate_from_config(fg_encoder_config)
        else:
            self.fg_encoder = None

        if exists(bg_encoder_config):
            self.bg_encoder = instantiate_from_config(bg_encoder_config)
            self.controller = ReferenceAttentionControl(
                # time_embed_ch = self.model.diffusion_model.model_channels * 4,
                reader_module = self.model.diffusion_model,
                writer_module = self.bg_encoder,
                # only_decoder = True
            )

        else:
            self.bg_encoder = None
            self.controller = None


    def setup_training_params(self):
        optim_modules: list[Union[torch.nn.Module|torch.nn.ModuleList]] = []
        frozen_modules: list[Union[torch.nn.Module|torch.nn.ModuleList]] = []

        if self.stage == 0:
            optim_modules += [self.model.diffusion_model, self.control_encoder, self.proj]
        else:
            frozen_modules += [self.model.diffusion_model, self.control_encoder, self.proj]

            if self.stage == 1:
                if exists(self.loras):
                    optim_modules += self.loras.get_trainable_layers("foreground")
                if exists(self.fg_encoder):
                    optim_modules += [self.fg_encoder, self.model.diffusion_model.map_modules]
                if exists(self.model.diffusion_model.conv_fg):
                    optim_modules += [self.model.diffusion_model.conv_fg]

            elif self.stage == 2:
                assert exists(self.bg_encoder) and exists(self.controller)
                optim_modules += [self.bg_encoder]
                if exists(self.loras):
                    optim_modules += self.loras.get_trainable_layers("background")
                if exists(self.fg_encoder):
                    frozen_modules += [self.fg_encoder]

        for m in frozen_modules:
            m.requires_grad_(False).eval()

        for m in optim_modules:
            m.requires_grad_(True).train()

        if hasattr(self.proj, "mlp"):
            self.proj.mlp.requires_grad_(False).eval()


    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        if exists(self.fg_encoder):
            self.fg_encoder.dtype = self.dtype
        if exists(self.bg_encoder):
            self.bg_encoder.dtype = self.dtype

    def get_input(
            self,
            batch,
            bs = None,
            return_inputs = False,
            *args,
            **kwargs
    ):
        with torch.no_grad():
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]
            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key]
            xmr = batch.get('r' + self.mask_key, None)
            xms = batch.get('s' + self.mask_key, None)
            score = batch.get(self.score_key, None)

            # calculate size scalar embedding
            size = batch["size"]
            ih, iw, load_size = size[:, 0], size[:, 1], size[:, -1]
            h, w = torch.ones_like(ih) * x.shape[2], torch.ones_like(iw) * x.shape[3]
            resize_scale = torch.min(size[:,:1], 1).values / load_size
            oh, ow = map(lambda t: (t * resize_scale).to(torch.int), (h, w))
            scalar_inputs = (oh * ow) ** 0.5

            if exists(score):
                scalar_inputs = torch.cat([scalar_inputs, score])
                y = torch.cat(self.scalar_embedder(scalar_inputs).chunk(2), 1)
            else:
                y = self.scalar_embedder(scalar_inputs)
            y = y.to(self.dtype)

            x, xc, xs, xms, xmr = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype) if exists(t) else None,
                (x, xc, xs, xms, xmr)
            )
            bs = x.shape[0]
            thresh_s, thresh_r = (
                    self.thresh_min + self.thresh_rate * torch.rand((bs * 2), device=x.device)
            ).chunk(2)

            # inpainting training
            if self.inpaint_p > 0:
                inpaint_idx = append_dims(torch.rand(bs, device=xc.device) < self.inpaint_p, x.ndim)
                inpaint_xc = torch.where(
                    torch.roll(xmr, self.merge_offset, 0) < append_dims(thresh_r, xc.ndim),
                    xc,
                    torch.ones_like(xc)
                )
                xc = torch.where(inpaint_idx, inpaint_xc, xc)

            # TODO: check if need foreground mask training
            if self.masked_p > 0:
                masked_idx = append_dims(torch.rand(bs, device=xc.device) < self.masked_p, x.ndim)
                masked_xc = torch.where(xmr > thresh_r, torch.ones_like(xc), xc)
                xc = torch.where(masked_idx, masked_xc, xc)

            if self.stage > 0:
                fgbg = True

                # background bleaching
                x, xs, xc_bg = background_bleaching(
                    x, xs, xc, xms, xmr, thresh_s, thresh_r, self.p_white_bg, self.dtype
                )
                bg_wd_emb, _, bg_clip_emb = self.get_learned_embedding(
                    xc_bg,
                    pooled = self.pooled_wd_emb,
                    output_type = self.token_type
                )

            else:
                fgbg = False
                bg_wd_emb, bg_clip_emb = None, None

            # if self.control_drop > 0:
            #     xs = torch.where(zero_drop(xs, self.control_drop) > 0, xs, -torch.ones_like(xs))

            wd_emb, wd_logits, clip_emb = self.get_learned_embedding(
                xc,
                pooled = self.pooled_wd_emb,
                output_type = self.token_type,
                return_logits = self.return_logits
            )
            z = self.get_first_stage_encoding(x)

        controls = self.control_encoder(
            torch.cat([xs, zero_drop(xms, self.ucg_mask) * xms], 1) if self.concat_mask else xs
        )
        controls = [control * zero_drop(control, dp) for control, dp in zip(controls, self.control_drop)]

        cond = {
            "control": controls,
            "y": [y],
            # "smask": xms,
        }
        if self.smask_required:
            cond.update({"smask": xms})

        c = self.proj(clip_emb, wd_logits if self.return_logits else wd_emb)
        if fgbg or self.stage == 3:
            # bgc = self.proj(bg_clip_emb, bg_wd_emb, fgbg)
            bgc = self.proj(bg_clip_emb, bg_wd_emb)
            c = torch.cat([c, bgc], 1)

            if exists(self.fg_encoder):
                fgs = self.fg_encoder(
                    x = zc,
                    timesteps = torch.zeros((bs,), dtype=torch.long, device=z.device),
                    # context = c,
                )
                cond.update(dict(hs_fg=fgs))

            if exists(self.bg_encoder):
                with torch.no_grad():
                    z_bg = self.get_first_stage_encoding(warp_resize(xc_bg, (x.shape[2], x.shape[3])))
                    crops = batch["crop"]
                    ct, cl, rh, rw = crops[:, 0], crops[:, 1], crops[:, 2], crops[:, 3]
                    size_embed = torch.cat(self.scalar_embedder(torch.cat([ct, cl, rh, rw])).chunk(4), 1)

                self.bg_encoder(
                    x = torch.cat([
                        z_bg,
                        F.interpolate(xms, scale_factor=0.125),
                        warp_resize(xmr, (z_bg.shape[2:])),
                    ], 1),
                    timesteps = torch.zeros((bs,), dtype=torch.long, device=z_bg.device),
                    # context=bgc,
                    y = size_embed
                )
                self.controller.update()

            self.adjust_mask_threshold(append_dims(thresh_s, c.ndim))
            cond.update({"mask": F.interpolate(xms, scale_factor=0.125, mode="bicubic")})

        else:
            if exists(self.controller):
                self.controller.clean()

        cond.update({"context": [c]})
        out = [z, cond]
        if return_inputs:
            out.extend([x, xc, xs, xms, xmr])
        return out


class ForegroundTrainer(ColorizerXLTrainer):
    # def __init__(self, reference_num, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.reference_num = reference_num

    # def ref_shuffle(self, x, offset):
        # x = rearrange(x, "(b r) ... -> b r ...", r=self.reference_num)
        # x = torch.roll(x, offset, dims=1)
        # x = rearrange(x, "b r ... -> (b r) ...")
        # return x

    # def p_losses(self, x_start, cond, t = None):
        # noise = torch.randn_like(x_start)
        # t = t if exists(t) else\
            # torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # cond.update({"noise": noise})
        # base_loss = super().p_losses(x_start, cond, t)[0]
        # ref_loss = 0
        # for i in range(self.reference_num - 1):
            # context = self.ref_shuffle(cond["context"][0], 1)

            # z_fg =  self.ref_shuffle(cond["z_fg"], 1)
            # cond.update({"context": [context], "z_fg": z_fg})

            # hs_fg = cond["hs_fg"]
            # hs_fg = [self.ref_shuffle(hs, 1) for hs in hs_fg]
            # cond.update({"context": [context], "hs_fg": hs_fg})
            # ref_loss += super().p_losses(x_start, cond, t)[0]
        # loss = base_loss * self.preserve_lambda + ref_loss / (self.reference_num - 1)
        # return loss, {"loss": loss}


    def get_input(
            self,
            batch,
            bs = None,
            return_inputs = False,
            *args,
            **kwargs
    ):
        with torch.no_grad():
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]
            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key]
            xmr = batch['r' + self.mask_key]
            xms = batch['s' + self.mask_key]
            score = batch.get(self.score_key, None)

            x, xc, xs, xms, xmr = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype) if exists(t) else None,
                (x, xc, xs, xms, xmr)
            )
            bs = x.shape[0]

            # calculate size scalar embedding
            size = batch["size"]
            ih, iw, load_size = size[:, 0], size[:, 1], size[:, -1]
            h, w = torch.ones_like(ih) * x.shape[2], torch.ones_like(iw) * x.shape[3]
            resize_scale = torch.min(size[:,:1], 1).values / load_size
            oh, ow = map(lambda t: (t * resize_scale).to(torch.int), (h, w))
            scalar_inputs = (oh * ow) ** 0.5

            if exists(score):
                scalar_inputs = torch.cat([scalar_inputs, score])
                y = torch.cat(self.scalar_embedder(scalar_inputs).chunk(2), 1)
            else:
                y = self.scalar_embedder(scalar_inputs)
            y = y.to(self.dtype)

            thresh = self.thresh_min + self.thresh_rate * torch.rand((bs), device=x.device)
            wd_emb, _, clip_emb = self.get_learned_embedding(
                xc,
                pooled = self.pooled_wd_emb,
                output_type = self.token_type
            )
            c = self.proj(clip_emb, wd_emb).to(self.dtype)
            z = self.get_first_stage_encoding(x)

            self.adjust_mask_threshold(append_dims(thresh, c.ndim))
            with torch.no_grad():
                zc = self.get_first_stage_encoding(warp_resize(xc, (x.shape[2], x.shape[3])))
            cond = {
                # "context": [c],
                "context": [c.repeat(1, 2, 1)],
                "control": self.control_encoder(xs),
                "y": [y],
                "inject_mask": xms,
                # "z_fg": zc,

                # used for split-cross attention
                "mask": F.interpolate(xms, scale_factor=0.125),
                "threshold": append_dims(thresh, z.ndim),
            }

        if exists(self.fg_encoder):
            fgs = self.fg_encoder(
                x = zc,
                timesteps = torch.zeros((bs,), dtype=torch.long, device=z.device),
                # context = c,
            )
            cond.update(dict(hs_fg=fgs))
        out = [z, cond]
        if return_inputs:
            out.extend([x, xc, xs, xms, xmr])
        return out

    @torch.inference_mode()
    def log_images(
            self,
            batch,
            N = 4,
            sampler = "dpm_vp",
            step = 20,
            unconditional_guidance_scale = 9.0,
            return_inputs = False,
            **kwargs
    ):
        """
            This function is used for batch processing.
            Used with image logger.
        """

        out = self.get_input(
            batch,
            bs = N,
            return_inputs=return_inputs,
            **kwargs
        )

        log = dict()
        if return_inputs:
            z, c, x, xc, xs, xms, xmr = out
            log["inputs"] = x
            log["control"] = xs
            log["conditioning"] = self.ref_shuffle(xc, 2)
            log["reconstruction"] = self.decode_first_stage(z.to(self.first_stage_model.dtype))

            if exists(xms):
                log["smask"] = (xms - 0.5) / 0.5
            if exists(xmr):
                log["rmask"] = (xmr - 0.5) / 0.5

        else:
            z, c = out

        crossattn = self.ref_shuffle(c["context"][0], 2)
        # y = c["y"][0].chunk(2)[1]
        # z_fg = self.ref_shuffle(c["z_fg"], 2)
        fgs = c["hs_fg"]
        fgs = [self.ref_shuffle(fg, 2) for fg in fgs]
        c.update({
            "context": [crossattn],
            "hs_fg": fgs
            # "y": [y],
            # "z_fg": z_fg
        })

        B, _, H, W = z.shape
        uc_full = c.copy()
        if unconditional_guidance_scale > 1.:
            uc_cross = torch.zeros_like(crossattn)
            uc_full.update({"context": [uc_cross]})
        else:
            uc_full = None

        samples = self.sample(
            cond = c,
            bs = B,
            shape = (self.channels, H, W),
            step = step,
            sampler = sampler,
            uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            device = z.device,
        )
        x_samples = self.decode_first_stage(samples.to(self.first_stage_model.dtype))
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples
        return log

class BackgroundTrainer(ColorizerXLTrainer):
    def apply_model(self, x_noisy, t, cond):
        self.bg_encoder(
            torch.cat([x_noisy, cond.pop("z_bg"), cond.get("mask"), cond.pop("rmask")], 1),
            timesteps = t,
            **cond
        )
        self.controller.update()
        return super().apply_model(x_noisy, t, cond)

from refnet.gram_hook import GramHooker, GramLoss

class GramTrainer(ColorizerXLTrainer):
    def __init__(
            self,
            gram_config,
            gram_start_epoch = None,
            gram_start_step = None,
            gram_clip_proj_grad = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert exists(gram_start_epoch) or exists(gram_start_step), 'You must provide a start epoch or a start step'

        self.gram_hooker = GramHooker(**gram_config.get("hooker", {}))
        self.gram_loss = GramLoss(**gram_config.get("loss", {}))
        self.gram_activated = False
        self.gram_start_epoch = gram_start_epoch
        self.gram_start_step = gram_start_step
        self.gram_clip_proj_grad = gram_clip_proj_grad

    def p_losses(self, x_start, cond, t=None):
        noise = cond.get("noise", torch.randn_like(x_start))
        cond["noise"] = noise
        t = default(t, torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long())
        self.gram_hooker.clean()
        smask = cond.pop("smask")
        diffusion_loss = super().p_losses(x_start, cond, t)[0]

        loss_dict = {"diffusion_loss": diffusion_loss}
        if self.gram_activated:
            # shuffle reference
            # self.gram_hooker_r.clean()
            if self.gram_clip_proj_grad:
                cond["context"] = [torch.roll(cond["context"][0], 1, dims=0).detach()]
            else:
                cond["context"] = [torch.roll(cond["context"][0], 1, dims=0)]
            x_noisy = self.add_noise(x_start=x_start, t=t, noise=noise).to(self.dtype)
            self.apply_model(x_noisy, t, cond)
            # with torch.no_grad():
                # self.ref_model(x_noisy, t, **cond)
            gram_loss = 0.
            gram_matrix_size = len(self.gram_hooker.get_hidden_grams()) // 2
            anchor_gram_matrices = self.gram_hooker.get_hidden_grams()[:gram_matrix_size]
            gram_matrices = self.gram_hooker.get_hidden_grams()[gram_matrix_size:]

            gram_loss = self.gram_loss(gram_matrices, anchor_gram_matrices, smask, mask_threshold=self.thresh_min)
            total_loss = diffusion_loss + gram_loss
            loss_dict["gram_loss"] = gram_loss
            loss_dict["total_loss"] = total_loss
        else:
            total_loss = diffusion_loss
        return total_loss, loss_dict


    def hook_trigger(self, epoch):
        if exists(self.gram_start_epoch) and epoch >= self.gram_start_epoch:
            self.gram_activated = True
            if not self.gram_hooker.registered:
                self.gram_hooker.register(self.model.diffusion_model)

    def on_train_epoch_start(self, epoch, *args, **kwargs):
        super().on_train_epoch_start(*args, **kwargs)
        self.hook_trigger(epoch)

    def on_train_batch_end(self, global_step, *args, **kwargs):
        if exists(self.gram_start_step) and global_step >= self.gram_start_step:
            self.gram_activated = True
            if not self.gram_hooker.registered:
                self.gram_hooker.register(self.model.diffusion_model)


from ckpt_util import load_weights

class GramReferenceTrainer(ColorizerXLTrainer):
    def __init__(
            self,
            anchor_ckpt_path,
            proj_config,
            unet_config,
            gram_weight = 1.,
            gram_type = "ff",
            gram_ids = [23, 63],
            *args,
            **kwargs
    ):
        super().__init__(proj_config=proj_config, unet_config=unet_config, *args, **kwargs)
        self.anchor_hooker = GramHooker(gram_ids=gram_ids, gram_type=gram_type)
        self.target_hooker = GramHooker(gram_ids=gram_ids, gram_type=gram_type)

        self.gram_weight = 0.
        self.activate_gram_weight = gram_weight

        self.anchor_proj = instantiate_from_config(proj_config)
        self.anchor_model = instantiate_from_config(unet_config)
        self.anchor_proj.eval().requires_grad_(False)
        self.anchor_model.eval().requires_grad_(False)
        
        sd = load_weights(anchor_ckpt_path)
        self.anchor_proj.load_state_dict(sd, strict=False)
        self.anchor_model.load_state_dict(sd, strict=False)

        self.anchor_hooker.register(self.anchor_model)
        self.target_hooker.register(self.model.diffusion_model)

    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        self.anchor_model.dtype = self.dtype

    def p_losses(self, x_start, cond, t=None):
        noise = cond.get("noise", torch.randn_like(x_start))
        wd_emb, clip_emb = cond.pop("reference")

        cond["noise"] = noise
        t = default(t, torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long())
        diffusion_loss = super().p_losses(x_start, cond, t)[0]
        loss_dict = {"diffusion_loss": diffusion_loss}

        self.anchor_hooker.clean()
        self.target_hooker.clean()
        
        # target branch
        cond["context"] = [torch.roll(cond["context"][0], 1, dims=0)]
        x_noisy = self.add_noise(x_start=x_start, t=t, noise=noise).to(self.dtype)
        self.apply_model(x_noisy, t, cond)
        
        # anchor branch
        with torch.no_grad():
            cond["context"] = [self.anchor_proj(torch.roll(clip_emb, 1, dims=0), torch.roll(wd_emb, 1, dims=0))]
            self.anchor_model(x_noisy, t, **cond)

        gram_loss = 0.
        for anchor_feat, feat in zip(
            self.anchor_hooker.get_hidden_grams(), 
            self.target_hooker.get_hidden_grams()
        ):
            gram_loss += F.mse_loss(
                feat.float(), anchor_feat.to(feat.device).float(), reduction="none"
            ).mean()

        total_loss = diffusion_loss + gram_loss * self.gram_weight
        loss_dict["gram_loss"] = gram_loss
        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict

    def get_input(
            self,
            batch,
            bs = None,
            return_inputs = False,
            *args,
            **kwargs
    ):
        with torch.no_grad():
            if exists(bs):
                for key in batch.keys():
                    batch[key] = batch[key][:bs]
            x = batch[self.first_stage_key]
            xc = batch[self.cond_stage_key]
            xs = batch[self.control_key]
            xmr = batch.get('r' + self.mask_key, None)
            xms = batch.get('s' + self.mask_key, None)
            score = batch.get(self.score_key, None)

            # calculate size scalar embedding
            size = batch["size"]
            ih, iw, load_size = size[:, 0], size[:, 1], size[:, -1]
            h, w = torch.ones_like(ih) * x.shape[2], torch.ones_like(iw) * x.shape[3]
            resize_scale = torch.min(size[:,:1], 1).values / load_size
            oh, ow = map(lambda t: (t * resize_scale).to(torch.int), (h, w))
            scalar_inputs = (oh * ow) ** 0.5

            if exists(score):
                scalar_inputs = torch.cat([scalar_inputs, score])
                y = torch.cat(self.scalar_embedder(scalar_inputs).chunk(2), 1)
            else:
                y = self.scalar_embedder(scalar_inputs)
            y = y.to(self.dtype)

            x, xc, xs, xms, xmr = map(
                lambda t: t.to(memory_format=torch.contiguous_format).to(self.dtype) if exists(t) else None,
                (x, xc, xs, xms, xmr)
            )
            bs = x.shape[0]
            thresh_s, thresh_r = (
                    self.thresh_min + self.thresh_rate * torch.rand((bs * 2), device=x.device)
            ).chunk(2)

            # inpainting training
            if self.inpaint_p > 0:
                inpaint_idx = append_dims(torch.rand(bs, device=xc.device) < self.inpaint_p, x.ndim)
                inpaint_xc = torch.where(
                    torch.roll(xmr, self.merge_offset, 0) < append_dims(thresh_r, xc.ndim),
                    xc,
                    torch.ones_like(xc)
                )
                xc = torch.where(inpaint_idx, inpaint_xc, xc)

            # TODO: check if need foreground mask training
            if self.masked_p > 0:
                masked_idx = append_dims(torch.rand(bs, device=xc.device) < self.masked_p, x.ndim)
                masked_xc = torch.where(xmr > thresh_r, torch.ones_like(xc), xc)
                xc = torch.where(masked_idx, masked_xc, xc)

            if self.stage > 0:
                fgbg = True

                # background bleaching
                x, xs, xc_bg = background_bleaching(
                    x, xs, xc, xms, xmr, thresh_s, thresh_r, self.p_white_bg, self.dtype
                )
                bg_wd_emb, _, bg_clip_emb = self.get_learned_embedding(
                    xc_bg,
                    pooled = self.pooled_wd_emb,
                    output_type = self.token_type
                )

            else:
                fgbg = False
                bg_wd_emb, bg_clip_emb = None, None

            # if self.control_drop > 0:
            #     xs = torch.where(zero_drop(xs, self.control_drop) > 0, xs, -torch.ones_like(xs))

            wd_emb, wd_logits, clip_emb = self.get_learned_embedding(
                xc,
                pooled = self.pooled_wd_emb,
                output_type = self.token_type,
                return_logits = self.return_logits
            )
            z = self.get_first_stage_encoding(x)

        controls = self.control_encoder(
            torch.cat([xs, zero_drop(xms, self.ucg_mask) * xms], 1) if self.concat_mask else xs
        )

        cond = {
            "control": controls,
            "y": [y],
            "reference": (wd_emb, clip_emb),
            # "smask": xms,
        }
        if hasattr(self, "gram_mask"):
            cond.update({"smask": xms})

        c = self.proj(clip_emb, wd_logits if self.return_logits else wd_emb)

        cond.update({"context": [c]})
        out = [z, cond]
        if return_inputs:
            out.extend([x, xc, xs, xms, xmr])
        return out
