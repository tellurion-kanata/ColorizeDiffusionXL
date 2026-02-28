import os
import time
import gc
import random as python_random
import contextlib

import torch
import torchvision
import numpy as np
import PIL.Image as Image

from tqdm import tqdm
from glob import glob
from refnet.util import default
from accelerate import Accelerator

MAXM_SAMPLE_SIZE = 8
scale_factor = 1
ckpt_fmt = "safetensors"


def format_time(second):
    s = int(second)
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def custom_save_state(accelerator: Accelerator, output_dir: str):
    """
    Custom save_state that handles the case where accelerator.save_state() fails
    (e.g. optimizer/model count mismatch in DMD training with 2 optimizers but 1 model).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model(s) — use accelerator.get_state_dict for FSDP-aware gathering
    for i, model in enumerate(accelerator._models):
        model_path = os.path.join(output_dir, f"model_{i}")
        os.makedirs(model_path, exist_ok=True)
        state_dict = accelerator.get_state_dict(model, unwrap=False)
        if accelerator.is_main_process:
            torch.save(state_dict, os.path.join(model_path, "model.bin"))
        del state_dict

    # Save optimizer(s) — each optimizer saved separately
    for i, optimizer in enumerate(accelerator._optimizers):
        optimizer_path = os.path.join(output_dir, f"optimizer_{i}.bin")
        torch.save(optimizer.state_dict(), optimizer_path)

    # Save random states
    random_states = {
        "random_state": torch.get_rng_state(),
        "numpy_random_state": np.random.get_state(),
        "python_random_state": python_random.getstate(),
    }
    if torch.cuda.is_available():
        random_states["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(random_states, os.path.join(output_dir, "random_states.bin"))


def custom_load_state(accelerator: Accelerator, input_dir: str):
    """Counterpart to custom_save_state."""
    # Load model(s)
    for i, model in enumerate(accelerator._models):
        model_path = os.path.join(input_dir, f"model_{i}")
        bin_path = os.path.join(model_path, "model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_state_dict(state_dict, strict=False)
            del state_dict

    # Load optimizer(s)
    for i, optimizer in enumerate(accelerator._optimizers):
        optimizer_path = os.path.join(input_dir, f"optimizer_{i}.bin")
        if os.path.exists(optimizer_path):
            state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state_dict)

    # Load random states
    random_states_path = os.path.join(input_dir, "random_states.bin")
    if os.path.exists(random_states_path):
        random_states = torch.load(random_states_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(random_states["random_state"])
        np.random.set_state(random_states["numpy_random_state"])
        if "python_random_state" in random_states:
            python_random.setstate(random_states["python_random_state"])
        if torch.cuda.is_available() and "cuda_random_state" in random_states:
            torch.cuda.set_rng_state_all(random_states["cuda_random_state"])


def is_custom_checkpoint(checkpoint_dir: str) -> bool:
    """Check if a checkpoint was saved using custom_save_state."""
    return os.path.exists(os.path.join(checkpoint_dir, "optimizer_0.bin"))


class CustomCheckpoint:
    def __init__(
            self,
            save_first_step,
            not_save_first_step_epoch,
            save_freq_step,
            not_save_weight_only,
            ckpt_path,
            start_save_ep,
            save_freq,
            top_k,
            **kwargs
    ):
        self.save_first_step = save_first_step
        self.save_first_step_epoch = not not_save_first_step_epoch
        self.save_freq_step = save_freq_step
        self.save_weight_only = not not_save_weight_only
        self.ckpt_path = ckpt_path
        self.start_save_ep = start_save_ep
        self.save_freq = save_freq
        self.top_k = top_k
        self.prev_ckpts_step = glob(os.path.join(self.ckpt_path, "step-*"))
        self.prev_ckpts_epoch = glob(os.path.join(self.ckpt_path, "epoch-*"))
        self.prev_time = time.time()
        self.training_state = {}

    def _safe_save_state(self, trainer: Accelerator, output_dir: str):
        """Save state with fallback to custom save when accelerator.save_state() fails."""
        trainer.wait_for_everyone()
        num_models = len(trainer._models)
        num_optimizers = len(trainer._optimizers)
        if num_optimizers > num_models:
            custom_save_state(trainer, output_dir)
        else:
            trainer.save_state(output_dir)
        if self.training_state and trainer.is_main_process:
            torch.save(self.training_state, os.path.join(output_dir, "training_state.bin"))

    @staticmethod
    def load_training_state(input_dir):
        """Load training_state.bin from a checkpoint directory. Returns dict or None."""
        path = os.path.join(input_dir, "training_state.bin")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=False)
        return None

    def on_train_start(self, trainer: Accelerator):
        if self.save_first_step:
            filename = os.path.join(self.ckpt_path, f'latest')
            self._safe_save_state(trainer, filename)

            if trainer.is_main_process:
                message = f"Saving latest model to {filename}"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


    def on_train_batch_end(self, trainer: Accelerator, global_step, batch_idx):
        if global_step > 2 and batch_idx % self.save_freq_step == 0 and (batch_idx > 0 or self.save_first_step_epoch):
            filename = os.path.join(self.ckpt_path, f'step-{global_step}')
            self._safe_save_state(trainer, filename)
            self._safe_save_state(trainer, os.path.join(self.ckpt_path, 'latest'))

            if trainer.is_main_process:
                self.prev_ckpts_step.append(filename)
                if len(self.prev_ckpts_step) >= self.top_k:
                    import shutil
                    filename = self.prev_ckpts_step.pop(0)
                    if os.path.exists(filename):
                        shutil.rmtree(filename)

                curtime = time.time()
                interval = curtime - self.prev_time
                self.prev_time = curtime
                message = (f"***** Saving global step [{global_step}] model, interval: {format_time(interval)} *****")
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


    def on_train_epoch_end(self, trainer: Accelerator, current_epoch):
        if current_epoch >= self.start_save_ep and current_epoch % self.save_freq == 0:
            filename = os.path.abspath(os.path.join(self.ckpt_path, f"epoch-{current_epoch}"))
            self._safe_save_state(trainer, filename)

            if trainer.is_main_process:
                self.prev_ckpts_epoch.append(filename)
                if len(self.prev_ckpts_epoch) >= self.top_k:
                    import shutil
                    filename = self.prev_ckpts_epoch.pop(0)
                    if os.path.exists(filename):
                        shutil.rmtree(filename)

                message = f"***** Saving epoch [{current_epoch}] model to {filename} *****"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
                train_log.write(message + '\n')
                train_log.close()


class ConsoleLogger:
    def __init__(
            self,
            print_freq,
            ckpt_path,
            batch_per_epoch,
            stage="train",
            **kwargs
    ):
        self.step_frequcy = print_freq
        self.ckpt_path = ckpt_path
        self.batch_per_epoch = batch_per_epoch

        current_time = time.time()
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time
        self.total_time = 0

        self.stage = stage

    def on_train_batch_end(
            self,
            batch_idx,
            loss_dict,
            current_epoch,
            max_epoch,
            global_step,
            learning_rate,
            **kwargs
    ):
        if batch_idx % self.step_frequcy == 0:
            current_time = time.time()
            fmt_curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            iter_time = current_time - self.pre_iter_time
            self.total_time += iter_time
            self.pre_iter_time = current_time

            message = f"{fmt_curtime}"
            message += f", iter_time: {format_time(iter_time)}"
            message += f", total_time: {format_time(self.total_time)}"
            message += f", epoch: [{current_epoch}/{max_epoch}]"
            message += f", step: [{batch_idx}/{self.batch_per_epoch}]"
            message += f", global_step: {global_step}"
            message += f", lr: {learning_rate:.7f}"

            for label in loss_dict:
                message += f', {label}: {loss_dict[label]:.6f}'

            tqdm.write(message)
            train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
            train_log.write(message + '\n')
            train_log.close()

    def on_train_epoch_end(self, current_epoch, global_step, learning_rate):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        epoch_time = current_time - self.pre_epoch_time
        self.total_time += iter_time
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time

        message = "{ "
        message += f"Epcoh {current_epoch} finished"
        message += f",\tglobal_step: {global_step}"
        message += f",\ttotal_time: {format_time(self.total_time)}"
        message += f",\tepoch_time: {format_time(epoch_time)}"
        message += f", current_lr: {learning_rate:.7f}"
        message += " }"

        tqdm.write(message)
        train_log = open(os.path.join(self.ckpt_path, 'logs.txt'), 'a')
        train_log.write(message + '\n\n')
        train_log.close()


class ImageLogger:
    def __init__(
            self,
            ckpt_path,
            not_log_samples = False,
            save_freq_step = 1,
            sample_num = None,
            clamp = True,
            increase_log_steps = True,
            batch_size = None,
            rescale = True,
            disabled = False,
            check_memory_use = False,
            log_on_batch_idx = True,
            log_first_step = True,
            sampler = "dpm",
            scheduler = "Karras",
            step = 20,
            guidance_scale = 1.0,
            save_input = False,
            load_checkpoint = None,
            pretrained = None,
            log_img_step = None,
            wandb_log_images = False,
            **kwargs
    ):
        self.log = not not_log_samples
        self.wandb_log_images = wandb_log_images
        self.clamp = clamp
        self.rescale = rescale
        self.save_input = save_input
        self.log_on_batch_idx = log_on_batch_idx
        self.disabled = disabled or sample_num == 0
        self.sample_num = default(sample_num, batch_size)
        self.batch_freq = default(log_img_step, save_freq_step)
        self.save_path = os.path.join(ckpt_path, kwargs.pop("mode"))
        self.log_first_step = log_first_step if not check_memory_use else False

        self.autocast = contextlib.nullcontext
        self.sampling_params = {
            "unconditional_guidance_scale": guidance_scale,
            "sampler": sampler,
            "scheduler": scheduler,
            "step": step,
        }

        if load_checkpoint is None and pretrained is None and not check_memory_use:
            self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        else:
            self.log_steps = []
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]


    def log_local(self, images, global_step, current_epoch, batch_idx, is_train):
        if not getattr(self, 'save_locally', True):
            return

        def save_image(img, path):
            if self.rescale:
                img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            img = img.permute(1, 2, 0).squeeze(-1)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            w, h = img.size
            img.resize((w // scale_factor, h // scale_factor)).save(path)

        for k in images:
            dirpath = os.path.join(self.save_path, k)
            os.makedirs(dirpath, exist_ok=True)

            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu().float()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)
            if len(images[k].shape) == 3:
                images[k] = images[k].unsqueeze(0)

            if is_train:
                # save grid images during training
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}.png"
                path = os.path.join(dirpath, filename)
                save_image(grid, path)

            else:
                # save images separately during testing
                for idx in range(len(images[k])):
                    img = images[k][idx]
                    filename = f"{batch_idx}_{idx}.png"
                    path = os.path.join(dirpath, filename)
                    save_image(img, path)

    def log_img(self, model, batch, batch_idx, global_step=None, current_epoch=None, **kwargs):
        is_train = model.training
        if (
                self.log and hasattr(model, "log_images")
                and callable(model.log_images) and self.sample_num > 0
        ) or not is_train:

            if is_train:
                model.eval()

            with self.autocast():
                images = model.log_images(
                    N = min(self.sample_num, MAXM_SAMPLE_SIZE) if is_train else self.sample_num,
                    batch = batch,
                    return_inputs = is_train or self.save_input,
                    **self.sampling_params,
                    **kwargs,
                )

            self.log_local(images, global_step, current_epoch, batch_idx, is_train)
            if self.wandb_log_images and getattr(self, 'save_locally', True) and is_train:
                self._log_wandb(images, global_step)
            if is_train:
                model.train()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _log_wandb(self, images, global_step):
        import wandb
        log_dict = {}
        for k, v in images.items():
            if isinstance(v, torch.Tensor):
                if self.rescale:
                    v = (v + 1.0) / 2.0
                grid = torchvision.utils.make_grid(v, nrow=4)
                grid = grid.permute(1, 2, 0).numpy()
                grid = (grid * 255).clip(0, 255).astype(np.uint8)
                log_dict[f"train/{k}"] = wandb.Image(grid)
        if log_dict:
            wandb.log(log_dict, step=global_step)

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, model, global_step, current_epoch, batch, batch_idx, **kwargs):
        check_idx = batch_idx if self.log_on_batch_idx else global_step
        if not self.disabled and global_step > 0 and self.check_frequency(check_idx):
            self.log_img(model, batch, batch_idx, global_step, current_epoch, **kwargs)

    def on_test_batch_end(self, model, batch, batch_idx):
        self.log_img(model, batch, batch_idx)