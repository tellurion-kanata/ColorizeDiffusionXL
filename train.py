"""Training script for ColorizeDiffusion XL.
Uses Accelerate + DeepSpeed for distributed multi-GPU training.
See configs/training/ for model configuration files and options.py for CLI arguments.
"""

import logger
import psutil
import os.path as osp

from tqdm import tqdm
from options import Options
from ckpt_util import load_config
from data.dataloader import create_dataloader
from refnet.util import instantiate_from_config, default, exists

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


def get_configurations():
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = "train"
    device_num = torch.cuda.device_count()
    configs = load_config(default(opt.config_file, opt.model_config_file))

    if opt.dynamic_lr:
        base_lr = configs.model.base_learning_rate
        opt.learning_rate = base_lr * opt.batch_size * opt.acumulate_batch_size * len(opt.gpus)
    return parser, opt, configs, device_num


def get_system_memory_usage_gb():
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3), memory.used / memory.total * 100.


if __name__ == '__main__':
    parser, opt, configs, device_num = get_configurations()

    # setup model and data loader
    dataloader, data_size, sampler = create_dataloader(
        opt = opt,
        cfg = configs.dataloader,
        device_num = device_num,
        sample_per_zip = configs.model.params.get("reference_num", 1)
    )
    model = instantiate_from_config(configs.model)

    # use this if testing Colorize Diffusion older than XL
    # params = model.get_trainable_params()
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    tqdm.write(f"Total parameter number: {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f} B, "
               f"trainable parameter number: {sum(p.numel() for p in params) / 1000 ** 3:.2f} B !!!")
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate)
    if exists(opt.pretrained):
        model.init_from_ckpt(
            opt.pretrained,
            make_it_fit = opt.fitting_model,
            ignore_keys = opt.ignore_keys,
            logging = opt.load_logging
        )

    projection_config = ProjectConfiguration(project_dir=opt.ckpt_path)
    accelerator = Accelerator(
        mixed_precision = opt.precision if opt.precision != "fp32" else None,
        log_with = "tensorboard",
        project_config = projection_config,
    )
    if exists(accelerator.state.deepspeed_plugin):
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = opt.batch_size
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if accelerator.is_main_process:
        parser.print_options(opt)

    # resume training
    global_step = 0
    resume_epoch = opt.start_epoch
    resume_batch = 0

    if opt.load_checkpoint:
        accelerator.load_state(opt.load_checkpoint)
        state = logger.CustomCheckpoint.load_training_state(opt.load_checkpoint)
        if state:
            resume_epoch = state["epoch"]
            resume_batch = state.get("batch_idx", 0)
            global_step = state["global_step"]
            tqdm.write(f"Resumed: epoch={resume_epoch}, batch={resume_batch}, global_step={global_step}")
        else:
            tqdm.write(f"WARNING: No training state in checkpoint, resuming from step 0")

    _skip_resumed_callbacks = bool(opt.load_checkpoint)

    # setup loggers
    # TODO: Check if there are alternative methods in huggingface libraries
    vars_opt = vars(opt)
    batch_per_epoch = len(dataloader)
    ckpt_callback = logger.CustomCheckpoint(**vars_opt)
    if accelerator.is_main_process:
        vis_logger = logger.ImageLogger(**vars_opt)
        cli_logger = logger.ConsoleLogger(batch_per_epoch=batch_per_epoch, **vars_opt)
        pbar_epoch = tqdm(initial=resume_epoch, total=opt.epoch, desc="Training process")

    # start training
    model.training = True
    model.on_train_start(resume_epoch)
    ckpt_callback.training_state = {"epoch": resume_epoch, "batch_idx": 0, "global_step": global_step}
    ckpt_callback.on_train_start(accelerator)
    accelerator.init_trackers(opt.name)
    for epoch in range(resume_epoch, opt.epoch):
        if exists(sampler):
            sampler.set_epoch(epoch)

        active_dataloader = accelerator.skip_first_batches(dataloader, resume_batch)
        skip = resume_batch
        resume_batch = 0

        if accelerator.is_main_process:
            pbar_iter = tqdm(initial=skip, total=batch_per_epoch, desc=f"Current epoch {epoch}, process")

        # Epoch starts
        model.on_train_epoch_start(epoch)
        for idx, batch in enumerate(active_dataloader):
            actual_idx = idx + skip

            # forward and backward
            optimizer.zero_grad()
            loss, loss_dict = model.training_step(batch)
            accelerator.backward(loss)
            optimizer.step()

            # batch end callbacks
            model.on_train_batch_end(epoch, global_step)
            if accelerator.sync_gradients:
                loss_dict = {
                    k: accelerator.gather(v.repeat(opt.batch_size)).mean().item() for k, v in loss_dict.items()
                }
                ckpt_callback.training_state = {
                    "epoch": epoch, "batch_idx": actual_idx + 1, "global_step": global_step + 1,
                }
                if not _skip_resumed_callbacks:
                    ckpt_callback.on_train_batch_end(accelerator, global_step, actual_idx)

                if accelerator.is_main_process:
                    logging_dict = loss_dict
                    training_state = {
                        "max_epoch": opt.epoch,
                        "learning_rate": opt.learning_rate,
                        "loss_dict": loss_dict,
                        "batch": batch,
                        "batch_idx": actual_idx,
                        "global_step": global_step,
                        "current_epoch": epoch,
                    }

                    accelerator.log(logging_dict, step=global_step)
                    if not _skip_resumed_callbacks:
                        vis_logger.on_train_batch_end(model, **training_state)
                    cli_logger.on_train_batch_end(**training_state)
                    pbar_iter.set_postfix(loss_dict)
                    pbar_iter.update(1)
                    del logging_dict, training_state

            global_step += 1
            _skip_resumed_callbacks = False
            del loss_dict

        # epoch end callbacks
        model.on_train_epoch_end(epoch)

        if accelerator.sync_gradients:
            ckpt_callback.training_state = {
                "epoch": epoch + 1, "batch_idx": 0, "global_step": global_step,
            }
            ckpt_callback.on_train_epoch_end(accelerator, epoch)

            if accelerator.is_main_process:
                cli_logger.on_train_epoch_end(epoch, global_step, opt.learning_rate)
                pbar_epoch.update(1)
                pbar_iter.close()

    accelerator.save_state(osp.join(opt.ckpt_path, "final"))
    accelerator.end_training()
    if accelerator.is_main_process:
        pbar_epoch.close()