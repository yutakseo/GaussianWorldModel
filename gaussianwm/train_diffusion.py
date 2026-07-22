import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import cv2
import imageio
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

import util.distributed_utils as distributed_utils
from util.logging_utils import print_rich_single_line_metrics, _recursive_flatten_dict
from util.timer_utils import Timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussianwm.gwm_predictor import GaussianPredictor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset

def collate_fn(batch):
    """Custom collate function to handle different dataset formats"""
    if isinstance(batch[0], tuple) and len(batch[0]) == 4:
        obs = torch.stack([item[0] for item in batch])
        action = torch.stack([item[1] for item in batch])
        reward = torch.stack([item[2] for item in batch])
        pad_mask = torch.stack([item[3] for item in batch])
        return obs, action, reward, pad_mask
    elif isinstance(batch[0], tuple) and len(batch[0]) == 3:
        obs = torch.stack([item[0] for item in batch])
        action = torch.stack([item[1] for item in batch])
        reward = torch.stack([item[2] for item in batch])
        return obs, action, reward
    else:
        raise ValueError(f"Unsupported batch format: {type(batch[0])}")


def train_step(model, batch, optimizer, step, cfg):
    """Train for one step"""
    # [B, T, H, W, C] -> [B, T, C, H, W]
    batch[0] = batch[0].permute(0, 1, 4, 2, 3).to(model.device)

    total_loss, metrics = model(
        batch,
        update_tokenizer=cfg.train.update_tokenizer,
        update_model=cfg.train.update_model
    )
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return metrics



def log_metrics(metrics, step, logger, use_wandb=False):
    """Log metrics to console and wandb if enabled"""
    # logger.info(f"Step {step} metrics:")
    # for k, v in metrics.items():
    #     logger.info(f"{k}: {v:.6f}")
    if use_wandb:
        import wandb
        wandb.log(metrics, step=step)


@hydra.main(config_path="../configs", config_name="train_gwm")
def main(cfg: DictConfig):
    distributed_utils.init_distributed_mode(cfg.distributed)
    device = torch.device(cfg.device)

    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))
    
    work_dir = Path(os.getcwd())
    logger.info(f"Working directory: {work_dir}")
    
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        # torch.backends.cudnn.deterministic = True
    
    if cfg.use_wandb and distributed_utils.is_main_process():
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name or f"gwm_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    model = GaussianPredictor(cfg.world_model).to(device)
    optimizer = model.model_optimizer
    if cfg.distributed.distributed:
        model = DDP(model, device_ids=[cfg.distributed.gpu], find_unused_parameters=True)

    train_dataset = build_gaussian_splatting_reconstruction_dataset("train", cfg.dataset)
    val_dataset = build_gaussian_splatting_reconstruction_dataset("val", cfg.dataset)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.world_model.batch_size,
        # sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
        # collate_fn=collate_fn,
        # drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.world_model.batch_size,
        # sampler=val_sampler,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
        # collate_fn=collate_fn,
        # drop_last=True
    )
    
    start_step = 0
    # if cfg.resume and os.path.exists(cfg.resume):
    #     logger.info(f"Loading checkpoint from {cfg.resume}")
    #     model.load_snapshot(cfg.resume)
    #     # Extract step number if available in the checkpoint name
    #     if '_' in os.path.basename(cfg.resume):
    #         try:
    #             checkpoint_step = os.path.basename(cfg.resume).split('_')[-1]
    #             if checkpoint_step.endswith('.pt'):
    #                 checkpoint_step = checkpoint_step[:-3]
    #             start_step = int(checkpoint_step) + 1
    #             logger.info(f"Resuming from step {start_step}")
    #         except:
    #             logger.info("Could not determine start step from checkpoint filename")
    
    is_main_process = distributed_utils.is_main_process()

    logger.info("Starting training...")
    step = start_step

    train_iter = iter(train_loader)
    
    progress_bar = tqdm(range(start_step, cfg.train.max_steps), desc="Training", initial=start_step, total=cfg.train.max_steps)
    timer = Timer()
    for step in progress_bar:
        with timer.context("data"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
        metrics = {}
        with timer.context("train"):
            step_metrics = train_step(model, batch, optimizer, step, cfg)

        step_metrics["lr"] = optimizer.param_groups[0]['lr']
        metrics.update({"training": step_metrics})
        metrics.update({"timer": timer.get_average_times()})

        metrics_flat = _recursive_flatten_dict(metrics)
        metrics_final = {k: v for k, v in zip(*metrics_flat)}

        if step % cfg.train.log_every == 0 and step > 0:
            # metrics = {k: v / num_steps_for_avg for k, v in metrics_accumulator.items()}
            if is_main_process:
                log_metrics(metrics_final, step, logger, cfg.use_wandb)
                print_rich_single_line_metrics(metrics)
        
        if is_main_process and step % cfg.train.save_every == 0 and step > 0:
            logger.info(f"Saving model checkpoint at step {step}")
            # checkpoint_dir = work_dir / "checkpoints"
            checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            model_to_save = model.module if cfg.distributed.distributed else model
            model_to_save.save_snapshot(checkpoint_dir, suffix=f"_{step}")
            model_to_save.save_snapshot(checkpoint_dir, suffix="_latest")
    
    logger.info("Saving final model")
    final_dir = checkpoint_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model_to_save = model.module if cfg.distributed.distributed else model
    model_to_save.save_snapshot(final_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
