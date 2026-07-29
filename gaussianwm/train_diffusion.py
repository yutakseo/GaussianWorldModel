import json
import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from gaussianwm.gwm_predictor import GaussianPredictor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset
from gaussianwm.util import distributed_utils
from gaussianwm.util.logging_utils import (
    _recursive_flatten_dict,
    print_rich_single_line_metrics,
)
from gaussianwm.util.plot_training_metrics import render as render_loss_plot
from gaussianwm.util.runtime import (
    make_data_loader,
    prepare_sequence_batch,
    resolve_path,
    seed_everything,
    unwrap_model,
)
from gaussianwm.util.timer_utils import Timer


def train_step(model, batch, optimizer, step, cfg):
    """Train for one step"""
    batch = prepare_sequence_batch(batch, unwrap_model(model).device)

    optimizer.zero_grad(set_to_none=True)
    total_loss, metrics = model(
        batch,
        update_tokenizer=cfg.train.update_tokenizer,
        update_model=cfg.train.update_model
    )
    total_loss.backward()
    max_grad_norm = cfg.world_model.optimizer.max_grad_norm
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in unwrap_model(model).parameters()
                if parameter.requires_grad
            ),
            float(max_grad_norm),
        )
    optimizer.step()
    return metrics


@torch.no_grad()
def validate(model, val_loader, cfg):
    """Evaluate a bounded number of validation batches and return mean metrics."""
    metric_sums = {}
    num_batches = 0

    for batch in val_loader:
        batch = prepare_sequence_batch(batch, unwrap_model(model).device)
        _, metrics = model(
            batch,
            update_tokenizer=cfg.train.update_tokenizer,
            update_model=cfg.train.update_model,
            eval_mode=True,
        )
        for name, value in metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        num_batches += 1
        if num_batches >= cfg.eval.num_batches:
            break

    if num_batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {name: value / num_batches for name, value in metric_sums.items()}


def checkpoint_step_from_path(path):
    """Extract a step from legacy names such as model_64424.pt."""
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def training_step_range(start_step, max_step):
    """Return an inclusive range whose final optimizer step is max_step."""
    if start_step > max_step:
        return range(0)
    return range(start_step, max_step + 1)


def append_metrics(path, step, split, metrics):
    """Append one structured metrics record for plotting and later analysis."""
    record = {
        "timestamp": time.time(),
        "step": int(step),
        "split": split,
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        metrics_file.flush()



def log_metrics(metrics, step, logger, use_wandb=False):
    """Log metrics to console and wandb if enabled"""
    # logger.info(f"Step {step} metrics:")
    # for k, v in metrics.items():
    #     logger.info(f"{k}: {v:.6f}")
    if use_wandb:
        import wandb
        wandb.log(metrics, step=step)


@hydra.main(
    version_base=None, config_path="../configs", config_name="train_gwm"
)
def main(cfg: DictConfig):
    distributed_utils.init_distributed_mode(cfg.distributed)
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    
    if cfg.use_wandb and distributed_utils.is_main_process():
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name or f"gwm_{time.strftime('%Y%m%d_%H%M%S')}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    model = GaussianPredictor(cfg.world_model, device=device).to(device)
    optimizer = model.model_optimizer
    if cfg.distributed.distributed:
        model = DDP(model, device_ids=[cfg.distributed.gpu], find_unused_parameters=True)

    train_dataset = build_gaussian_splatting_reconstruction_dataset("train", cfg.dataset)
    val_dataset = build_gaussian_splatting_reconstruction_dataset("val", cfg.dataset)
    test_dataset = build_gaussian_splatting_reconstruction_dataset(
        "test", cfg.dataset
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = make_data_loader(
        train_dataset,
        batch_size=cfg.world_model.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    val_loader = make_data_loader(
        val_dataset,
        batch_size=cfg.world_model.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    test_loader = make_data_loader(
        test_dataset,
        batch_size=cfg.world_model.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    if distributed_utils.is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = cfg.start_step
    if cfg.resume:
        resume_path = resolve_path(cfg.resume, Path.cwd())
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info(f"Loading checkpoint from {resume_path}")
        model_to_load = unwrap_model(model)
        loaded_step = model_to_load.load_snapshot(
            resume_path.parent,
            suffix=resume_path.stem.removeprefix("model"),
            optimizer=optimizer,
        )
        checkpoint_step = loaded_step
        if checkpoint_step is None:
            checkpoint_step = checkpoint_step_from_path(resume_path)
            logger.warning(
                "Legacy checkpoint has no optimizer state; AdamW will restart "
                "with fresh momentum."
            )
        if checkpoint_step is None and start_step <= 0:
            raise ValueError(
                "Could not infer checkpoint step; set start_step explicitly."
            )
        if checkpoint_step is not None:
            start_step = checkpoint_step + 1
        logger.info(f"Resuming from step {start_step}")
    
    is_main_process = distributed_utils.is_main_process()
    metrics_path = resolve_path(cfg.paths.metrics_file, Path.cwd())
    loss_plot_path = resolve_path(cfg.paths.loss_plot, Path.cwd())
    if is_main_process:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")
    logger.info(
        "Training step range: %d through %d (inclusive)",
        start_step,
        cfg.train.max_steps,
    )
    if start_step > cfg.train.max_steps:
        logger.info(
            "Checkpoint already reached max_steps=%d; nothing to train.",
            cfg.train.max_steps,
        )
        return
    step = start_step

    train_iter = iter(train_loader)
    
    progress_bar = tqdm(
        training_step_range(start_step, cfg.train.max_steps),
        desc="Training",
        initial=start_step,
        total=cfg.train.max_steps + 1,
    )
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
                append_metrics(metrics_path, step, "train", step_metrics)
                if step % cfg.train.plot_every == 0:
                    render_loss_plot(
                        metrics_path, loss_plot_path,
                        smooth_window=max(1, cfg.train.plot_every // cfg.train.log_every),
                    )
                print_rich_single_line_metrics(metrics)

        if step % cfg.eval.eval_every == 0 and step > start_step:
            with timer.context("validation"):
                validation_metrics = validate(model, val_loader, cfg)
            if is_main_process:
                logger.info(
                    "Validation at step %d: %s", step, validation_metrics
                )
                log_metrics(
                    {f"validation/{k}": v for k, v in validation_metrics.items()},
                    step,
                    logger,
                    cfg.use_wandb,
                )
                append_metrics(
                    metrics_path, step, "validation", validation_metrics
                )
                render_loss_plot(
                    metrics_path, loss_plot_path,
                    smooth_window=max(1, cfg.train.plot_every // cfg.train.log_every),
                )
                print_rich_single_line_metrics(
                    {"validation": validation_metrics}
                )
        
        if is_main_process and step % cfg.train.save_every == 0 and step > 0:
            logger.info(f"Saving model checkpoint at step {step}")
            model_to_save = unwrap_model(model)
            model_to_save.save_snapshot(
                checkpoint_dir, suffix=f"_{step}", optimizer=optimizer, step=step
            )
            model_to_save.save_snapshot(
                checkpoint_dir, suffix="_latest", optimizer=optimizer, step=step
            )
    
    logger.info("Saving final model")
    if is_main_process:
        model_to_save = unwrap_model(model)
        model_to_save.save_snapshot(
            checkpoint_dir, suffix=f"_{step}", optimizer=optimizer, step=step
        )
        model_to_save.save_snapshot(
            checkpoint_dir, suffix="_latest", optimizer=optimizer, step=step
        )
        final_dir = checkpoint_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model_to_save.save_snapshot(
            final_dir, optimizer=optimizer, step=step
        )
    test_metrics = validate(model, test_loader, cfg)
    if is_main_process:
        logger.info("Final test metrics: %s", test_metrics)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
