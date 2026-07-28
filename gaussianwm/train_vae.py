import os
import time
import json
import logging
import wandb
import math
from pathlib import Path
from typing import Iterable
import numpy as np
from tqdm import tqdm
from termcolor import cprint
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
import einops
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_gather, knn_points, sample_farthest_points as fps

from gaussianwm.processor.regressor import Splatt3rRegressor
from gaussianwm.encoder.models_ae import create_autoencoder
from gaussianwm.util import distributed_utils, lr_utils
from gaussianwm.util import tensor_utils as TensorUtils
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset
from gaussianwm.util.distributed_utils import NativeScalerWithGradNormCount as NativeScaler
from gaussianwm.util.plot_training_metrics import render as render_loss_plot


class GaussianFeatureCache:
    """Disk cache for post-Splatt3r, post-FPS point clouds."""

    _DTYPES = {"float16": torch.float16, "float32": torch.float32}

    def __init__(self, cache_dir, enabled, dtype, split="train"):
        self.enabled = enabled
        self.dtype = self._DTYPES[dtype]
        self.cache_dir = Path(cache_dir) / split
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, batch_index):
        return self.cache_dir / f"batch_{batch_index:06d}.pt"

    def load(self, batch_index):
        if not self.enabled:
            return None
        path = self._path(batch_index)
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def save(self, batch_index, points):
        if not self.enabled:
            return
        path = self._path(batch_index)
        if path.is_file():
            return
        tmp_path = path.with_suffix(".tmp")
        torch.save(points.detach().to(device="cpu", dtype=self.dtype).contiguous(), tmp_path)
        os.replace(tmp_path, path)


def vae_reconstruction_loss(model, points, cfg):
    """Train the same latent-only decoder path used at inference."""
    encoded = model.encode(points)
    if isinstance(encoded, tuple):
        kl, latents = encoded
        loss_kl = kl.sum() / kl.shape[0]
    else:
        latents = encoded
        loss_kl = None

    reconstructed = model.decode(latents).float()
    targets = points.float()
    loss_chamfer, _ = chamfer_distance(
        reconstructed[..., :3],
        targets[..., :3],
        batch_reduction="mean",
        point_reduction="mean",
    )
    nearest = knn_points(
        reconstructed[..., :3], targets[..., :3], K=1
    )
    target_features = knn_gather(
        targets[..., 3:], nearest.idx
    ).squeeze(2)
    loss_features = F.smooth_l1_loss(
        reconstructed[..., 3:], target_features
    )

    loss = (
        cfg.vae.loss.chamfer_weight * loss_chamfer
        + cfg.vae.loss.feature_weight * loss_features
    )
    if loss_kl is not None:
        loss = loss + cfg.vae.loss.kl_weight * loss_kl
    return loss, loss_chamfer, loss_features, loss_kl


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, cfg=None):
    model.train()
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', distributed_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    splatt3r = None
    cache = GaussianFeatureCache(
        cfg.gaussian_cache.dir,
        cfg.gaussian_cache.enabled,
        cfg.gaussian_cache.dtype,
    )
    accum_iter = cfg.train.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        cprint(f'log_dir: {log_writer.log_dir}', 'green')

    max_batches = len(data_loader)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # DroidDataset is backed by an RLDS pipeline whose source repeats
        # indefinitely.  __len__ defines the intended epoch boundary, so stop
        # explicitly even if the underlying iterator keeps yielding samples.
        if data_iter_step >= max_batches:
            break

        points = cache.load(data_iter_step)
        if points is None:
            if splatt3r is None:
                splatt3r = Splatt3rRegressor().to(device).eval()

            image1 = TensorUtils.to_device(TensorUtils.to_float(batch[0]), device)
            image1 = einops.rearrange(image1, 'b t h w c -> (b t) c h w')
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=cfg.train.amp
            ):
                points, _ = splatt3r.forward_tensor(image1)

            points, _ = fps(points.float(), K=cfg.vae.point_cloud_size)
            cache.save(data_iter_step, points)
        points = points.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
            loss, loss_chamfer, loss_features, loss_kl = (
                vae_reconstruction_loss(model, points, cfg)
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_chamfer=loss_chamfer.item())
        metric_logger.update(loss_features=loss_features.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = distributed_utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, cfg, split="val"):
    model.eval()
    splatt3r = None
    cache = GaussianFeatureCache(
        cfg.gaussian_cache.dir,
        cfg.gaussian_cache.enabled,
        cfg.gaussian_cache.dtype,
        split=split,
    )

    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch_index, batch in enumerate(
        tqdm(metric_logger.log_every(data_loader, 50, header), desc="Evaluation")
    ):
        points = cache.load(batch_index)
        if points is None:
            if splatt3r is None:
                splatt3r = Splatt3rRegressor().to(device).eval()

            image1 = TensorUtils.to_device(TensorUtils.to_float(batch[0]), device)
            image1 = einops.rearrange(image1, 'b t h w c -> (b t) c h w')
            with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
                points, _ = splatt3r.forward_tensor(image1)

            points, _ = fps(points.float(), K=cfg.vae.point_cloud_size)
            cache.save(batch_index, points)

        points = points.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
            loss, loss_chamfer, loss_features, loss_kl = (
                vae_reconstruction_loss(model, points, cfg)
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_chamfer=loss_chamfer.item())
        metric_logger.update(loss_features=loss_features.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    metric_logger.synchronize_between_processes()
    print(f"* Eval loss: {metric_logger.loss.global_avg:.6f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@hydra.main(version_base=None, config_path="../configs", config_name="train_vae")
def main(cfg: DictConfig):
    cfg.distributed.distributed = cfg.distributed.world_size > 1

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    distributed_utils.init_distributed_mode(cfg.distributed)

    logger = logging.getLogger(__name__)
    logger.info(f'Job dir: {os.path.dirname(os.path.realpath(__file__))}')
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    if cfg.use_wandb and distributed_utils.is_main_process():
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name or cfg.vae.name,
            sync_tensorboard=True
        )

    if cfg.seed is not None:
        seed = cfg.seed + distributed_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_gaussian_splatting_reconstruction_dataset('train', cfg=cfg.dataset)
    dataset_val = build_gaussian_splatting_reconstruction_dataset('val', cfg=cfg.dataset)
    dataset_test = build_gaussian_splatting_reconstruction_dataset(
        "test", cfg=cfg.dataset
    )

    logger.info(f'Train dataset size: {len(dataset_train)}')
    logger.info(f'Val dataset size: {len(dataset_val)}')
    logger.info(f'Test dataset size: {len(dataset_test)}')

    is_main_process = distributed_utils.is_main_process()

    if is_main_process and cfg.log_dir is not None and not cfg.eval_only:
        os.makedirs(cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=cfg.log_dir)
        cprint(f"Log directory: {cfg.log_dir}", "green")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader.num_workers,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader.num_workers,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader.num_workers,
    )
    model = create_autoencoder(
        depth=cfg.vae.vae_depth,
        dim=cfg.vae.latent_dim,
        M=cfg.vae.num_latents,
        latent_dim=cfg.vae.latent_dim,
        output_dim=cfg.vae.output_dim,
        N=cfg.vae.point_cloud_size,
        deterministic=not cfg.vae.use_kl,
        decoder_num_queries=cfg.vae.get("decoder_num_queries", None),
    ).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model_without_ddp}")
    logger.info(f'Number of params (M): {n_parameters / 1e6:.2f}')

    eff_batch_size = cfg.train.batch_size * cfg.train.accum_iter * distributed_utils.get_world_size()

    if cfg.optimizer.lr is None:
        cfg.optimizer.lr = cfg.optimizer.blr * eff_batch_size / 256

    logger.info(f'Base lr: {cfg.optimizer.lr * 256 / eff_batch_size:.2e}')
    logger.info(f'Actual lr: {cfg.optimizer.lr:.2e}')
    logger.info(f'Accumulate grad iterations: {cfg.train.accum_iter}')
    logger.info(f'Effective batch size: {eff_batch_size}')

    if cfg.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=cfg.optimizer.lr)
    loss_scaler = NativeScaler()
    logger.info(
        "Criterion: center Chamfer + nearest Gaussian feature SmoothL1"
    )

    distributed_utils.load_model(args=cfg, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if cfg.eval_only:
        test_stats = evaluate(
            model, data_loader_test, device, cfg, split="test"
        )
        logger.info(
            "Eval loss on %d test samples: %.6f",
            len(dataset_test),
            test_stats["loss"],
        )
        return

    logger.info(f"Starting training for {cfg.train.epochs} epochs")
    start_time = time.time()

    for epoch in range(cfg.start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            cfg.optimizer.clip_grad,
            log_writer=log_writer,
            cfg=cfg
        )

        val_stats = None
        if cfg.train.eval_every > 0 and (
            epoch % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs
        ):
            val_stats = evaluate(model, data_loader_val, device, cfg)
            if log_writer is not None:
                for name, value in val_stats.items():
                    log_writer.add_scalar(f'val/{name}', value, epoch)

        if epoch % cfg.train.save_every == 0 or epoch + 1 == cfg.train.epochs:
            distributed_utils.save_model(
                args=cfg, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **({f'val_{k}': v for k, v in val_stats.items()} if val_stats else {}),
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if is_main_process:
            if log_writer is not None:
                log_writer.flush()
            log_path = Path(cfg.paths.metrics_file)
            with log_path.open(mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            render_loss_plot(
                log_path,
                Path(cfg.paths.loss_plot),
                smooth_window=min(10, max(1, epoch + 1)),
            )

        if cfg.use_wandb and is_main_process:
            wandb.log(log_stats)

    total_time = time.time() - start_time
    test_stats = evaluate(
        model, data_loader_test, device, cfg, split="test"
    )
    logger.info("Final test metrics: %s", test_stats)
    logger.info(f'Training time: {total_time / 3600:.2f} hours')

if __name__ == '__main__':
    main()
