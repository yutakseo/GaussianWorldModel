import os
import time
import json
import logging
import hashlib
import wandb
import math
import sys
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

from gaussianwm.processor.regressor import Splatt3rRegressor
from gaussianwm.encoder.models_ae import (
    create_autoencoder,
    sample_farthest_gaussians,
)
from gaussianwm.rendering import (
    estimate_intrinsics_from_dense_gaussians,
    render_gaussians,
)
from gaussianwm.util import distributed_utils
from gaussianwm.util import tensor_utils as TensorUtils
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset
from gaussianwm.util.distributed_utils import NativeScalerWithGradNormCount as NativeScaler
from gaussianwm.util.plot_training_metrics import render as render_loss_plot


class GaussianFeatureCache:
    """Versioned cache for VAE input Gaussians and camera calibration.

    Cache entries are keyed by the source image tensor, not by iterator index.
    RLDS training is shuffled, so an index-only cache can silently pair a
    Gaussian target from one trajectory with a later trajectory.
    """

    _DTYPES = {"float16": torch.float16, "float32": torch.float32}
    # V6 changes the unit from a shuffled *batch* to one RGB sequence.  A
    # batch key is unsafe operationally: the same sequences are regrouped on
    # every epoch, so it continually creates duplicate cache entries.
    VERSION = 6

    def __init__(
        self, cache_dir, enabled, dtype, split="train", max_entries=None
    ):
        self.enabled = enabled
        self.dtype = self._DTYPES[dtype]
        self.cache_dir = Path(cache_dir) / split
        self.max_entries = (
            int(max_entries) if max_entries is not None else None
        )
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError("gaussian_cache.max_entries must be positive")
        self._entry_count = 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._entry_count = sum(
                1 for _ in self.cache_dir.glob("sample_*.pt")
            )

    @staticmethod
    def _key_for_tensor(images):
        """Return a stable content key for one RGB sequence tensor."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "Gaussian cache keys require a torch.Tensor image sequence, got "
                f"{type(images)!r}"
            )
        tensor = images.detach().to(device="cpu").contiguous()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def keys_for_images(self, images):
        """Return one cache key per ``[T,H,W,C]`` sequence in a batch."""
        if not isinstance(images, torch.Tensor) or images.ndim != 5:
            raise ValueError(
                "Expected source images [B,T,H,W,C] for Gaussian cache keys, "
                f"got {type(images)!r} with shape "
                f"{getattr(images, 'shape', None)}"
            )
        return [self._key_for_tensor(sequence) for sequence in images]

    def _path(self, key):
        return self.cache_dir / f"sample_{key}.pt"

    def load(self, key):
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            return None
        if "points" not in payload or "intrinsics" not in payload:
            return None
        points = payload["points"]
        intrinsics = payload["intrinsics"]
        # V6 stores one sequence at a time: [T,2048,14] and [T,3,3].
        # Reject malformed entries rather than failing later in the VAE.
        if (
            not isinstance(points, torch.Tensor)
            or not isinstance(intrinsics, torch.Tensor)
            or points.ndim != 3
            or intrinsics.ndim != 3
            or points.shape[0] != intrinsics.shape[0]
        ):
            return None
        return points, intrinsics

    def save(self, key, points, intrinsics):
        if not self.enabled:
            return
        if (
            points.ndim != 3
            or intrinsics.ndim != 3
            or points.shape[0] != intrinsics.shape[0]
        ):
            raise ValueError(
                "Gaussian cache entries must be per-sequence [T,N,D] and "
                f"[T,3,3], got {tuple(points.shape)} and "
                f"{tuple(intrinsics.shape)}"
            )
        path = self._path(key)
        is_new_entry = not path.is_file()
        if (
            is_new_entry
            and self.max_entries is not None
            and self._entry_count >= self.max_entries
        ):
            return
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "version": self.VERSION,
                "points": points.detach()
                .to(device="cpu", dtype=self.dtype)
                .contiguous(),
                "intrinsics": intrinsics.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        if is_new_entry:
            self._entry_count += 1


class LegacyGaussianBatchCache:
    """Read version-4 batch caches as an immutable sequence collection.

    Version 4 stored two 12-frame sequences in each ``batch_*.pt`` file.
    Mapping a new DataLoader batch directly to the same file index is wrong
    when the new batch size differs. This adapter indexes the cached sequence
    stream explicitly and can therefore split a legacy file safely.
    """

    VERSION = 4

    def __init__(
        self,
        cache_dir,
        split,
        segment_length,
        point_cloud_size,
        feature_dim=14,
    ):
        self.enabled = cache_dir is not None
        self.segment_length = int(segment_length)
        self.point_cloud_size = int(point_cloud_size)
        self.feature_dim = int(feature_dim)
        self.cache_dir = (
            Path(cache_dir) / split if self.enabled else None
        )
        self.files = []
        self.sequences_per_full_file = 0
        self.num_sequences = 0
        self._loaded_file_index = None
        self._loaded_payload = None
        if not self.enabled:
            return
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"Legacy Gaussian cache split not found: {self.cache_dir}"
            )

        self.files = sorted(self.cache_dir.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No legacy batch cache files found in {self.cache_dir}"
            )
        indices = [
            int(path.stem.removeprefix("batch_")) for path in self.files
        ]
        expected_indices = list(range(len(self.files)))
        if indices != expected_indices:
            raise ValueError(
                "Legacy cache files must be contiguous from batch_000000.pt"
            )

        first = self._load_file(0)
        first_frames = first[0].shape[0]
        if first_frames % self.segment_length:
            raise ValueError(
                "Legacy cache frame count is not divisible by segment "
                f"length: {first_frames} vs {self.segment_length}"
            )
        self.sequences_per_full_file = (
            first_frames // self.segment_length
        )
        if self.sequences_per_full_file <= 0:
            raise ValueError("Legacy cache contains no complete sequence")

        last = self._load_file(len(self.files) - 1)
        last_frames = last[0].shape[0]
        if (
            last_frames % self.segment_length
            or last_frames > first_frames
        ):
            raise ValueError(
                f"Malformed final legacy cache file with {last_frames} frames"
            )
        self.num_sequences = (
            (len(self.files) - 1) * self.sequences_per_full_file
            + last_frames // self.segment_length
        )
        self._loaded_file_index = None
        self._loaded_payload = None

    def _validate_payload(self, path, payload):
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError(
                f"Expected legacy cache version {self.VERSION}: {path}"
            )
        points = payload.get("points")
        intrinsics = payload.get("intrinsics")
        if (
            not isinstance(points, torch.Tensor)
            or not isinstance(intrinsics, torch.Tensor)
            or points.ndim != 3
            or tuple(points.shape[1:])
            != (self.point_cloud_size, self.feature_dim)
            or intrinsics.ndim != 3
            or tuple(intrinsics.shape[1:]) != (3, 3)
            or points.shape[0] != intrinsics.shape[0]
        ):
            raise ValueError(
                f"Malformed legacy cache payload in {path}: "
                f"points={getattr(points, 'shape', None)}, "
                f"intrinsics={getattr(intrinsics, 'shape', None)}"
            )
        return points, intrinsics

    def _load_file(self, file_index):
        if self._loaded_file_index == file_index:
            return self._loaded_payload
        path = self.files[file_index]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validated = self._validate_payload(path, payload)
        self._loaded_file_index = file_index
        self._loaded_payload = validated
        return validated

    def load_sequences(self, sequence_start, sequence_count):
        """Load a contiguous sequence range, splitting files as needed."""
        if not self.enabled:
            return None
        sequence_start = int(sequence_start)
        sequence_count = int(sequence_count)
        if sequence_start < 0 or sequence_count <= 0:
            raise ValueError("Invalid legacy sequence range")
        sequence_end = sequence_start + sequence_count
        if sequence_end > self.num_sequences:
            raise IndexError(
                f"Legacy cache has {self.num_sequences} sequences, requested "
                f"[{sequence_start}, {sequence_end})"
            )

        point_chunks = []
        intrinsics_chunks = []
        cursor = sequence_start
        while cursor < sequence_end:
            file_index = cursor // self.sequences_per_full_file
            sequence_in_file = (
                cursor % self.sequences_per_full_file
            )
            points, intrinsics = self._load_file(file_index)
            available = (
                points.shape[0] // self.segment_length
                - sequence_in_file
            )
            take = min(available, sequence_end - cursor)
            frame_start = sequence_in_file * self.segment_length
            frame_end = frame_start + take * self.segment_length
            point_chunks.append(points[frame_start:frame_end])
            intrinsics_chunks.append(intrinsics[frame_start:frame_end])
            cursor += take

        return (
            torch.cat(point_chunks, dim=0),
            torch.cat(intrinsics_chunks, dim=0),
        )


def load_or_compute_gaussian_batch(images, cache, splatt3r, device, cfg):
    """Return cached or frozen-Splatt3R VAE inputs for a BTHWC RGB batch.

    The cache is intentionally per input sequence instead of per DataLoader
    batch.  RLDS shuffles and re-groups sequences each epoch; caching an
    entire batch therefore grows without bound while yielding almost no hits.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 5:
        raise ValueError(
            "Expected VAE source images [B,T,H,W,C], got "
            f"{type(images)!r} with shape {getattr(images, 'shape', None)}"
        )

    batch_size, time_steps, height, width, _ = images.shape
    if cache.enabled:
        keys = cache.keys_for_images(images)
        entries = [cache.load(key) for key in keys]
    else:
        keys = [None] * batch_size
        entries = [None] * batch_size
    missing_indices = [
        index for index, entry in enumerate(entries) if entry is None
    ]

    if missing_indices:
        if splatt3r is None:
            splatt3r = Splatt3rRegressor().to(device).eval()

        missing_images = images[missing_indices]
        missing_images = TensorUtils.to_device(
            TensorUtils.to_float(missing_images), device
        )
        flattened_images = einops.rearrange(
            missing_images, "b t h w c -> (b t) c h w"
        )
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, enabled=cfg.train.amp
        ):
            dense_points, _ = splatt3r.forward_tensor(flattened_images)
            intrinsics = estimate_intrinsics_from_dense_gaussians(
                dense_points, flattened_images.shape[-2:]
            )

        sampled_points, _ = sample_farthest_gaussians(
            dense_points.float(), cfg.vae.point_cloud_size
        )
        sampled_points = sampled_points.reshape(
            len(missing_indices), time_steps, *sampled_points.shape[1:]
        )
        intrinsics = intrinsics.reshape(
            len(missing_indices), time_steps, *intrinsics.shape[1:]
        )
        for local_index, batch_index in enumerate(missing_indices):
            # Keep all entries on CPU before stacking; a mixed hit/miss batch
            # otherwise mixes CPU cache tensors with CUDA tensors.
            points = sampled_points[local_index].detach().to(
                device="cpu",
                dtype=cache.dtype if cache.enabled else sampled_points.dtype,
            )
            camera = intrinsics[local_index].detach().to(
                device="cpu", dtype=torch.float32
            )
            cache.save(keys[batch_index], points, camera)
            entries[batch_index] = (points, camera)

    if any(entry is None for entry in entries):
        raise RuntimeError("Gaussian cache did not produce every batch entry")

    points = torch.stack([entry[0] for entry in entries], dim=0).reshape(
        batch_size * time_steps, *entries[0][0].shape[1:]
    )
    intrinsics = torch.stack(
        [entry[1] for entry in entries], dim=0
    ).reshape(batch_size * time_steps, *entries[0][1].shape[1:])
    return (
        points.to(device, non_blocking=True),
        intrinsics.to(device, non_blocking=True),
        (height, width),
        splatt3r,
    )


def vae_reconstruction_loss(model, points, intrinsics, image_size, cfg):
    """Use upstream input queries with the paper's reconstruction losses."""
    decoder_queries = (
        points
        if cfg.vae.get("decoder_num_queries", None) is not None
        else None
    )
    outputs = model(points, decoder_queries)
    reconstructed = outputs["logits"].float()
    loss_kl = outputs.get("kl")
    if loss_kl is not None:
        loss_kl = loss_kl.sum() / loss_kl.shape[0]
    targets = points.float()
    loss_chamfer, _ = chamfer_distance(
        reconstructed[..., :3],
        targets[..., :3],
        batch_reduction="mean",
        point_reduction="mean",
    )
    # The CUDA Gaussian rasterizer is differentiable but float32-only.
    with torch.autocast(device_type=points.device.type, enabled=False):
        rendered_reconstruction = render_gaussians(
            reconstructed.float(), image_size, intrinsics.float()
        )
        with torch.no_grad():
            rendered_target = render_gaussians(
                targets.float(), image_size, intrinsics.float()
            )
    loss_render = F.l1_loss(rendered_reconstruction, rendered_target)

    loss = (
        cfg.vae.loss.chamfer_weight * loss_chamfer
        + cfg.vae.loss.render_weight * loss_render
    )
    if loss_kl is not None:
        loss = loss + cfg.vae.loss.kl_weight * loss_kl
    return loss, loss_chamfer, loss_render, loss_kl


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
        max_entries=cfg.gaussian_cache.get("max_entries", None),
    )
    legacy_cache = LegacyGaussianBatchCache(
        cfg.gaussian_cache.get("legacy_batch_dir", None),
        split="train",
        segment_length=cfg.dataset.segment_length,
        point_cloud_size=cfg.vae.point_cloud_size,
    )
    if legacy_cache.enabled:
        expected_sequences = len(data_loader.dataset)
        if legacy_cache.num_sequences < expected_sequences:
            raise ValueError(
                "Legacy train cache is incomplete: "
                f"{legacy_cache.num_sequences} < {expected_sequences}"
            )
        cprint(
            "[Gaussian cache] read-only v4 sequence adapter: "
            f"{legacy_cache.num_sequences} train sequences",
            "cyan",
        )
    accum_iter = cfg.train.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        cprint(f'log_dir: {log_writer.log_dir}', 'green')

    max_batches = len(data_loader)
    if cfg.train.max_batches_per_epoch is not None:
        max_batches = min(max_batches, cfg.train.max_batches_per_epoch)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # DroidDataset is backed by an RLDS pipeline whose source repeats
        # indefinitely.  __len__ defines the intended epoch boundary, so stop
        # explicitly even if the underlying iterator keeps yielding samples.
        if data_iter_step >= max_batches:
            break

        if legacy_cache.enabled:
            current_batch_size = batch[0].shape[0]
            sequence_start = data_iter_step * cfg.train.batch_size
            points, intrinsics = legacy_cache.load_sequences(
                sequence_start, current_batch_size
            )
            image_size = tuple(batch[0].shape[2:4])
            points = points.to(device, non_blocking=True)
            intrinsics = intrinsics.to(device, non_blocking=True)
        else:
            points, intrinsics, image_size, splatt3r = (
                load_or_compute_gaussian_batch(
                    batch[0], cache, splatt3r, device, cfg
                )
            )

        with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
            loss, loss_chamfer, loss_render, loss_kl = (
                vae_reconstruction_loss(
                    model, points, intrinsics, image_size, cfg
                )
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

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_chamfer=loss_chamfer.item())
        metric_logger.update(loss_render=loss_render.item())

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
        max_entries=cfg.gaussian_cache.get("max_entries", None),
    )
    legacy_cache = LegacyGaussianBatchCache(
        cfg.gaussian_cache.get("legacy_batch_dir", None),
        split=split,
        segment_length=cfg.dataset.segment_length,
        point_cloud_size=cfg.vae.point_cloud_size,
    )
    if legacy_cache.enabled:
        expected_sequences = len(data_loader.dataset)
        if legacy_cache.num_sequences < expected_sequences:
            raise ValueError(
                f"Legacy {split} cache is incomplete: "
                f"{legacy_cache.num_sequences} < {expected_sequences}"
            )
        cprint(
            "[Gaussian cache] read-only v4 sequence adapter: "
            f"{legacy_cache.num_sequences} {split} sequences",
            "cyan",
        )

    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch_index, batch in enumerate(
        tqdm(metric_logger.log_every(data_loader, 50, header), desc="Evaluation")
    ):
        if (
            cfg.train.max_eval_batches is not None
            and batch_index >= cfg.train.max_eval_batches
        ):
            break
        if legacy_cache.enabled:
            current_batch_size = batch[0].shape[0]
            sequence_start = batch_index * cfg.train.batch_size
            points, intrinsics = legacy_cache.load_sequences(
                sequence_start, current_batch_size
            )
            image_size = tuple(batch[0].shape[2:4])
            points = points.to(device, non_blocking=True)
            intrinsics = intrinsics.to(device, non_blocking=True)
        else:
            points, intrinsics, image_size, splatt3r = (
                load_or_compute_gaussian_batch(
                    batch[0], cache, splatt3r, device, cfg
                )
            )

        with torch.amp.autocast(device_type=device.type, enabled=cfg.train.amp):
            loss, loss_chamfer, loss_render, loss_kl = (
                vae_reconstruction_loss(model, points, intrinsics, image_size, cfg)
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_chamfer=loss_chamfer.item())
        metric_logger.update(loss_render=loss_render.item())

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
        dim=cfg.vae.model_dim,
        M=cfg.vae.num_latents,
        latent_dim=cfg.vae.latent_dim,
        output_dim=cfg.vae.output_dim,
        N=cfg.vae.point_cloud_size,
        deterministic=not cfg.vae.use_kl,
        decoder_num_queries=cfg.vae.get("decoder_num_queries", None),
        min_scale=cfg.vae.min_scale,
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
        "Criterion: center Chamfer + differentiable Gaussian rendering L1"
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
