"""Shared runtime helpers for training and inference entry points."""

from pathlib import Path
from typing import Any, MutableSequence

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(base_dir) / resolved
    return resolved.resolve()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def prepare_sequence_batch(
    batch: MutableSequence[torch.Tensor], device: torch.device
) -> MutableSequence[torch.Tensor]:
    """Convert RGB observations from BTHWC to BTCHW and move them to device."""
    if len(batch) not in (3, 4):
        raise ValueError(f"Expected a 3- or 4-item batch, got {len(batch)}")
    observations = batch[0]
    if observations.ndim != 5:
        raise ValueError(
            "Expected observations with shape [B,T,H,W,C], "
            f"got {tuple(observations.shape)}"
        )
    batch[0] = observations.permute(0, 1, 4, 2, 3).contiguous().to(device)
    return batch


def make_data_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
