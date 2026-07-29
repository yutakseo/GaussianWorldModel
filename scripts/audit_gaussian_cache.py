#!/usr/bin/env python3
"""Audit every tensor in a legacy Gaussian FPS cache before reuse."""

import argparse
import json
from pathlib import Path

import torch


def _batch_index(path):
    return int(path.stem.removeprefix("batch_"))


def audit_split(root, split, segment_length, point_cloud_size):
    split_dir = root / split
    files = sorted(split_dir.glob("batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch cache files in {split_dir}")
    indices = [_batch_index(path) for path in files]
    if indices != list(range(len(files))):
        raise ValueError(f"Non-contiguous batch indices in {split_dir}")

    summary = {
        "split": split,
        "files": len(files),
        "frames": 0,
        "sequences": 0,
        "point_dtype": None,
        "intrinsics_dtype": None,
        "scale_min": float("inf"),
        "scale_max": float("-inf"),
        "opacity_min": float("inf"),
        "opacity_max": float("-inf"),
        "quaternion_norm_min": float("inf"),
        "quaternion_norm_max": float("-inf"),
    }
    full_frames = None
    for file_number, path in enumerate(files):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("version") != 4:
            raise ValueError(f"Invalid cache version in {path}")
        points = payload.get("points")
        intrinsics = payload.get("intrinsics")
        if (
            not isinstance(points, torch.Tensor)
            or tuple(points.shape[1:]) != (point_cloud_size, 14)
            or not isinstance(intrinsics, torch.Tensor)
            or tuple(intrinsics.shape[1:]) != (3, 3)
            or points.shape[0] != intrinsics.shape[0]
            or points.shape[0] % segment_length
        ):
            raise ValueError(
                f"Invalid tensor shapes in {path}: "
                f"{getattr(points, 'shape', None)}, "
                f"{getattr(intrinsics, 'shape', None)}"
            )
        if file_number == 0:
            full_frames = points.shape[0]
            summary["point_dtype"] = str(points.dtype)
            summary["intrinsics_dtype"] = str(intrinsics.dtype)
        elif file_number + 1 < len(files) and points.shape[0] != full_frames:
            raise ValueError(f"Short non-final cache file: {path}")
        if points.shape[0] > full_frames:
            raise ValueError(f"Oversized cache file: {path}")
        if not torch.isfinite(points).all():
            raise ValueError(f"Non-finite Gaussian value in {path}")
        if not torch.isfinite(intrinsics).all():
            raise ValueError(f"Non-finite intrinsics in {path}")

        scales = points[..., 3:6].float()
        quaternions = points[..., 6:10].float()
        opacities = points[..., 13].float()
        quaternion_norms = quaternions.norm(dim=-1)
        if (scales <= 0).any():
            raise ValueError(f"Non-positive Gaussian scale in {path}")
        if (quaternion_norms <= 1.0e-6).any():
            raise ValueError(f"Degenerate quaternion in {path}")
        if (opacities < 0).any() or (opacities > 1).any():
            raise ValueError(f"Opacity outside [0,1] in {path}")

        summary["frames"] += points.shape[0]
        summary["scale_min"] = min(
            summary["scale_min"], scales.min().item()
        )
        summary["scale_max"] = max(
            summary["scale_max"], scales.max().item()
        )
        summary["opacity_min"] = min(
            summary["opacity_min"], opacities.min().item()
        )
        summary["opacity_max"] = max(
            summary["opacity_max"], opacities.max().item()
        )
        summary["quaternion_norm_min"] = min(
            summary["quaternion_norm_min"],
            quaternion_norms.min().item(),
        )
        summary["quaternion_norm_max"] = max(
            summary["quaternion_norm_max"],
            quaternion_norms.max().item(),
        )
        if (file_number + 1) % 1000 == 0:
            print(
                f"[{split}] audited {file_number + 1}/{len(files)} files",
                flush=True,
            )

    summary["sequences"] = summary["frames"] // segment_length
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--segment-length", type=int, default=12)
    parser.add_argument("--point-cloud-size", type=int, default=2048)
    args = parser.parse_args()

    summaries = [
        audit_split(
            args.cache_root,
            split,
            args.segment_length,
            args.point_cloud_size,
        )
        for split in args.splits
    ]
    print(json.dumps({"status": "ok", "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()
