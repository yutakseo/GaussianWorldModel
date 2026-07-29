"""Differentiable rendering utilities for Gaussian world states."""

import sys
from pathlib import Path

import torch

SPLATT3R_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "splatt3r"
if str(SPLATT3R_ROOT) not in sys.path:
    sys.path.append(str(SPLATT3R_ROOT))

from src.pixelsplat_src.cuda_splatting import render_cuda
from utils.geometry import build_covariance


def estimate_intrinsics_from_dense_gaussians(gaussians, image_size):
    """Estimate normalized intrinsics from Splatt3R's pixel-aligned output."""
    height, width = image_size
    if gaussians.ndim == 2:
        gaussians = gaussians.unsqueeze(0)
    if gaussians.shape[1] != height * width:
        raise ValueError(
            "Dense Gaussian count must match the source image: "
            f"{gaussians.shape[1]} != {height} * {width}"
        )

    means = gaussians[..., :3].reshape(-1, height, width, 3).float()
    y, x = torch.meshgrid(
        (
            torch.arange(height, device=means.device, dtype=means.dtype) + 0.5
        )
        / height,
        (
            torch.arange(width, device=means.device, dtype=means.dtype) + 0.5
        )
        / width,
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(means.shape[0], -1, -1)
    y = y.unsqueeze(0).expand(means.shape[0], -1, -1)

    z = means[..., 2]
    qx = means[..., 0] / z.clamp_min(1.0e-6)
    qy = means[..., 1] / z.clamp_min(1.0e-6)
    valid = torch.isfinite(means).all(dim=-1) & (z > 1.0e-4)

    def fit_focal(q, pixel):
        mask = valid & torch.isfinite(q) & (q.abs() > 1.0e-4)
        numerator = (q * (pixel - 0.5) * mask).sum(dim=(1, 2))
        denominator = (q.square() * mask).sum(dim=(1, 2)).clamp_min(1.0e-8)
        return (numerator / denominator).abs().clamp(0.25, 4.0)

    intrinsics = torch.eye(
        3, device=means.device, dtype=means.dtype
    ).unsqueeze(0).repeat(means.shape[0], 1, 1)
    intrinsics[:, 0, 0] = fit_focal(qx, x)
    intrinsics[:, 1, 1] = fit_focal(qy, y)
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    return intrinsics


def render_gaussians(gaussians, image_size, intrinsics):
    """Differentiably render physical ``[B, N, 14]`` Gaussian parameters."""
    if gaussians.ndim == 2:
        gaussians = gaussians.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if gaussians.shape[0] != intrinsics.shape[0]:
        raise ValueError(
            "Gaussian and intrinsics batch sizes differ: "
            f"{gaussians.shape[0]} != {intrinsics.shape[0]}"
        )

    decoded = gaussians.float()
    intrinsics = intrinsics.to(device=decoded.device, dtype=decoded.dtype)
    means = decoded[..., 0:3].contiguous()
    scales = decoded[..., 3:6].clamp_min(1.0e-5)
    rotations = decoded[..., 6:10]
    rotations = rotations / rotations.norm(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    sh = decoded[..., 10:13, None].contiguous()
    opacities = decoded[..., 13].clamp(0.0, 1.0)
    covariances = build_covariance(scales, rotations).contiguous()

    batch_size = decoded.shape[0]
    extrinsics = torch.eye(
        4, device=decoded.device, dtype=decoded.dtype
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    near = torch.full(
        (batch_size,), 0.1, device=decoded.device, dtype=decoded.dtype
    )
    far = torch.full(
        (batch_size,), 1000.0, device=decoded.device, dtype=decoded.dtype
    )
    background = torch.zeros(
        batch_size, 3, device=decoded.device, dtype=decoded.dtype
    )

    return render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_size,
        background,
        means,
        covariances,
        sh,
        opacities,
        scale_invariant=True,
        use_sh=True,
    )
