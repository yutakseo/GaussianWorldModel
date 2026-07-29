"""
Demo script for Gaussian World Model inference.
"""

import json
import time
from pathlib import Path

import cv2
import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch3d.loss import chamfer_distance
from termcolor import cprint

from gaussianwm.gwm_predictor import GaussianPredictor
from gaussianwm.processor.regressor import gaussian_feature_to_dim
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset
from gaussianwm.rendering import (
    estimate_intrinsics_from_dense_gaussians,
    render_gaussians,
)
from gaussianwm.util.runtime import resolve_path, seed_everything


def save_rollout_video(
    gt_frames, source_frames, vae_frames, pred_frames, save_path, fps=4
):
    frames = []

    for t in range(len(gt_frames)):
        gt_frame = gt_frames[t]
        source_frame = source_frames[t]
        vae_frame = vae_frames[t]
        pred_frame = pred_frames[t]
        
        # gt_frame and pred_frame are already in HWC format and properly typed
        frame_error = np.abs(gt_frame.astype(float) - pred_frame.astype(float)).astype(np.uint8)
        
        combined_frame = np.concatenate(
            [gt_frame, source_frame, vae_frame, pred_frame, frame_error],
            axis=1,
        )
        frames.append(combined_frame)
    
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    cprint(f"Saved rollout video to {save_path.absolute()}", 'green')


def decode_gaussians(model, latent, decoder_queries=None):
    """Convert one token or legacy grid frame into Gaussian parameters."""
    if model.uses_gaussian_tokens:
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        return model.decode_latents(
            latent, queries=decoder_queries
        ).float()
    if latent.ndim == 3:
        return latent.permute(1, 2, 0).reshape(1, -1, latent.shape[0]).float()
    if latent.ndim == 4:
        return latent.permute(0, 2, 3, 1).reshape(
            latent.shape[0], -1, latent.shape[1]
        ).float()
    return latent.unsqueeze(0).float() if latent.ndim == 2 else latent.float()


def estimate_intrinsics_from_gaussians(gaussians, source_shape):
    return estimate_intrinsics_from_dense_gaussians(
        gaussians, source_shape
    )[0]


def render_gaussian_tensor(gaussians, image_size, intrinsics):
    """Rasterize an already-decoded [N, 14] Gaussian tensor."""
    rendered = render_gaussians(gaussians, image_size, intrinsics)[0]
    return (
        rendered.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)


def gaussian_rasterize_preview(
    model, latent, image_size, intrinsics, decoder_queries=None
):
    """Decode and rasterize Gaussian parameters from the first-camera frame."""
    decoded = decode_gaussians(
        model, latent, decoder_queries=decoder_queries
    )
    expected_dim = sum(
        width
        for name, width in gaussian_feature_to_dim.items()
        if name != "means_in_other_view"
    )
    if decoded.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} decoded Gaussian channels, "
            f"got {decoded.shape[-1]}"
        )

    return render_gaussian_tensor(decoded, image_size, intrinsics)


def gaussian_preview(
    model, latent, image_size, intrinsics=None, decoder_queries=None
):
    """Backward-compatible alias for the real Gaussian rasterization path."""
    if model.args.observation.use_gs:
        if intrinsics is None:
            raise ValueError("Gaussian rendering requires calibrated intrinsics")
        return gaussian_rasterize_preview(
            model,
            latent,
            image_size,
            intrinsics,
            decoder_queries=decoder_queries,
        )
    else:
        rgb = latent[:3]
        rgb = F.interpolate(
            rgb.unsqueeze(0), size=image_size, mode="bilinear",
            align_corners=False,
        )[0]
        return (
            rgb.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
        ).astype(np.uint8)


def demo_inference(model, dataset, cfg, num_samples=5, output_dir='demo_outputs'):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cprint(f"Running inference on {num_samples} samples...", 'blue')
    cprint(f"Results will be saved to: {output_dir.absolute()}", 'blue')
    
    metrics_summary = []
    
    with torch.no_grad():
        # Handle iterable datasets (like DROID)
        dataset_iter = iter(dataset)
        i = 0
        attempts = 0
        max_attempts = max(100, num_samples * cfg.dataset.segment_length * 4)
        while i < num_samples and attempts < max_attempts:
            try:
                sample = next(dataset_iter)
                attempts += 1
                if len(sample) == 4:
                    obs, action, reward, pad_mask = sample
                else:
                    obs, action, reward = sample
                    pad_mask = torch.ones(obs.shape[0], dtype=torch.bool)
            except StopIteration:
                cprint(f"Dataset exhausted after {i} samples", 'yellow')
                break

            # RLDS windows at the beginning of a trajectory are left-padded by
            # repeating the first frame. They are valid for training with a
            # pad mask, but make an uninformative rollout demo.
            frame_change = (
                obs[1:].float() - obs[:-1].float()
            ).abs().mean()
            if frame_change < 0.1:
                continue

            cprint(
                f"Processing sample {i+1}/{num_samples} "
                f"(dataset item {attempts}, frame change {frame_change:.3f})",
                'yellow',
            )

            obs = obs.unsqueeze(0).permute(0, 1, 4, 2, 3).to(model.device)  # [1, T, C, H, W]
            action = action.unsqueeze(0).to(model.device)  # [1, T, A]
            
            context_length = cfg.world_model.context_length
            # action[k] predicts observation[k + 1], so the first rollout
            # target at observation[context_length] uses action[context - 1].
            action = action[:, context_length - 1:obs.shape[1] - 1]
            
            def replay_policy(_, t):
                if t < action.shape[1]:
                    return action[:, t].to(model.device)
                return action[:, -1].to(model.device)
            
            start_time = time.time()
            if cfg.world_model.observation.use_gs:
                # The trained GWM predicts 14-channel Gaussian features, not
                # RGB pixels. Encode the full sequence once for the reference,
                # then autoregressively predict every frame after the context.
                gt_latents, raw_gaussians = model.encode_observations(
                    obs.float() / 255.0, return_gaussians=True
                )
                frames = [gt_latents[:, t] for t in range(context_length)]
                predicted = []
                predicted_gaussians = []
                decoder_query_state = None
                if (
                    model.uses_gaussian_tokens
                    and model.requires_decoder_queries
                ):
                    # The public VAE was trained with input Gaussians as
                    # decoder queries. At rollout time future inputs do not
                    # exist, so seed from the last observation and then feed
                    # back only predictions.
                    decoder_query_state = model.prepare_decoder_queries(
                        raw_gaussians[:, context_length - 1]
                    )
                for t in range(obs.shape[1] - context_length):
                    ctx = torch.stack(frames[-context_length:], dim=1)
                    if model.uses_gaussian_tokens:
                        next_latent = model.sample_next_latents(
                            ctx, replay_policy(None, t)
                        )
                        if model.requires_decoder_queries:
                            next_gaussians = model.decode_latents(
                                next_latent,
                                queries=decoder_query_state,
                            )
                            predicted_gaussians.append(next_gaussians)
                            decoder_query_state = next_gaussians.detach()
                    else:
                        next_latent = model.diffusion_sampler.sample(
                            ctx, replay_policy(None, t)
                        )[0]
                    predicted.append(next_latent)
                    frames.append(next_latent)
                rollout_obs = torch.stack(predicted, dim=1)
                target_obs = gt_latents[:, context_length:]
            else:
                initial_obs = torch.cat(
                    [obs[:, t] for t in range(context_length)], dim=1
                )
                rollout_all, _, _ = model.rollout(
                    initial_obs,
                    replay_policy,
                    horizon=obs.shape[1] - context_length,
                )
                channels = obs.shape[2]
                rollout_obs = rollout_all[:, 1:, -channels:]
                target_obs = obs[:, context_length:].float() / 255.
            inference_time = time.time() - start_time
            
            cprint(f"Inference time: {inference_time:.3f}s", 'cyan')
            
            latent_mse = ((rollout_obs - target_obs) ** 2).mean()
            metrics_summary.append({
                'sample_id': i,
                'latent_mse': latent_mse.item(),
                'inference_time': inference_time
            })
            cprint(f"Latent MSE: {latent_mse.item():.6f}", 'magenta')
            
            gt_frames = []
            source_frames = []
            vae_frames = []
            pred_frames = []

            target_decoder_queries = None
            if (
                model.uses_gaussian_tokens
                and model.requires_decoder_queries
            ):
                target_decoder_queries = model.prepare_decoder_queries(
                    raw_gaussians[:, context_length:]
                )
            decoded_targets = (
                model.decode_latents(
                    target_obs,
                    queries=target_decoder_queries,
                )
                if model.uses_gaussian_tokens
                else None
            )
            decoded_predictions = (
                (
                    torch.stack(predicted_gaussians, dim=1)
                    if model.requires_decoder_queries
                    else model.decode_latents(rollout_obs)
                )
                if model.uses_gaussian_tokens
                else None
            )

            if (
                model.uses_gaussian_tokens
                and model.requires_decoder_queries
            ):
                target_gaussians = target_decoder_queries
                pred_flat = decoded_predictions.flatten(0, 1)
                vae_flat = decoded_targets.flatten(0, 1)
                target_flat = target_gaussians.flatten(0, 1)
                prediction_chamfer, _ = chamfer_distance(
                    pred_flat[..., :3],
                    target_flat[..., :3],
                    batch_reduction="mean",
                    point_reduction="mean",
                )
                vae_chamfer, _ = chamfer_distance(
                    vae_flat[..., :3],
                    target_flat[..., :3],
                    batch_reduction="mean",
                    point_reduction="mean",
                )
                metrics_summary[-1]["prediction_center_chamfer"] = (
                    prediction_chamfer.item()
                )
                metrics_summary[-1]["vae_center_chamfer"] = (
                    vae_chamfer.item()
                )
                cprint(
                    "Prediction center Chamfer: "
                    f"{prediction_chamfer.item():.6f}",
                    "magenta",
                )
                cprint(
                    f"VAE center Chamfer: {vae_chamfer.item():.6f}",
                    "magenta",
                )

            rollout_intrinsics = None
            if cfg.world_model.observation.use_gs:
                # Camera calibration is part of the observed context. Never
                # estimate it from a future target frame during inference.
                rollout_intrinsics = estimate_intrinsics_from_gaussians(
                    raw_gaussians[0, context_length - 1],
                    obs.shape[-2:],
                )

            for t in range(rollout_obs.shape[1]):
                gt_frame = obs[0, context_length + t].cpu().numpy().transpose(1,2,0).astype(np.uint8)
                gt_frame = np.ascontiguousarray(gt_frame)
                if cfg.world_model.observation.use_gs:
                    source_gaussians = raw_gaussians[0, context_length + t]
                    source_frame = render_gaussian_tensor(
                        source_gaussians,
                        gt_frame.shape[:2],
                        rollout_intrinsics,
                    )
                    if model.uses_gaussian_tokens:
                        vae_frame = render_gaussian_tensor(
                            decoded_targets[0, t],
                            gt_frame.shape[:2],
                            rollout_intrinsics,
                        )
                        pred_frame = render_gaussian_tensor(
                            decoded_predictions[0, t],
                            gt_frame.shape[:2],
                            rollout_intrinsics,
                        )
                    else:
                        vae_frame = gaussian_preview(
                            model,
                            target_obs[0, t],
                            gt_frame.shape[:2],
                            rollout_intrinsics,
                        )
                        pred_frame = gaussian_preview(
                            model,
                            rollout_obs[0, t],
                            gt_frame.shape[:2],
                            rollout_intrinsics,
                        )
                else:
                    source_frame = gt_frame
                    vae_frame = gt_frame
                    pred_frame = (
                        rollout_obs[0, t].clamp(0, 1).cpu().numpy().transpose(1,2,0) * 255
                    ).astype(np.uint8)
                pred_frame = np.ascontiguousarray(pred_frame)
                
                gt_frames.append(gt_frame)
                source_frames.append(np.ascontiguousarray(source_frame))
                vae_frames.append(np.ascontiguousarray(vae_frame))
                pred_frames.append(pred_frame)

            source_rendered_mse = np.mean(
                [
                    np.mean(
                        (
                            gt.astype(np.float32) / 255.0
                            - source.astype(np.float32) / 255.0
                        )
                        ** 2
                    )
                    for gt, source in zip(gt_frames, source_frames)
                ]
            )
            vae_rendered_mse = np.mean(
                [
                    np.mean(
                        (
                            gt.astype(np.float32) / 255.0
                            - vae.astype(np.float32) / 255.0
                        )
                        ** 2
                    )
                    for gt, vae in zip(gt_frames, vae_frames)
                ]
            )
            rendered_mse = np.mean(
                [
                    np.mean(
                        (
                            gt.astype(np.float32) / 255.0
                            - pred.astype(np.float32) / 255.0
                        )
                        ** 2
                    )
                    for gt, pred in zip(gt_frames, pred_frames)
                ]
            )
            metrics_summary[-1]["vae_rendered_rgb_mse"] = float(
                vae_rendered_mse
            )
            metrics_summary[-1]["source_rendered_rgb_mse"] = float(
                source_rendered_mse
            )
            metrics_summary[-1]["rendered_rgb_mse"] = float(rendered_mse)
            cprint(
                f"Source Gaussian RGB MSE: {source_rendered_mse:.6f}",
                "magenta",
            )
            cprint(
                f"VAE rendered RGB MSE: {vae_rendered_mse:.6f}", "magenta"
            )
            cprint(f"Rendered RGB MSE: {rendered_mse:.6f}", "magenta")
            
            video_path = output_dir / f"sample_{i:03d}_rollout.gif"
            save_rollout_video(
                gt_frames, source_frames, vae_frames, pred_frames, video_path
            )
            
            frame_dir = output_dir / f"sample_{i:03d}_frames"
            frame_dir.mkdir(exist_ok=True)
            
            for t, (gt_frame, source_frame, vae_frame, pred_frame) in enumerate(
                zip(gt_frames, source_frames, vae_frames, pred_frames)
            ):
                cv2.imwrite(str(frame_dir / f"gt_frame_{t:03d}.png"), cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    str(frame_dir / f"source_frame_{t:03d}.png"),
                    cv2.cvtColor(source_frame, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(frame_dir / f"vae_frame_{t:03d}.png"),
                    cv2.cvtColor(vae_frame, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(str(frame_dir / f"pred_frame_{t:03d}.png"), cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
            
            cprint(f"Saved frames to {frame_dir.absolute()}", 'green')
            i += 1

    if metrics_summary:
        avg_mse = np.mean([m['latent_mse'] for m in metrics_summary])
        avg_time = np.mean([m['inference_time'] for m in metrics_summary])
        avg_rendered_mse = np.mean(
            [m["rendered_rgb_mse"] for m in metrics_summary]
        )
        avg_vae_rendered_mse = np.mean(
            [m["vae_rendered_rgb_mse"] for m in metrics_summary]
        )
        avg_source_rendered_mse = np.mean(
            [m["source_rendered_rgb_mse"] for m in metrics_summary]
        )
        has_center_metrics = all(
            "prediction_center_chamfer" in metric
            for metric in metrics_summary
        )

        summary = {
            'num_samples': len(metrics_summary),
            'average_latent_mse': avg_mse,
            'average_source_rendered_rgb_mse': avg_source_rendered_mse,
            'average_vae_rendered_rgb_mse': avg_vae_rendered_mse,
            'average_rendered_rgb_mse': avg_rendered_mse,
            'average_inference_time': avg_time,
            'per_sample_metrics': metrics_summary
        }
        if has_center_metrics:
            summary["average_prediction_center_chamfer"] = float(
                np.mean(
                    [
                        metric["prediction_center_chamfer"]
                        for metric in metrics_summary
                    ]
                )
            )
            summary["average_vae_center_chamfer"] = float(
                np.mean(
                    [
                        metric["vae_center_chamfer"]
                        for metric in metrics_summary
                    ]
                )
            )
        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        cprint(f"Average latent MSE: {avg_mse:.6f}", 'blue')
        cprint(
            f"Average source Gaussian RGB MSE: {avg_source_rendered_mse:.6f}",
            "blue",
        )
        cprint(
            f"Average VAE rendered RGB MSE: {avg_vae_rendered_mse:.6f}",
            "blue",
        )
        cprint(
            f"Average rendered RGB MSE: {avg_rendered_mse:.6f}", "blue"
        )
        if has_center_metrics:
            cprint(
                "Average prediction center Chamfer: "
                f"{summary['average_prediction_center_chamfer']:.6f}",
                "blue",
            )
            cprint(
                "Average VAE center Chamfer: "
                f"{summary['average_vae_center_chamfer']:.6f}",
                "blue",
            )
        cprint(f"Average inference time: {avg_time:.3f}s", 'blue')
        cprint(f"Metrics summary saved to {summary_path.absolute()}", 'green')
    
    cprint(f"Demo completed! Results saved to {output_dir.absolute()}", 'blue')


@hydra.main(version_base=None, config_path="../configs", config_name="train_gwm")
def main(cfg: DictConfig):
    demo_samples = cfg.demo.num_samples
    checkpoint_dir = Path(cfg.checkpoint_dir)
    output_dir = Path(cfg.demo.output_dir).expanduser()
    output_dir = resolve_path(output_dir, get_original_cwd())

    # Training saves snapshots as model_<step>.pt / model_latest.pt.  Allow a
    # checkpoint to be supplied through the existing `resume` config option.
    checkpoint_path = (
        Path(cfg.resume).expanduser()
        if cfg.resume
        else checkpoint_dir / "model_latest.pt"
    )
    checkpoint_path = resolve_path(checkpoint_path, get_original_cwd())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"GWM checkpoint not found: {checkpoint_path}\n"
            "The upstream repository does not currently publish pretrained "
            "VAE/DiT weights. Train one with scripts/train.sh dit, or run "
            "the demo with resume=/absolute/path/to/model_latest.pt."
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    cprint(f"Using device: {device}", 'blue')
    
    seed_everything(cfg.seed)
    
    cprint("Creating model...", 'blue')
    model = GaussianPredictor(cfg.world_model, device=device).to(device)
    
    cprint(f"Loading specific checkpoint: {checkpoint_path}", 'green')
    suffix = checkpoint_path.stem.removeprefix("model")
    model.load_snapshot(checkpoint_path.parent, suffix=suffix)

    cprint("Loading dataset...", 'blue')
    # The partial DROID validation shard currently exposes only the first
    # (fully left-padded) window of each episode. Use train windows for a
    # qualitative motion sanity check; metrics are not a held-out evaluation.
    dataset = build_gaussian_splatting_reconstruction_dataset("train", cfg.dataset)
    cprint(f"Dataset size: {len(dataset)}", 'blue')
    
    demo_inference(model, dataset, cfg, num_samples=demo_samples, output_dir=output_dir)

if __name__ == "__main__":
    main()
