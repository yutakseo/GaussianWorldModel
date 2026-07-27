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
from termcolor import cprint

from gaussianwm.gwm_predictor import GaussianPredictor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset
from gaussianwm.util.runtime import resolve_path, seed_everything


def save_rollout_video(gt_frames, pred_frames, save_path, fps=4):
    frames = []
    
    for t in range(len(gt_frames)):
        gt_frame = gt_frames[t]
        pred_frame = pred_frames[t]
        
        # gt_frame and pred_frame are already in HWC format and properly typed
        frame_error = np.abs(gt_frame.astype(float) - pred_frame.astype(float)).astype(np.uint8)
        
        combined_frame = np.concatenate([gt_frame, pred_frame, frame_error], axis=1)
        frames.append(combined_frame)
    
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    cprint(f"Saved rollout video to {save_path.absolute()}", 'green')


def gaussian_preview(model, latent, image_size):
    """Build a diagnostic RGB preview; this is not Gaussian rasterization."""
    if model.args.vae.use_vae:
        latent = latent.permute(1, 2, 0).reshape(
            1, model.args.vae.num_latents, -1
        )
        decoded = model.vae.decode(latent)[0]
        side = int(np.sqrt(decoded.shape[0]))
        decoded = decoded[: side * side].reshape(side, side, -1)
        rgb = decoded[..., 10:13].permute(2, 0, 1).unsqueeze(0)
        rgb = F.interpolate(
            rgb,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        rgb = latent[10:13]

    rgb = rgb * 0.28209479177387814 + 0.5
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
                gt_latents = model.encode_observations(obs.float() / 255.)
                frames = [gt_latents[:, t] for t in range(context_length)]
                predicted = []
                for t in range(obs.shape[1] - context_length):
                    ctx = torch.stack(frames[-context_length:], dim=1)
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
            
            obs_mse = ((rollout_obs - target_obs) ** 2).mean()
            metrics_summary.append({
                'sample_id': i,
                'mse': obs_mse.item(),
                'inference_time': inference_time
            })
            cprint(f"MSE: {obs_mse.item():.6f}", 'magenta')
            
            gt_frames = []
            pred_frames = []
            
            for t in range(rollout_obs.shape[1]):
                gt_frame = obs[0, context_length + t].cpu().numpy().transpose(1,2,0).astype(np.uint8)
                gt_frame = np.ascontiguousarray(gt_frame)
                if cfg.world_model.observation.use_gs:
                    pred_frame = gaussian_preview(
                        model, rollout_obs[0, t], gt_frame.shape[:2]
                    )
                else:
                    pred_frame = (
                        rollout_obs[0, t].clamp(0, 1).cpu().numpy().transpose(1,2,0) * 255
                    ).astype(np.uint8)
                pred_frame = np.ascontiguousarray(pred_frame)
                
                gt_frames.append(gt_frame)
                pred_frames.append(pred_frame)
            
            video_path = output_dir / f"sample_{i:03d}_rollout.gif"
            save_rollout_video(gt_frames, pred_frames, video_path)
            
            frame_dir = output_dir / f"sample_{i:03d}_frames"
            frame_dir.mkdir(exist_ok=True)
            
            for t, (gt_frame, pred_frame) in enumerate(zip(gt_frames, pred_frames)):
                cv2.imwrite(str(frame_dir / f"gt_frame_{t:03d}.png"), cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(frame_dir / f"pred_frame_{t:03d}.png"), cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
            
            cprint(f"Saved frames to {frame_dir.absolute()}", 'green')
            i += 1

    if metrics_summary:
        avg_mse = np.mean([m['mse'] for m in metrics_summary])
        avg_time = np.mean([m['inference_time'] for m in metrics_summary])
        
        summary = {
            'num_samples': len(metrics_summary),
            'average_mse': avg_mse,
            'average_inference_time': avg_time,
            'per_sample_metrics': metrics_summary
        }
        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        cprint(f"Average MSE: {avg_mse:.6f}", 'blue')
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
            "VAE/DiT weights. Train one with scripts/pretrain/dit.sh, or run "
            "the demo with resume=/absolute/path/to/model_latest.pt."
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    cprint(f"Using device: {device}", 'blue')
    
    seed_everything(cfg.seed)
    
    cprint("Creating model...", 'blue')
    model = GaussianPredictor(cfg.world_model).to(device)
    
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
