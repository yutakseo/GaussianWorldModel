"""
Demo script for Gaussian World Model inference.
"""

import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import cv2
import imageio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf

from gaussianwm.gwm_predictor import GaussianPredictor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset


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


def demo_inference(model, dataset, cfg, num_samples=5, output_dir='demo_outputs'):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    cprint(f"Running inference on {num_samples} samples...", 'blue')
    cprint(f"Results will be saved to: {output_dir.absolute()}", 'blue')
    
    metrics_summary = []
    
    with torch.no_grad():
        # Handle iterable datasets (like DROID)
        dataset_iter = iter(dataset)
        for i in range(num_samples):
            try:
                cprint(f"Processing sample {i+1}/{num_samples}", 'yellow')
                sample = next(dataset_iter)
                obs, action, reward = sample
            except StopIteration:
                cprint(f"Dataset exhausted after {i} samples", 'yellow')
                break

            obs = obs.unsqueeze(0).permute(0, 1, 4, 2, 3).to(model.device)  # [1, T, C, H, W]
            action = action.unsqueeze(0).to(model.device)  # [1, T, A]
            
            obs_gt = torch.cat([obs[:, :-2], obs[:, 1:-1], obs[:, 2:]], dim=2)  # [1, T, C*3, H, W]
            action = action[:, 2:]  # Align actions with frame stacking
            
            def replay_policy(_, t):
                if t < action.shape[1]:
                    return action[:, t].to(model.device)
                return action[:, -1].to(model.device)
            
            start_time = time.time()
            rollout_obs, rollout_actions, rollout_rewards = model.rollout(
                obs_gt[:, 0],  # Initial stacked observation
                replay_policy,
                horizon=obs_gt.shape[1] - 1
            )
            inference_time = time.time() - start_time
            
            cprint(f"Inference time: {inference_time:.3f}s", 'cyan')
            
            obs_mse = ((rollout_obs[:, 1:] - (obs_gt[:, 1:]/255.).to(rollout_obs.device)) ** 2).mean()
            metrics_summary.append({
                'sample_id': i,
                'mse': obs_mse.item(),
                'inference_time': inference_time
            })
            cprint(f"MSE: {obs_mse.item():.6f}", 'magenta')
            
            gt_frames = []
            pred_frames = []
            
            for t in range(1, rollout_obs.shape[1]):
                gt_frame = (obs_gt[0, t, -3:].cpu().numpy().transpose(1,2,0)).astype(np.uint8)
                gt_frame = np.ascontiguousarray(gt_frame)
                pred_frame = (rollout_obs[0, t, -3:].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
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

    if metrics_summary:
        avg_mse = np.mean([m['mse'] for m in metrics_summary])
        avg_time = np.mean([m['inference_time'] for m in metrics_summary])
        
        summary = {
            'num_samples': len(metrics_summary),
            'average_mse': avg_mse,
            'average_inference_time': avg_time,
            'per_sample_metrics': metrics_summary
        }
        import json
        summary_path = output_dir / "metrics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        cprint(f"Average MSE: {avg_mse:.6f}", 'blue')
        cprint(f"Average inference time: {avg_time:.3f}s", 'blue')
        cprint(f"Metrics summary saved to {summary_path.absolute()}", 'green')
    
    cprint(f"Demo completed! Results saved to {output_dir.absolute()}", 'blue')


@hydra.main(version_base=None, config_path="../configs", config_name="train_gwm")
def main(cfg: DictConfig):
    demo_samples = 5
    checkpoint_dir = cfg.output_dir
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = checkpoint_dir / 'demo_outputs'

    # Training saves snapshots as model_<step>.pt / model_latest.pt.  Allow a
    # checkpoint to be supplied through the existing `resume` config option.
    checkpoint_path = (
        Path(cfg.resume).expanduser()
        if cfg.resume
        else checkpoint_dir / "checkpoints" / "model_latest.pt"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(get_original_cwd()) / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"GWM checkpoint not found: {checkpoint_path}\n"
            "The upstream repository does not currently publish pretrained "
            "VAE/DiT weights. Train one with scripts/pretrain/dit.sh, or run "
            "the demo with resume=/absolute/path/to/model_latest.pt."
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    cprint(f"Using device: {device}", 'blue')
    
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    cprint("Creating model...", 'blue')
    model = GaussianPredictor(cfg.world_model).to(device)
    
    cprint(f"Loading specific checkpoint: {checkpoint_path}", 'green')
    suffix = checkpoint_path.stem.replace('model', '')
    model.load_snapshot(checkpoint_path.parent, suffix=suffix)

    cprint("Loading dataset...", 'blue')
    dataset = build_gaussian_splatting_reconstruction_dataset("val", cfg.dataset)
    cprint(f"Dataset size: {len(dataset)}", 'blue')
    
    demo_inference(model, dataset, cfg, num_samples=demo_samples, output_dir=output_dir)

if __name__ == "__main__":
    main()
