import os
import math
from pathlib import Path

import time

import torch
import torch.nn as nn

from gaussianwm.diffusion.denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from gaussianwm.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from gaussianwm.diffusion.models import InnerModelConfig
from gaussianwm.reward.reward_model import RewardModel, RewardModelConfig
from termcolor import cprint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.ops import sample_farthest_points as fps


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class GaussianPredictor(nn.Module):
    # def __init__(self, **kwargs) -> None:
    def __init__(self, args) -> None:
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.device = device

        # Initialize diffusion sampler
        denoiser_config = DenoiserConfig(
            inner_model=InnerModelConfig(
                input_size=args.model.input_size,
                patch_size=args.model.patch_size,
                in_channels=args.model.in_channels,
                action_dim=args.action_dim,
                hidden_size=args.model.hidden_size,
                depth=args.model.depth,
                num_heads=args.model.num_heads,
                mlp_ratio=args.model.mlp_ratio,
                class_dropout_prob=args.model.class_dropout_prob,
                learn_sigma=args.model.learn_sigma,
                context_length=args.context_length,
            ),
            sigma_data=args.diffusion.sigma_data,
            sigma_offset_noise=args.diffusion.sigma_offset_noise,
            noise_previous_obs=args.diffusion.noise_previous_obs,
            # Gaussian parameters are continuous physical values. Applying
            # the legacy RGB clamp/8-bit quantization destroys them.
            quantize_output=not args.observation.use_gs,
            autoregressive_training=args.diffusion.get(
                "autoregressive_training", False
            ),
        )
        reward_model_config = RewardModelConfig(
                lstm_dim=args.model.hidden_size,
                img_channels=args.model.in_channels,
                img_size=args.model.input_size,
                cond_channels=args.model.hidden_size,
                depths=[2, 2, 2],
                channels=[32, 32, 32],
                attn_depths=[0, 0, 0],
                action_dim=args.action_dim,
            )

        # Splatt3r and VAE
        self.gaussian_feature_dim = 14
        if args.observation.use_gs:
            from gaussianwm.processor.regressor import Splatt3rRegressor, gaussian_feature_to_dim
            self.splatt3r = Splatt3rRegressor().to(device).eval()
        if args.vae.use_vae:
            from gaussianwm.encoder.models_ae import create_autoencoder
            self.latent_dim = args.vae.latent_dim
            self.num_latents = args.vae.num_latents
            vae_checkpoint = None
            decoder_num_queries = args.vae.get(
                "decoder_num_queries", None
            )
            if args.vae.get("pretrained_path"):
                if not os.path.isfile(args.vae.pretrained_path):
                    raise FileNotFoundError(
                        "Gaussian VAE checkpoint not found: "
                        f"{args.vae.pretrained_path}. Train it first with "
                        "`bash scripts/train.sh vae`; do not reuse a raw-"
                        "Gaussian DiT checkpoint with the latent DiT."
                    )
                vae_checkpoint = torch.load(
                    args.vae.pretrained_path, map_location="cpu"
                )
                checkpoint_args = vae_checkpoint.get("args", {})
                checkpoint_vae = (
                    checkpoint_args.get("vae", {})
                    if hasattr(checkpoint_args, "get")
                    else {}
                )
                checkpoint_queries = (
                    checkpoint_vae.get("decoder_num_queries")
                    if hasattr(checkpoint_vae, "get")
                    else None
                )
                if decoder_num_queries is None:
                    decoder_num_queries = checkpoint_queries
                elif checkpoint_queries != decoder_num_queries:
                    raise ValueError(
                        "Configured VAE decoder_num_queries "
                        f"({decoder_num_queries}) does not match checkpoint "
                        f"metadata ({checkpoint_queries})."
                    )
            self.vae = create_autoencoder(
                depth=args.vae.vae_depth,
                # dim=self.gaussian_feature_dim,
                dim=self.latent_dim,
                M=self.num_latents,
                latent_dim=self.latent_dim,
                output_dim=self.gaussian_feature_dim,
                N=args.observation.point_cloud_size,
                deterministic=not args.vae.use_kl,
                decoder_num_queries=decoder_num_queries,
            ).to(device)
            if vae_checkpoint is not None:
                self.vae.load_state_dict(vae_checkpoint["model"])
                self.vae.requires_grad_(False).eval()
            cprint(f"[VAE] Trainable parameters: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad)/1e6}M", 'yellow')
            cprint(f"[VAE] Total parameters: {sum(p.numel() for p in self.vae.parameters())/1e6}M", 'yellow')

        # Modify denoiser config for latent space if using either component
        if args.observation.use_gs:
            denoiser_config.inner_model.in_channels = 14
            if args.reward.use_reward_model:
                reward_model_config.img_channels = 14
        if args.vae.use_vae:
            denoiser_config.inner_model.in_channels = args.vae.latent_dim
            denoiser_config.inner_model.input_size = args.vae.num_latents
            denoiser_config.inner_model.patch_size = 1
            
            # Pre-compute spatial dimensions for reshaping when using VAE
            self.nh = int(math.sqrt(args.vae.num_latents))
            self.nw = self.nh
            if self.nh * self.nw != args.vae.num_latents:
                raise ValueError(
                    "The public 2D DiT requires a square number of VAE "
                    f"latents, got {args.vae.num_latents}."
                )
            # Update input_size to spatial dimensions
            denoiser_config.inner_model.input_size = self.nh

            if args.reward.use_reward_model:
                reward_model_config.img_size = self.nh
                reward_model_config.img_channels = args.vae.latent_dim

        self.model = Denoiser(denoiser_config).to(device)
        self.model.setup_training(
            SigmaDistributionConfig(
                loc=args.diffusion.sigma_loc,
                scale=args.diffusion.sigma_scale,
                sigma_min=args.diffusion.sigma_min,
                sigma_max=args.diffusion.sigma_max,
            )
        )
        cprint(f"[Model] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6}M", 'yellow')
        cprint(f"[Model] Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6}M", 'yellow')

        sampler_config = DiffusionSamplerConfig(
            num_steps_denoising=args.diffusion.num_steps_denoising,
            sigma_min=args.diffusion.sigma_min,
            sigma_max=args.diffusion.sigma_max,
            rho=args.diffusion.rho,
            order=args.diffusion.order,
        )
        self.diffusion_sampler = DiffusionSampler(self.model, sampler_config)

        # Prepare the latent dynamics optimizer. The pretrained VAE is frozen.
        self.model_optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.optimizer.model_lr)

        if args.reward.use_reward_model:
            self.reward_model = RewardModel(reward_model_config)
            self.reward_model_optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=args.optimizer.reward_model_lr)


    @torch.no_grad()
    def extract_gaussians(self, obs):
        """Run frozen Splatt3r once for a normalized BTCHW RGB sequence."""
        B, T, C, H, W = obs.shape
        obs_flat = obs.reshape(B * T, C, H, W)
        points, _ = self.splatt3r.forward_tensor(obs_flat)
        return points.view(B, T, points.shape[-2], points.shape[-1])

    @torch.no_grad()
    def encode_gaussians(self, points):
        """Encode a [B,T,N,14] Gaussian sequence into the DiT latent grid."""
        B, T, N, D = points.shape
        flat_points = points.reshape(B * T, N, D).float()
        flat_points, _ = fps(
            flat_points, K=self.args.observation.point_cloud_size
        )
        encoded = self.vae.encode(flat_points)
        if isinstance(encoded, tuple):
            _, encoded = encoded
        encoded = encoded.view(B, T, -1, encoded.shape[-1])
        return encoded.permute(0, 1, 3, 2).contiguous().view(
            B, T, encoded.shape[-1], self.nh, self.nw
        )

    def _process_obs(self, obs):
        """Convert normalized BTCHW RGB observations to model embeddings."""
        if not self.args.observation.use_gs:
            return obs

        B, T, _, H, W = obs.shape
        points = self.extract_gaussians(obs)
        if self.args.vae.use_vae:
            return self.encode_gaussians(points)

        return points.permute(0, 1, 3, 2).contiguous().view(
            B, T, points.shape[-1], H, W
        )

    @torch.no_grad()
    def encode_observations(self, obs, return_gaussians=False):
        """Encode RGB sequences into frozen VAE latents for disk caching."""
        if not return_gaussians:
            return self._process_obs(obs).detach()
        if not self.args.observation.use_gs:
            raise ValueError("Raw Gaussians require observation.use_gs=true")
        points = self.extract_gaussians(obs)
        embeddings = (
            self.encode_gaussians(points)
            if self.args.vae.use_vae
            else points.permute(0, 1, 3, 2).contiguous().view(
                obs.shape[0], obs.shape[1], points.shape[-1],
                obs.shape[-2], obs.shape[-1],
            )
        )
        return embeddings.detach(), points.detach()

    def forward(
        self,
        batch,
        update_tokenizer=True,
        update_model=True,
        precomputed_latents=False,
        eval_mode=False,
    ):
        start = time.time()
        metrics = {}
        if update_tokenizer:
            raise ValueError(
                "Joint VAE/DiT training is unsupported. Train the VAE with "
                "gaussianwm.train_vae and keep train.update_tokenizer=false."
            )
        total_loss = torch.zeros((), device=self.device)
        
        if len(batch) == 3:
            obs, action, reward = batch
            pad_mask = None
        else:
            obs, action, reward, pad_mask = batch
            pad_mask = pad_mask.to(self.device)
        obs = obs.to(self.device)
        if not precomputed_latents:
            obs = obs / 255.
        action = action.to(self.device)     # [B, T, A]
        reward = reward.to(self.device)     # [B, T]
        if self.args.symlog:
            reward = symlog(reward)

        # Calculate model loss without optimization
        if update_model:
            self.model.train(not eval_mode)
            if self.args.reward.use_reward_model:
                self.reward_model.train(not eval_mode)
            
            # Process observations to latent space
            latent_embeddings = obs if precomputed_latents else self._process_obs(obs)
            
            # Forward through diffusion model
            diff_loss = self.model(
                latent_embeddings, 
                action,
                batch_mask_padding=pad_mask
            )
            total_loss += diff_loss
            
            reward_loss, reward_pred = 0.0, None
            if self.args.reward.use_reward_model:
                reward_loss, reward_pred = self.reward_model(
                    latent_embeddings[:, self.args.context_length:-1], 
                    action[:, self.args.context_length:-1],
                    latent_embeddings[:, self.args.context_length+1:], 
                    reward[:, self.args.context_length:-1]
                )

            # Calculate total model loss
            if self.args.reward.use_reward_model:
                total_loss += self.args.reward.reward_weight * reward_loss

            metrics.update({
                'diff_loss': diff_loss.item(),
                **({'reward_loss': reward_loss.item(),
                   'model_train/reward_mean': reward[:, self.args.context_length:].mean().item(),
                   'model_train/reward_pred_mean': reward_pred.mean().item()} 
                   if self.args.reward.use_reward_model else {}),
            })

        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics
    
    @torch.no_grad()
    def rollout(self, obs, policy, horizon):
        self.model.eval()
        args = self.args

        x = obs.to(self.device).float()
        B, Ctot, H, W = x.shape

        if args.observation.use_gs:
            ch_per_frame = (args.vae.latent_dim if args.vae.use_vae else 14)
            assert Ctot % args.context_length == 0
            frames_img = [x[:, i*(Ctot//args.context_length):(i+1)*(Ctot//args.context_length)] for i in range(args.context_length)]
            context_imgs = torch.stack(frames_img, dim=1)  # [B, T, C_img, H, W]
            context_latents = self._process_obs(context_imgs / 255.)  # [B, T, Cg, H', W']
            frames = [context_latents[:, i] for i in range(args.context_length)]  # list of [B, Cg, H', W']

            obss = [torch.cat(frames, dim=1)]
            actions, rewards = [], []

            for t in range(horizon):
                ctx = torch.stack(frames[-args.context_length:], dim=1)
                obs_for_policy = torch.cat(frames[-args.context_length:], dim=1)
                action = policy(obs_for_policy, t)
                next_latent = self.diffusion_sampler.sample(ctx, action)[0]

                if args.reward.use_reward_model:
                    prev_lat = ctx[:, -1].unsqueeze(1)
                    rew_pred, _ = self.reward_model.predict_rew(prev_lat, action, next_latent)
                    rew_pred = rew_pred.squeeze(1)
                else:
                    rew_pred = torch.zeros(action.size(0), device=self.device)

                frames.append(next_latent)
                frames.pop(0)

                obss.append(torch.cat(frames[-args.context_length:], dim=1))
                actions.append(action)
                rewards.append(rew_pred)

            actions = [torch.zeros_like(actions[0])] + actions
            rewards = [torch.zeros_like(rewards[0])] + rewards
            if args.symlog:
                rewards = [symexp(r) for r in rewards]

            return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

        frames = [x[:, i*(Ctot//args.context_length):(i+1)*(Ctot//args.context_length)] for i in range(args.context_length)]
        obss, actions, rewards = [torch.cat(frames, dim=1)], [], []

        for t in range(horizon):
            ctx_imgs = torch.stack(frames[-args.context_length:], dim=1)  # [B,T,C,H,W]
            ctx_latents = self._process_obs(ctx_imgs / 255.)
            action = policy(torch.cat(frames[-args.context_length:], dim=1), t)
            next_latent = self.diffusion_sampler.sample(ctx_latents, action)[0]
            next_obs = next_latent

            if args.reward.use_reward_model:
                reward_pred, _ = self.reward_model.predict_rew(ctx_latents[:, -1].unsqueeze(1), action, next_obs)
                reward_pred = reward_pred.squeeze(1)
            else:
                reward_pred = torch.zeros_like(action[:, 0])

            frames.append(next_obs.clamp(0.0, 1.0))
            frames.pop(0)
            obss.append(torch.cat(frames[-args.context_length:], dim=1))
            actions.append(action)
            rewards.append(reward_pred)

        actions = [torch.zeros_like(actions[0])] + actions
        rewards = [torch.zeros_like(rewards[0])] + rewards
        if args.symlog:
            rewards = [symexp(reward) for reward in rewards]

        return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

    def save_snapshot(self, workdir, suffix='', optimizer=None, step=None):
        # Save unwrapped model if using DDP
        model_to_save = self.module if isinstance(self, DDP) else self
        checkpoint = {"model": model_to_save.model.state_dict()}
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if step is not None:
            checkpoint["step"] = step
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = workdir / f"model{suffix}.pt"
        temporary_path = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def load_snapshot(self, workdir, suffix='', optimizer=None):
        # Load works for both DDP and single GPU
        checkpoint_path = Path(workdir) / f"model{suffix}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=f'cuda:{dist.get_rank()}' if dist.is_initialized() else 'cpu',
        )
        # Backward compatibility with checkpoints that contain only model weights.
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("step") if "model" in checkpoint else None
