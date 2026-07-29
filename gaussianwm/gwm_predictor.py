import os
from pathlib import Path

import time

import torch
import torch.nn as nn

from gaussianwm.diffusion.denoiser import (
    Denoiser,
    DenoiserConfig,
    GaussianLatentDenoiser,
    SigmaDistributionConfig,
)
from gaussianwm.diffusion.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
    GaussianDiffusionSampler,
)
from gaussianwm.diffusion.models import InnerModelConfig
from gaussianwm.reward.reward_model import RewardModel, RewardModelConfig
from termcolor import cprint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from gaussianwm.encoder.models_ae import sample_farthest_gaussians


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class GaussianPredictor(nn.Module):
    # def __init__(self, **kwargs) -> None:
    def __init__(self, args, device=None) -> None:
        super().__init__()

        device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.args = args

        uses_gaussian_tokens = (
            args.observation.use_gs and args.vae.use_vae
        )
        self.uses_gaussian_tokens = uses_gaussian_tokens
        if uses_gaussian_tokens and args.reward.use_reward_model:
            raise NotImplementedError(
                "The reward model still expects image grids. Disable "
                "reward.use_reward_model for the Gaussian token pipeline."
            )
        if uses_gaussian_tokens and not args.vae.get("pretrained_path"):
            raise ValueError(
                "The Gaussian token DiT requires a trained VAE checkpoint. "
                "Set world_model.vae.pretrained_path before training or "
                "inference."
            )

        # Initialize the paper's one-step EDM dynamics model.
        denoiser_config = DenoiserConfig(
            inner_model=InnerModelConfig(
                input_size=(
                    args.vae.num_latents
                    if uses_gaussian_tokens
                    else args.model.input_size
                ),
                patch_size=1 if uses_gaussian_tokens else args.model.patch_size,
                in_channels=(
                    args.vae.latent_dim
                    if uses_gaussian_tokens
                    else args.model.in_channels
                ),
                action_dim=args.action_dim,
                hidden_size=args.model.hidden_size,
                depth=args.model.depth,
                num_heads=args.model.num_heads,
                mlp_ratio=args.model.mlp_ratio,
                class_dropout_prob=args.model.class_dropout_prob,
                learn_sigma=args.model.learn_sigma,
                context_length=args.context_length,
                token_based=uses_gaussian_tokens,
            ),
            sigma_data=args.diffusion.sigma_data,
            sigma_offset_noise=args.diffusion.sigma_offset_noise,
            noise_previous_obs=args.diffusion.noise_previous_obs,
            # Gaussian parameters are continuous physical values. Applying
            # the legacy RGB clamp/8-bit quantization destroys them.
            quantize_output=not args.observation.use_gs,
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
            self.splatt3r = (
                Splatt3rRegressor()
                .to(device)
                .requires_grad_(False)
                .eval()
            )
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
                    args.vae.pretrained_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if vae_checkpoint.get("format_version") != 5:
                    raise ValueError(
                        "The VAE checkpoint predates the paper-aligned "
                        "2,048-to-512 token format and "
                        "cannot be reused. "
                        "Retrain the VAE before training or running the DiT."
                    )
                expected_vae = {
                    "spec_version": 5,
                    "representation": "gaussian_vae",
                    "latent_query_source": "fps_2048_to_512",
                    "latent_interface": "token_sequence_v1",
                    "model_dim": int(args.vae.model_dim),
                    "depth": int(args.vae.vae_depth),
                    "num_inputs": int(
                        args.observation.point_cloud_size
                    ),
                    "num_latents": int(args.vae.num_latents),
                    "latent_dim": int(args.vae.latent_dim),
                    "decoder_num_queries": (
                        int(decoder_num_queries)
                        if decoder_num_queries is not None
                        else None
                    ),
                    "use_kl": bool(args.vae.use_kl),
                    "output_dim": self.gaussian_feature_dim,
                    "min_scale": float(args.vae.min_scale),
                }
                actual_vae = vae_checkpoint.get("architecture")
                if actual_vae != expected_vae:
                    raise ValueError(
                        "VAE checkpoint architecture does not match the "
                        "configured paper-aligned model. Expected "
                        f"{expected_vae}, got {actual_vae}."
                    )
            self.vae = create_autoencoder(
                depth=args.vae.vae_depth,
                dim=args.vae.model_dim,
                M=self.num_latents,
                latent_dim=self.latent_dim,
                output_dim=self.gaussian_feature_dim,
                N=args.observation.point_cloud_size,
                deterministic=not args.vae.use_kl,
                decoder_num_queries=decoder_num_queries,
                min_scale=args.vae.min_scale,
            ).to(device)
            if vae_checkpoint is not None:
                self.vae.load_state_dict(vae_checkpoint["model"])
            if uses_gaussian_tokens:
                # The three-stage pipeline trains the VAE separately; DiT
                # optimization must never update or sample from it.
                self.vae.requires_grad_(False).eval()
            cprint(f"[VAE] Trainable parameters: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad)/1e6}M", 'yellow')
            cprint(f"[VAE] Total parameters: {sum(p.numel() for p in self.vae.parameters())/1e6}M", 'yellow')

        # Modify denoiser config for latent space if using either component
        if args.observation.use_gs and not args.vae.use_vae:
            denoiser_config.inner_model.in_channels = 14
            if args.reward.use_reward_model:
                reward_model_config.img_channels = 14
        if uses_gaussian_tokens:
            self.model = GaussianLatentDenoiser(denoiser_config).to(device)
        else:
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
        if uses_gaussian_tokens:
            self.diffusion_sampler = GaussianDiffusionSampler(
                self.model, sampler_config
            )
        else:
            self.diffusion_sampler = DiffusionSampler(self.model, sampler_config)

        optimizer_groups = [
            {
                "params": self.model.parameters(),
                "lr": args.optimizer.model_lr,
            }
        ]
        if args.reward.use_reward_model:
            self.reward_model = RewardModel(reward_model_config).to(device)
            optimizer_groups.append(
                {
                    "params": self.reward_model.parameters(),
                    "lr": args.optimizer.reward_model_lr,
                }
            )
        # The pretrained VAE and Splatt3R are frozen. Dynamics and the
        # optional reward model are checkpointed and optimized together.
        self.model_optimizer = torch.optim.AdamW(optimizer_groups)

    @property
    def device(self):
        return self.model.device

    @torch.no_grad()
    def extract_gaussians(self, obs):
        """Run frozen Splatt3r once for a normalized BTCHW RGB sequence."""
        B, T, C, H, W = obs.shape
        obs_flat = obs.reshape(B * T, C, H, W)
        points, _ = self.splatt3r.forward_tensor(obs_flat)
        return points.view(B, T, points.shape[-2], points.shape[-1])

    @torch.no_grad()
    def encode_gaussians(self, points):
        """VAE encoder stage: Gaussian inputs -> ``[B,T,512,64]`` tokens."""
        if not self.uses_gaussian_tokens:
            raise RuntimeError("Gaussian VAE tokens are disabled")
        if points.ndim != 4 or points.shape[-1] != self.gaussian_feature_dim:
            raise ValueError(
                "Expected Gaussian inputs [B,T,N,14], got "
                f"{tuple(points.shape)}"
            )
        B, T, N, D = points.shape
        flat_points = points.reshape(B * T, N, D).float()
        # DROID/Splatt3R adaptation: establish the paper's 2,048-point VAE
        # input set. The VAE itself then performs its paper-specified
        # 2,048 -> 512 query FPS in ``vae.encode``.
        flat_points, _ = sample_farthest_gaussians(
            flat_points, self.args.observation.point_cloud_size
        )
        encoded = self.vae.encode(flat_points)
        if isinstance(encoded, tuple):
            _, encoded = encoded
        return encoded.view(B, T, encoded.shape[1], encoded.shape[2])

    @torch.no_grad()
    def decode_latents(self, latents):
        """VAE decoder stage: ``[B,(T),512,64]`` tokens -> Gaussians.

        The decoder is intentionally applied only after latent-DiT sampling;
        EDM steps and autoregressive context remain in latent space.
        """
        if not self.uses_gaussian_tokens:
            raise RuntimeError("Gaussian VAE tokens are disabled")
        squeeze_time = latents.ndim == 3
        if squeeze_time:
            latents = latents.unsqueeze(1)
        if latents.ndim != 4:
            raise ValueError(
                "Expected latent tokens [B,N,D] or [B,T,N,D], got "
                f"{tuple(latents.shape)}"
            )
        batch, time_steps, num_tokens, channels = latents.shape
        expected = (self.args.vae.num_latents, self.args.vae.latent_dim)
        if (num_tokens, channels) != expected:
            raise ValueError(
                f"Expected VAE latents [*,*,{expected[0]},{expected[1]}], "
                f"got {tuple(latents.shape)}"
            )
        decoded = self.vae.decode(
            latents.reshape(batch * time_steps, num_tokens, channels)
        ).float()
        decoded = decoded.view(
            batch, time_steps, decoded.shape[1], decoded.shape[2]
        )
        return decoded[:, 0] if squeeze_time else decoded

    @torch.no_grad()
    def sample_next_latents(self, context_latents, action):
        """DiT stage: predict one ``[B,512,64]`` future latent frame."""
        if not self.uses_gaussian_tokens:
            raise RuntimeError("Gaussian VAE tokens are disabled")
        return self.diffusion_sampler.sample(context_latents, action)[0]

    @torch.no_grad()
    def predict_next_gaussians(self, context_latents, action):
        """Run DiT, then VAE-decode its final latent prediction."""
        return self.decode_latents(
            self.sample_next_latents(context_latents, action)
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

        if args.observation.use_gs and self.uses_gaussian_tokens:
            assert Ctot % args.context_length == 0
            frames_img = [
                x[:, i * (Ctot // args.context_length):
                  (i + 1) * (Ctot // args.context_length)]
                for i in range(args.context_length)
            ]
            context_imgs = torch.stack(frames_img, dim=1)
            context_latents = self._process_obs(context_imgs / 255.)
            # Canonical Gaussian-latent frame shape: [B, N=512, D=64].
            frames = [
                context_latents[:, i] for i in range(args.context_length)
            ]

            obss = [torch.stack(frames, dim=1)]
            actions, rewards = [], []

            for t in range(horizon):
                ctx = torch.stack(frames[-args.context_length:], dim=1)
                obs_for_policy = ctx
                action = policy(obs_for_policy, t)
                next_latent = self.sample_next_latents(ctx, action)

                if args.reward.use_reward_model:
                    prev_lat = ctx[:, -1].unsqueeze(1)
                    rew_pred, _ = self.reward_model.predict_rew(prev_lat, action, next_latent)
                    rew_pred = rew_pred.squeeze(1)
                else:
                    rew_pred = torch.zeros(action.size(0), device=self.device)

                frames.append(next_latent)
                frames.pop(0)

                obss.append(torch.stack(frames[-args.context_length:], dim=1))
                actions.append(action)
                rewards.append(rew_pred)

            actions = [torch.zeros_like(actions[0])] + actions
            rewards = [torch.zeros_like(rewards[0])] + rewards
            if args.symlog:
                rewards = [symexp(r) for r in rewards]

            return torch.stack(obss, 1).float(), torch.stack(actions, 1).float(), torch.stack(rewards, 1).float()

        if args.observation.use_gs:
            # Preserve the legacy raw-Gaussian grid rollout for configurations
            # that explicitly disable the VAE.  The paper-aligned default
            # above remains the direct token path.
            assert Ctot % args.context_length == 0
            frames_img = [
                x[:, i * (Ctot // args.context_length):
                  (i + 1) * (Ctot // args.context_length)]
                for i in range(args.context_length)
            ]
            context_imgs = torch.stack(frames_img, dim=1)
            context_gaussians = self._process_obs(context_imgs / 255.0)
            frames = [
                context_gaussians[:, i]
                for i in range(args.context_length)
            ]
            obss, actions, rewards = [torch.cat(frames, dim=1)], [], []

            for t in range(horizon):
                context = torch.stack(
                    frames[-args.context_length:], dim=1
                )
                action = policy(
                    torch.cat(frames[-args.context_length:], dim=1), t
                )
                next_gaussians = self.diffusion_sampler.sample(
                    context, action
                )[0]
                frames.append(next_gaussians)
                frames.pop(0)
                obss.append(torch.cat(frames[-args.context_length:], dim=1))
                actions.append(action)
                rewards.append(torch.zeros_like(action[:, 0]))

            actions = [torch.zeros_like(actions[0])] + actions
            rewards = [torch.zeros_like(rewards[0])] + rewards
            if args.symlog:
                rewards = [symexp(reward) for reward in rewards]
            return (
                torch.stack(obss, 1).float(),
                torch.stack(actions, 1).float(),
                torch.stack(rewards, 1).float(),
            )

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
        checkpoint = {
            "model": model_to_save.model.state_dict(),
            "format_version": 5,
            "architecture": model_to_save._architecture_metadata(),
        }
        if model_to_save.args.reward.use_reward_model:
            checkpoint["reward_model"] = (
                model_to_save.reward_model.state_dict()
            )
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
            weights_only=False,
        )
        architecture = checkpoint.get("architecture")
        expected_architecture = self._architecture_metadata()
        if architecture != expected_architecture:
            raise ValueError(
                "DiT checkpoint architecture does not match the configured "
                "paper-aligned model. Expected "
                f"{expected_architecture}, got {architecture}. "
                "Legacy 2D-grid checkpoints must be retrained."
            )
        # Backward compatibility with checkpoints that contain only model weights.
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        if self.args.reward.use_reward_model:
            reward_state = checkpoint.get("reward_model")
            if reward_state is None:
                raise ValueError(
                    "Reward-enabled checkpoint is missing reward model state"
                )
            self.reward_model.load_state_dict(reward_state)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("step") if "model" in checkpoint else None

    def _architecture_metadata(self):
        return {
            "spec_version": 5,
            "vae_spec_version": (
                5 if self.args.observation.use_gs and self.args.vae.use_vae
                else None
            ),
            "representation": (
                "gaussian_tokens"
                if self.args.observation.use_gs and self.args.vae.use_vae
                else "image"
            ),
            "num_latents": (
                int(self.args.vae.num_latents)
                if self.args.vae.use_vae
                else None
            ),
            "latent_dim": (
                int(self.args.vae.latent_dim)
                if self.args.vae.use_vae
                else None
            ),
            "context_length": int(self.args.context_length),
            "action_dim": int(self.args.action_dim),
            "hidden_size": int(self.args.model.hidden_size),
            "depth": int(self.args.model.depth),
            "num_heads": int(self.args.model.num_heads),
            "mlp_ratio": float(self.args.model.mlp_ratio),
            "sigma_data": float(self.args.diffusion.sigma_data),
            "reward_model": bool(self.args.reward.use_reward_model),
        }
