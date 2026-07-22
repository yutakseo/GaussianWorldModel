from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# from data import Batch
from .models import DiT, InnerModelConfig
from typing import Tuple, Dict, Any


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor
    c_noise_cond: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float
    noise_previous_obs: bool
    upsampling_factor: Optional[int] = None


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_upsampler = cfg.upsampling_factor is not None
        cfg.inner_model.is_upsampler = self.is_upsampler
        self.inner_model = DiT(
            input_size=cfg.inner_model.input_size,
            patch_size=cfg.inner_model.patch_size,
            in_channels=cfg.inner_model.in_channels * (cfg.inner_model.context_length + 1), # horizon=1
            action_dim=cfg.inner_model.action_dim,
            hidden_size=cfg.inner_model.hidden_size,
            depth=cfg.inner_model.depth,
            num_heads=cfg.inner_model.num_heads,
            mlp_ratio=cfg.inner_model.mlp_ratio,
            class_dropout_prob=cfg.inner_model.class_dropout_prob,
            learn_sigma=cfg.inner_model.learn_sigma,
            context_length=cfg.inner_model.context_length,
        )
        self.sample_sigma_training = None
        # self.reward_head = nn.Sequential(
        #     nn.LayerNorm(cfg.inner_model.hidden_size),
        #     nn.Linear(cfg.inner_model.hidden_size, 1)
        # )

    @property
    def device(self) -> torch.device:
        device = next(self.inner_model.parameters()).device
        # if torch.cuda.is_available():
        #     device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # else:
        #     device = torch.device("cpu")
        return device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape 
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor, sigma_cond: Optional[Tensor]) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        c_noise_cond = sigma_cond.log() / 4 if sigma_cond is not None else torch.zeros_like(c_noise)
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise, c_noise_cond), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Optional[Tensor], cs: Conditioners) -> Tuple[Tensor, Tensor]:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        output, hidden_states = self.inner_model(rescaled_noise, cs.c_noise, cs.c_noise_cond, rescaled_obs, act)
        return output, hidden_states

    def forward(self, batch_obs, batch_action, batch_mask_padding=None) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """
        Args:
            batch_obs: (B, T, C, H, W)
            batch_action: (B, T, A)
            batch_mask_padding: (B, T)
        """
        
        b, t, c, h, w = batch_obs.size()
        H, W = (self.cfg.upsampling_factor * h, self.cfg.upsampling_factor * w) if self.is_upsampler else (h, w)
        n = self.cfg.inner_model.context_length
        seq_length = t - n  # t = n + 1 + num_autoregressive_steps

        # if self.is_upsampler:
        #     all_obs = torch.stack([x["full_res"] for x in batch.info]).to(self.device)
        #     low_res = F.interpolate(batch.obs.reshape(b * t, c, h, w), scale_factor=self.cfg.upsampling_factor, mode="bicubic").reshape(b, t, c, H, W)
        #     assert all_obs.shape == low_res.shape   # [B=4, T=3, C=3, H=512, W=512]
        # else:
        all_obs = batch_obs.clone()

        loss = 0
        reward_preds = []   # [B, seq_length]
        
        for i in range(seq_length):
            prev_obs = all_obs[:, i : n + i].reshape(b, n * c, H, W)
            # prev_act = None if self.is_upsampler else batch_action[:, i : n + i]
            
            # only use the last action
            prev_act = batch_action[:, n + i - 1, :].unsqueeze(1)

            obs = all_obs[:, n + i]
            if batch_mask_padding is not None:
                mask = batch_mask_padding[:, n + i] # (B,)
            else:
                mask = None

            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, self.device)
                prev_obs = self.apply_noise(prev_obs, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None

            # if self.is_upsampler:
            #     prev_obs = torch.cat((prev_obs, low_res[:, n + i]), dim=1)  # [B=4, C=6, H=512, W=512]

            sigma = self.sample_sigma_training(b, self.device)
            noisy_obs = self.apply_noise(obs, sigma, self.cfg.sigma_offset_noise)   # upsampler: [B=4, C=3, H=512, W=512]

            cs = self.compute_conditioners(sigma, sigma_cond)

            # print(f"{noisy_obs.shape=}, {prev_obs.shape=}, {prev_act.shape=}")
            model_output, hidden_states = self.compute_model_output(noisy_obs, prev_obs, prev_act, cs)

            target = (obs - cs.c_skip * noisy_obs) / cs.c_out

            if mask is not None:
                # Apply mask as weights (expanding dimensions for broadcasting)
                mask_expanded = mask.view(b, 1, 1, 1)
                weighted_diff = ((model_output - target) ** 2) * mask_expanded
                loss += weighted_diff.sum() / (mask.sum() * c * h * w + 1e-8)
            else:
                loss += F.mse_loss(model_output, target)

            # with torch.no_grad():
            denoised = self.wrap_model_output(noisy_obs, model_output, cs)
            # Remove the line to conduct teacher-forcing
            all_obs[:, n + i] = denoised
            # reward_preds.append(self.reward_head(hidden_states[-1].mean(dim=1)))

        loss /= seq_length
        
        # reward_pred = torch.stack(reward_preds, dim=1)  # [B, seq_length, 1]

        return loss #reward_pred

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, sigma_cond: Optional[Tensor], 
                    obs: Tensor, act: Optional[Tensor]) -> Tensor:
        """Denoising without reward prediction"""
        cs = self.compute_conditioners(sigma, sigma_cond)
        model_output, _ = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised
    
    # @torch.no_grad()
    # def predict_reward(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, 
    #                  act: Optional[Tensor]) -> Tensor:
    #     """Predict reward using model embeddings without returning denoised observation"""
    #     cs = self.compute_conditioners(sigma, None)
    #     _, hidden_states = self.compute_model_output(noisy_next_obs, obs, act, cs)
    #     reward_pred = self.reward_head(hidden_states[-1].mean(dim=1))   # .squeeze(-1)
    #     return reward_pred
