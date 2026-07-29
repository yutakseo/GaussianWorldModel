from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser, GaussianLatentDenoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    s_cond: float = 0


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample(
        self,
        prev_obs: Tensor,
        prev_act: Optional[Tensor],
        return_reward: bool = True,
    ) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        clean_prev_obs = prev_obs
        sigmas = self.sigmas.to(device=device, dtype=prev_obs.dtype)
        gamma_ = min(
            self.cfg.s_churn / max(len(sigmas) - 1, 1),
            2**0.5 - 1,
        )
        # EDM sampling starts from N(0, sigma_max^2 I), not unit noise.
        x = torch.randn(
            b, c, h, w, device=device, dtype=prev_obs.dtype
        ) * sigmas[0]
        trajectory = [x]
        if self.cfg.s_cond > 0:
            sigma_cond = torch.full(
                (b,),
                fill_value=self.cfg.s_cond,
                device=device,
                dtype=prev_obs.dtype,
            )
            conditioned_prev_obs = self.denoiser.apply_noise(
                clean_prev_obs,
                sigma_cond,
                sigma_offset_noise=self.denoiser.cfg.sigma_offset_noise,
            )
        else:
            sigma_cond = None
            conditioned_prev_obs = clean_prev_obs
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2).clamp_min(0).sqrt()
            denoised = self.denoiser.denoise(
                x, sigma_hat, sigma_cond, conditioned_prev_obs, prev_act
            )
            # reward = self.denoiser.predict_reward(x, sigma, prev_obs, prev_act) if return_reward else None
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(
                    x_2, next_sigma, sigma_cond, conditioned_prev_obs, prev_act
                )
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)

        # if return_reward:
        #     return x, trajectory, reward
        # else:
        return x, trajectory


class GaussianDiffusionSampler:
    """EDM sampler for direct ``[B,T,N,D]`` Gaussian VAE latents."""

    def __init__(
        self,
        denoiser: GaussianLatentDenoiser,
        cfg: DiffusionSamplerConfig,
    ) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(
            cfg.num_steps_denoising,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.rho,
            denoiser.device,
        )

    @torch.no_grad()
    def sample(
        self,
        context_latents: Tensor,
        action: Optional[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        """Sample one future latent frame from clean history latent tokens."""
        if context_latents.ndim != 4:
            raise ValueError(
                "Expected context latents [B,T,N,D], got "
                f"{tuple(context_latents.shape)}"
            )
        batch, context, tokens, channels = context_latents.shape
        if context != self.denoiser.cfg.inner_model.context_length:
            raise ValueError(
                f"Expected {self.denoiser.cfg.inner_model.context_length} "
                f"context frames, got {context}"
            )
        if (tokens, channels) != (
            self.denoiser.inner_model.num_tokens,
            self.denoiser.inner_model.latent_channels,
        ):
            raise ValueError(
                "Context latent shape does not match denoiser: expected "
                f"[B,T,{self.denoiser.inner_model.num_tokens},"
                f"{self.denoiser.inner_model.latent_channels}], got "
                f"{tuple(context_latents.shape)}"
            )

        sigmas = self.sigmas.to(
            device=context_latents.device, dtype=context_latents.dtype
        )
        gamma_base = min(
            self.cfg.s_churn / max(len(sigmas) - 1, 1),
            2**0.5 - 1,
        )
        x = torch.randn(
            batch,
            tokens,
            channels,
            device=context_latents.device,
            dtype=context_latents.dtype,
        ) * sigmas[0]
        trajectory = [x]
        # Conditional context is a fixed observation during one ODE solve.
        # Re-sampling it at every solver step turns a deterministic EDM path
        # into a different stochastic vector field at each evaluation.
        if self.cfg.s_cond > 0:
            sigma_cond = torch.full(
                (batch,),
                self.cfg.s_cond,
                device=context_latents.device,
                dtype=context_latents.dtype,
            )
            conditioned_context = self.denoiser.apply_noise(
                context_latents,
                sigma_cond,
                sigma_offset_noise=self.denoiser.cfg.sigma_offset_noise,
            )
        else:
            sigma_cond = None
            conditioned_context = context_latents

        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            gamma = (
                gamma_base
                if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax
                else 0
            )
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                x = x + torch.randn_like(x) * self.cfg.s_noise * (
                    sigma_hat.square() - sigma.square()
                ).clamp_min(0).sqrt()

            denoised = self.denoiser.denoise(
                x, sigma_hat, sigma_cond, conditioned_context, action
            )
            derivative = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                x = x + derivative * dt
            else:
                candidate = x + derivative * dt
                denoised_2 = self.denoiser.denoise(
                    candidate,
                    next_sigma,
                    sigma_cond,
                    conditioned_context,
                    action,
                )
                derivative_2 = (candidate - denoised_2) / next_sigma
                x = x + (derivative + derivative_2) * (dt / 2)
            trajectory.append(x)

        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0 < sigma_min <= sigma_max:
        raise ValueError(
            "Expected 0 < sigma_min <= sigma_max, got "
            f"{sigma_min} and {sigma_max}"
        )
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))
