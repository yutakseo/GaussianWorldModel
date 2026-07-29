# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from dataclasses import dataclass
from typing import List, Optional


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@dataclass
class InnerModelConfig:
    # model_type: str
    input_size: int
    patch_size: int
    in_channels: int
    action_dim: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    learn_sigma: bool
    context_length: int
    token_based: bool = False

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ActionEmbedder(nn.Module):
    """
    Embeds continuous actions into vector representations.
    """
    def __init__(self, hidden_size, action_dim, dropout_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Flatten(),   # [B, T, A] -> [B, T * A]
        )
        self.dropout_prob = dropout_prob

    def forward(self, actions, train):
        if train and self.dropout_prob > 0:
            actions = self.token_drop(actions)
        embeddings = self.mlp(actions)
        return embeddings

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        action_dim=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        context_length=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.img_channels = in_channels // (context_length + 1)
        self.out_channels = self.img_channels * 2 if learn_sigma else self.img_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder_cond = TimestepEmbedder(hidden_size)
        # self.a_embedder = ActionEmbedder(hidden_size // (context_length), action_dim, class_dropout_prob)
        self.a_embedder = ActionEmbedder(hidden_size, action_dim, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, noisy_next_obs, c_noise, c_noise_cond, obs, act=None):
        """
        Forward pass of DiT with EDM-style conditioning.
        noisy_next_obs: (N, C, H, W) tensor of noisy next observations
        c_noise: (N,) tensor of noise levels
        c_noise_cond: (N,) tensor of noise level conditioning
        obs: (N, C, H, W) tensor of current observations
        act: (N, A) tensor of actions (optional)
        """
        # print(f"{noisy_next_obs.shape=}, {obs.shape=}")
        # Process the concatenated input
        x = self.x_embedder(torch.cat((obs, noisy_next_obs), dim=1)) + self.pos_embed  # (N, T, D)
        
        # Handle noise level embeddings
        c_noise_emb = self.t_embedder(c_noise)                # (N, D)
        c_noise_cond_emb = (
            self.t_embedder_cond(c_noise_cond)
            if c_noise_cond is not None
            else 0
        )
        
        # Process action embedding
        if act is not None:
            act_emb = self.a_embedder(act, self.training)     # (N, D)
        else:
            act_emb = 0
            
        # Combine conditioning signals
        c = c_noise_emb + c_noise_cond_emb + act_emb          # (N, D)
        
        # backbone
        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            hidden_states.append(x)

        # Final processing
        x = self.final_layer(x, c)                            # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                                # (N, out_channels, H, W)
        return x, hidden_states  # Return both final output and first block features

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, _ = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def _rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rope(x):
    """Apply one-dimensional rotary position embeddings to token heads."""
    sequence_length = x.shape[-2]
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    positions = torch.arange(
        sequence_length, device=x.device, dtype=torch.float32
    )
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32)
        / head_dim
    )
    angles = positions[:, None] * frequencies[None]
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class RotarySelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = _apply_rope(q.transpose(1, 2))
        k = _apply_rope(k.transpose(1, 2))
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(attended)


class GaussianDiTBlock(nn.Module):
    """RMSNorm DiT block with RoPE and action cross-attention."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_self = nn.RMSNorm(hidden_size)
        self.self_attn = RotarySelfAttention(hidden_size, num_heads)
        self.norm_action = nn.RMSNorm(hidden_size)
        self.norm_action_tokens = nn.RMSNorm(hidden_size)
        self.action_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm_mlp = nn.RMSNorm(hidden_size)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 7 * hidden_size, bias=True),
        )

    def forward(self, x, time_condition, action_tokens):
        (
            shift_self,
            scale_self,
            gate_self,
            gate_action,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(time_condition).chunk(7, dim=1)

        self_input = modulate(
            self.norm_self(x), shift_self, scale_self
        )
        x = x + gate_self.unsqueeze(1) * self.self_attn(self_input)

        normalized_action = self.norm_action_tokens(action_tokens)
        action_output, _ = self.action_attn(
            self.norm_action(x),
            normalized_action,
            normalized_action,
            need_weights=False,
        )
        x = x + gate_action.unsqueeze(1) * action_output

        mlp_input = modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class GaussianDiT(nn.Module):
    """Token DiT for the paper's unordered Gaussian VAE latent points."""

    def __init__(
        self,
        num_tokens,
        in_channels,
        action_dim,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        context_length,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_channels = int(in_channels)
        self.context_length = int(context_length)
        self.action_dim = int(action_dim)
        if self.num_tokens <= 0 or self.latent_channels <= 0:
            raise ValueError("num_tokens and in_channels must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        input_channels = self.latent_channels * (context_length + 1)
        self.input_projection = nn.Linear(input_channels, hidden_size)
        self.noise_embedding = TimestepEmbedder(hidden_size)
        self.condition_noise_embedding = TimestepEmbedder(hidden_size)
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            GaussianDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        )
        self.final_norm = nn.RMSNorm(hidden_size)
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2)
        )
        self.output_projection = nn.Linear(hidden_size, self.latent_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_next_latents,
        c_noise,
        c_noise_cond,
        context_latents,
        act=None,
    ):
        """Denoise one Gaussian-latent frame.

        Args:
            noisy_next_latents: ``[B, N, D]`` noisy future VAE latents.
            context_latents: ``[B, T_context, N, D]`` clean history latents.
            c_noise: ``[B]`` EDM noise embedding input.
            act: current transition action, ``[B, A]`` or ``[B, 1, A]``.

        This is deliberately a token API.  Gaussian latents are a set of
        points, not a fabricated ``1 x N`` image grid.
        """
        if noisy_next_latents.ndim != 3:
            raise ValueError(
                "Expected noisy Gaussian latents [B,N,D], got "
                f"{tuple(noisy_next_latents.shape)}"
            )
        if context_latents.ndim != 4:
            raise ValueError(
                "Expected context Gaussian latents [B,T,N,D], got "
                f"{tuple(context_latents.shape)}"
            )
        batch, tokens, channels = noisy_next_latents.shape
        if context_latents.shape[0] != batch:
            raise ValueError("Noisy and context latent batch sizes differ")
        if context_latents.shape[1] != self.context_length:
            raise ValueError(
                f"Expected {self.context_length} context frames, got "
                f"{context_latents.shape[1]}"
            )
        if context_latents.shape[2:] != (tokens, channels):
            raise ValueError(
                "Noisy and context latent token shapes differ: "
                f"{tuple(noisy_next_latents.shape)} vs "
                f"{tuple(context_latents.shape)}"
            )
        if tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} Gaussian tokens, "
                f"got {tokens}"
            )
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, "
                f"got {channels}"
            )
        if c_noise.ndim != 1 or c_noise.shape[0] != batch:
            raise ValueError(
                f"Expected noise levels [B={batch}], got {tuple(c_noise.shape)}"
            )
        if c_noise_cond is not None and (
            c_noise_cond.ndim != 1 or c_noise_cond.shape[0] != batch
        ):
            raise ValueError(
                "Expected conditional noise levels with shape "
                f"[B={batch}], got {tuple(c_noise_cond.shape)}"
            )

        context_tokens = context_latents.transpose(1, 2).reshape(
            batch, tokens, self.context_length * channels
        )
        x = self.input_projection(
            torch.cat((context_tokens, noisy_next_latents), dim=-1)
        )

        time_condition = self.noise_embedding(c_noise)
        if c_noise_cond is not None:
            time_condition = (
                time_condition
                + self.condition_noise_embedding(c_noise_cond)
            )
        if act is None:
            action_tokens = torch.zeros(
                x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype
            )
        else:
            if act.ndim == 2:
                act = act.unsqueeze(1)
            if act.ndim != 3 or act.shape[0] != x.shape[0]:
                raise ValueError(
                    "Expected actions with shape [B,T,A], got "
                    f"{tuple(act.shape)}"
                )
            if act.shape[1] != 1:
                raise ValueError(
                    "GaussianDiT is trained on the current transition "
                    "action only; expected [B,A] or [B,1,A], got "
                    f"{tuple(act.shape)}"
                )
            if act.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected action dimension {self.action_dim}, "
                    f"got {act.shape[-1]}"
                )
            action_tokens = self.action_projection(act)

        hidden_states = []
        for block in self.blocks:
            x = block(x, time_condition, action_tokens)
            hidden_states.append(x)

        shift, scale = self.final_modulation(time_condition).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.output_projection(x)
        return x, hidden_states


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
