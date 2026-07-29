from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# from torch_cluster import fps
from pytorch3d.ops import sample_farthest_points as fps

from timm.layers import DropPath


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        assert hidden_dim % 6 == 0, "Hidden dim must be divisible by 6 for XYZ positional encoding"
        self.embedding_dim = hidden_dim
        
        # Positional encoding basis for XYZ coordinates
        e = torch.pow(2, torch.arange(hidden_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(hidden_dim//6), torch.zeros(hidden_dim//6)]),  # X
            torch.cat([torch.zeros(hidden_dim//6), e, torch.zeros(hidden_dim//6)]),   # Y
            torch.cat([torch.zeros(hidden_dim//6), torch.zeros(hidden_dim//6), e]),   # Z
        ])
        self.register_buffer('basis', e)  # 3 x (hidden_dim//2)

        # MLP processes: positional_embeddings + original_xyz + other_features
        self.mlp = nn.Linear(hidden_dim + 14, dim)  # 14 = 3(xyz) + 11(other features)

    @staticmethod
    def embed(input, basis):
        # input: [B, N, 3] XYZ coordinates
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings  # [B, N, hidden_dim]
    
    def forward(self, input):
        # input: [B, N, 14] full point cloud features
        xyz = input[..., :3]    # [B, N, 3]
        other_features = input[..., 3:] # [B, N, 11]
        
        # Get positional embeddings for XYZ
        pos_embed = self.embed(xyz, self.basis) # [B, N, hidden_dim], e.g., [64, 64, 48]
        
        combined = torch.cat([pos_embed, xyz, other_features], dim=-1)  # 48+3+11=62
        return self.mlp(combined)


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return self.mean.new_zeros(self.mean.shape[0])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=tuple(range(1, self.mean.ndim)))

    def nll(self, sample, dims=None):
        if self.deterministic:
            return self.mean.new_zeros(self.mean.shape[0])
        if dims is None:
            dims = tuple(range(1, sample.ndim))
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class GaussianOutputTransform(nn.Module):
    """Convert unconstrained decoder channels to physical Gaussians."""

    def __init__(self, min_scale=1.0e-5):
        super().__init__()
        self.min_scale = float(min_scale)

    def forward(self, raw):
        means = raw[..., 0:3]
        scales = F.softplus(raw[..., 3:6]) + self.min_scale

        rotations = raw[..., 6:10]
        rotation_norm = rotations.norm(dim=-1, keepdim=True)
        identity = torch.zeros_like(rotations)
        # Splatt3R and the renderer use scipy's XYZW quaternion order.
        identity[..., 3] = 1.0
        rotations = torch.where(
            rotation_norm > 1.0e-8,
            rotations / rotation_norm.clamp_min(1.0e-8),
            identity,
        )

        sh = raw[..., 10:13]
        opacities = raw[..., 13:14].sigmoid()
        return torch.cat((means, scales, rotations, sh, opacities), dim=-1)


def initialize_gaussian_output(
    layer, initial_scale=1.0e-2, initial_opacity=0.95
):
    """Initialize the Gaussian head in a valid rasterization regime."""
    if not isinstance(layer, nn.Linear) or layer.out_features != 14:
        return

    with torch.no_grad():
        layer.bias.zero_()
        scale = max(float(initial_scale) - 1.0e-5, 1.0e-8)
        layer.bias[3:6] = np.log(np.expm1(scale))
        # Rotation channels are XYZW, so identity is [0, 0, 0, 1].
        layer.bias[9] = 1.0
        opacity = min(max(float(initial_opacity), 1.0e-5), 1.0 - 1.0e-5)
        layer.bias[13] = np.log(opacity / (1.0 - opacity))


def sample_farthest_gaussians(gaussians, num_samples):
    """FPS Gaussian primitives by their 3D centers and gather all channels."""
    if gaussians.ndim != 3 or gaussians.shape[-1] < 3:
        raise ValueError(
            "Expected Gaussian features with shape [B,N,D>=3], got "
            f"{tuple(gaussians.shape)}"
        )
    num_samples = int(num_samples)
    if not 0 < num_samples <= gaussians.shape[1]:
        raise ValueError(
            f"num_samples must be in [1, {gaussians.shape[1]}], "
            f"got {num_samples}"
        )
    centers = gaussians[..., :3].float()
    if not torch.isfinite(centers).all():
        raise ValueError("Gaussian centers contain NaN or infinite values")
    # Keep the Gaussian query set stable across VAE/DiT encode calls.
    _, sampled_indices = fps(
        centers, K=num_samples, random_start_point=False
    )
    sampled = torch.gather(
        gaussians,
        1,
        sampled_indices.unsqueeze(-1).expand(
            -1, -1, gaussians.shape[-1]
        ),
    )
    return sampled, sampled_indices


def _validate_decoder_queries(queries, latents, expected_queries):
    """Validate the public repository's Gaussian decoder-query contract."""
    if queries is None:
        raise ValueError(
            "This VAE uses the public-repository query decoder. Pass "
            "Gaussian queries with shape [B,Q,14] to decode()."
        )
    if queries.ndim != 3 or queries.shape[-1] != 14:
        raise ValueError(
            "Expected decoder Gaussian queries [B,Q,14], got "
            f"{tuple(queries.shape)}"
        )
    if queries.shape[0] != latents.shape[0]:
        raise ValueError(
            "Decoder query and latent batch sizes differ: "
            f"{queries.shape[0]} != {latents.shape[0]}"
        )
    if queries.shape[1] != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} decoder queries, "
            f"got {queries.shape[1]}"
        )
    return queries


class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        decoder_num_queries=None,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        min_scale=1.0e-5,
    ):
        super().__init__()

        self.depth = depth  # control the depth of the decoder

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.decoder_num_queries = (
            int(decoder_num_queries)
            if decoder_num_queries is not None
            else None
        )
        if (
            self.decoder_num_queries is not None
            and self.decoder_num_queries <= 0
        ):
            raise ValueError("decoder_num_queries must be positive")

        # The public implementation uses one encoder cross-attention block;
        # ``depth`` controls the latent self-attention decoder stack.
        self.encoder_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, dim, heads=1, dim_head=dim
                            ),
                            context_dim=dim,
                        ),
                        PreNorm(dim, FeedForward(dim)),
                    ]
                )
                for _ in range(1)
            ]
        )

        self.point_embed = PointEmbed(dim=dim)
        self.encoder_norm = nn.LayerNorm(dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = (
            PreNorm(
                queries_dim,
                Attention(
                    queries_dim,
                    dim,
                    heads=1,
                    dim_head=dim,
                ),
                context_dim=dim,
            )
            if self.decoder_num_queries is not None
            else None
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.decoder_norm = nn.LayerNorm(queries_dim)

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        self.output_transform = (
            GaussianOutputTransform(min_scale=min_scale)
            if output_dim == 14
            else nn.Identity()
        )
        initialize_gaussian_output(self.to_outputs)

    def encode(self, pc):
        """Encode 2,048 Gaussian inputs into the paper's 512 latent queries."""
        B, N, D = pc.shape
        assert N == self.num_inputs, f"Expected {self.num_inputs} point cloud inputs, got {N}"

        # Paper Appendix B.2: input N=2,048 -> FPS queries M=512.
        # The full input point set remains the cross-attention context below.
        sampled_pc, _ = sample_farthest_gaussians(
            pc, self.num_latents
        )

        # print(f"{sampled_pc.shape=}") # [B, K ,3], e.g., [64, 128, 3]

        ###### Embed full 14D features
        sampled_pc_embeddings = self.point_embed(sampled_pc)  # [B, K, C]
        pc_embeddings = self.point_embed(pc)  # [B, N, C]

        x = sampled_pc_embeddings
        for cross_attn, cross_ff in self.encoder_layers:
            x = cross_attn(x, context=pc_embeddings, mask=None) + x
            x = cross_ff(x) + x

        return self.encoder_norm(x)


    def decode(self, x, queries=None):
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if self.decoder_cross_attn is not None:
            queries = _validate_decoder_queries(
                queries, x, self.decoder_num_queries
            )
            query_embeddings = self.point_embed(queries)
            # Match the public repository: output cardinality is determined
            # by Q decoder queries, not by M latent tokens.
            latents = self.decoder_cross_attn(
                query_embeddings, context=x
            )
        else:
            if queries is not None:
                raise ValueError(
                    "The latent-token decoder does not accept external queries"
                )
            latents = x

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.output_transform(
            self.to_outputs(self.decoder_norm(latents))
        )

    def forward(self, pc, queries=None):
        x = self.encode(pc)

        # print(f"{x.shape=}")  # [B, 128, 128]

        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o}

class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        decoder_num_queries = None,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        min_scale=1.0e-5,
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.decoder_num_queries = (
            int(decoder_num_queries)
            if decoder_num_queries is not None
            else None
        )
        if (
            self.decoder_num_queries is not None
            and self.decoder_num_queries <= 0
        ):
            raise ValueError("decoder_num_queries must be positive")

        # Keep the encoder topology of the public repository. Decoder
        # ``depth`` remains independently configurable.
        self.encoder_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, dim, heads=1, dim_head=dim
                            ),
                            context_dim=dim,
                        ),
                        PreNorm(dim, FeedForward(dim)),
                    ]
                )
                for _ in range(1)
            ]
        )

        self.point_embed = PointEmbed(dim=dim)
        self.encoder_norm = nn.LayerNorm(dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = (
            PreNorm(
                queries_dim,
                Attention(
                    queries_dim,
                    dim,
                    heads=1,
                    dim_head=dim,
                ),
                context_dim=dim,
            )
            if self.decoder_num_queries is not None
            else None
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.decoder_norm = nn.LayerNorm(queries_dim)

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        self.output_transform = (
            GaussianOutputTransform(min_scale=min_scale)
            if output_dim == 14
            else nn.Identity()
        )
        initialize_gaussian_output(self.to_outputs)

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        """Encode 2,048 Gaussian inputs into variational 512-token latents."""
        B, N, D = pc.shape
        assert N == self.num_inputs

        # Paper Appendix B.2: input N=2,048 -> FPS queries M=512.
        # ``pc`` remains the complete cross-attention context.
        sampled_pc, _ = sample_farthest_gaussians(
            pc, self.num_latents
        )

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        x = sampled_pc_embeddings
        for cross_attn, cross_ff in self.encoder_layers:
            x = cross_attn(x, context=pc_embeddings, mask=None) + x
            x = cross_ff(x) + x
        x = self.encoder_norm(x)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample() if self.training else posterior.mode()
        kl = posterior.kl()

        return kl, x


    def decode(self, x, queries=None):
        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if self.decoder_cross_attn is not None:
            queries = _validate_decoder_queries(
                queries, x, self.decoder_num_queries
            )
            query_embeddings = self.point_embed(queries)
            latents = self.decoder_cross_attn(
                query_embeddings, context=x
            )
        else:
            if queries is not None:
                raise ValueError(
                    "The latent-token decoder does not accept external queries"
                )
            latents = x

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.output_transform(
            self.to_outputs(self.decoder_norm(latents))
        )

    def forward(self, pc, queries=None):
        kl, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        # return o.squeeze(-1), kl
        return {'logits': o, 'kl': kl}

def create_autoencoder(
        dim=512, M=512, depth=24, latent_dim=64, output_dim=1, N=2048,
        deterministic=False, decoder_num_queries=None, min_scale=1.0e-5,
    ):
    if deterministic:
        model = AutoEncoder(
            depth=depth,
            dim=dim,
            queries_dim=dim,
            output_dim=output_dim,
            num_inputs=N,
            num_latents=M,
            decoder_num_queries=decoder_num_queries,
            heads=8,
            dim_head=64,
            min_scale=min_scale,
        )
    else:
        model = KLAutoEncoder(
            depth=depth,
            dim=dim,
            queries_dim=dim,
            output_dim=output_dim,
            num_inputs=N,
            num_latents=M,
            decoder_num_queries=decoder_num_queries,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
            min_scale=min_scale,
        )
    return model

def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, deterministic=False)
    
def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, deterministic=False)

def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, deterministic=False)

def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, deterministic=False)

def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, deterministic=False)

def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, deterministic=False)

def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, deterministic=False)

def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, deterministic=False)

###
def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, deterministic=True)

def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, deterministic=True)

def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, deterministic=True)

def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, deterministic=True)

###
def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, deterministic=True)

def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, deterministic=True)

def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, deterministic=True)

# low-resolution autoencoder
def ae_d64_m64(N=2048):
    return create_autoencoder(dim=128, M=128, depth=4, output_dim=3, N=N, deterministic=True)

