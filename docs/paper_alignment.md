# Paper alignment

The implementation is based on **GWM: Towards Scalable Gaussian World Models
for Robotic Manipulation** ([arXiv:2508.17600v2](https://arxiv.org/abs/2508.17600)).
The paper is the source of truth for model and objective choices; the public
code release is marked as work in progress.

## Implemented specification

| Component | Paper specification | Implementation |
| --- | --- | --- |
| Gaussian source | Feed-forward Splatt3R | Frozen `Splatt3rRegressor` |
| VAE input | 2,048 Gaussians | FPS to 2,048 Gaussians |
| VAE latent set | 512 latent points | 512 tokens of dimension 64 |
| VAE posterior | Variational | Gaussian mean/log-variance with KL |
| VAE objective | center Chamfer + rendered RGB L1 | `vae_reconstruction_loss` |
| Dynamics | one-step conditional EDM | `Denoiser` |
| EDM constants | `sigma_data=0.5`, `P_mean=-0.4`, `P_std=1.2` | `configs/world_model/gwm.yaml` |
| DiT position/normalization | RoPE and RMSNorm | `GaussianDiT` |
| Action conditioning | cross-attention keys/values | action tokens in every DiT block |
| Prediction horizon | one step per inference | recursive one-step rollout in `demo.py` |

## Necessary implementation details

The paper does not prescribe the raw neural parameterization of scale,
rotation, and opacity. The decoder therefore maps unconstrained outputs to
physical Gaussian parameters:

- scale: `softplus(raw) + 1e-5`;
- rotation: unit-quaternion normalization;
- opacity: `sigmoid(raw)`.

DROID provides unposed RGB rather than calibrated cameras. Intrinsics for the
rendering objective are estimated from Splatt3R's dense pixel-aligned point map
before FPS. Versioned VAE caches store these intrinsics with their Gaussian
samples so cached training uses the same rendering target.

## Checkpoint boundary

The previous deterministic 64-token VAE omitted the paper's rendering loss and
fed a fabricated square latent grid to a 2D image DiT. Those checkpoints are
not compatible with the paper-aligned architecture. VAE and DiT checkpoints
now include architecture metadata and incompatible or legacy checkpoints fail
with an explicit error rather than loading silently.

## Dataset scope

The paper evaluates MetaWorld, RoboCasa, and Franka PnP. This repository keeps
its existing DROID data pipeline and directory structure. DROID results should
therefore be treated as a new-domain reproduction, not a direct reproduction
of the paper's reported metrics.
