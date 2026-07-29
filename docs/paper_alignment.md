# Paper alignment

The implementation is based on **GWM: Towards Scalable Gaussian World Models
for Robotic Manipulation** ([arXiv:2508.17600v2](https://arxiv.org/abs/2508.17600)).
The paper is the source of truth for model and objective choices; the public
code release is marked as work in progress.

## Implemented specification

| Component | Paper specification | Implementation |
| --- | --- | --- |
| Gaussian source | Feed-forward Splatt3R | Frozen `Splatt3rRegressor` |
| VAE input | 2,048 Gaussians | center-only FPS to 2,048 Gaussians |
| VAE latent set | 512 latent points | 512 tokens of dimension 64 |
| VAE encoder | \(L\)-layer cross-attention | four cross-attention/FFN blocks |
| VAE decoder | mirrored self-attention | four self-attention/FFN blocks |
| VAE posterior | Variational | Gaussian mean/log-variance with KL |
| VAE objective | center Chamfer + rendered RGB L1 | `vae_reconstruction_loss` |
| Dynamics | one-step conditional EDM | `Denoiser` |
| EDM constants | `sigma_data=0.5`, `P_mean=-0.4`, `P_std=1.2` | `configs/world_model/gwm.yaml` |
| DiT position/normalization | RoPE and RMSNorm | `GaussianDiT` |
| Action conditioning | cross-attention keys/values | action tokens in every DiT block |
| EDM sampling prior | \(\mathcal{N}(0,\sigma_{\max}^2 I)\) | scaled initial noise in `DiffusionSampler` |
| Temporal setup | sequence 12, context 2, one-step prediction | `configs/dataset/droid.yaml` |
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
samples so cached training uses the same rendering target. This adaptation uses
the paper-supported single-view path with DROID's primary camera.

## Checkpoint boundary

The decoder reconstructs one Gaussian for each of the 512 latent points, as in
the paper's self-attention decoder. It does not introduce a separate set of
2,048 learned decoder queries.

The previous deterministic 64-token VAE omitted the paper's rendering loss and
fed a fabricated square latent grid to a 2D image DiT. Intermediate checkpoints
that used only one encoder cross-attention layer or 2,048 synthetic decoder
queries are also incompatible. VAE and DiT checkpoints include explicit
architecture metadata and incompatible or legacy checkpoints fail rather than
loading silently.

## Dataset scope

The paper evaluates MetaWorld, RoboCasa, and Franka PnP. This repository keeps
its existing DROID data pipeline and directory structure. DROID results should
therefore be treated as a new-domain adaptation, not a reproduction of the
paper's reported metrics. The paper does not specify camera intrinsics,
Gaussian raw-output activations, the VAE latent channel width, encoder depth,
or KL weight; the values documented above are implementation choices and must
not be presented as author-reported hyperparameters.

DROID samples in this repository do not contain the task rewards required by
the paper's model-based RL stage, so `reward.use_reward_model` remains disabled
by default. The optional Conv/ResBlock + LSTM reward model supports both image
and Gaussian-token inputs and is optimized/checkpointed with the dynamics model,
but it requires a dataset with real reward labels. Policy optimization and the
paper's RoboCasa/MetaWorld behavioral-cloning and MBPO loops are not included;
the code here covers world-state encoding, dynamics training, and rollout only.
