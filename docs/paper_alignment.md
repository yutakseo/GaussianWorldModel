# Paper and public-code alignment

This implementation combines the architecture exposed by the
[public GaussianWM repository](https://github.com/Gaussian-World-Model/gaussianwm)
with the objectives and 2,048-to-512 setting in
[GWM: Towards Scalable Gaussian World Models for Robotic Manipulation](https://arxiv.org/abs/2508.17600).
The repository is explicitly marked work in progress and its VAE training and
world-model inference paths do not currently share a complete decoder
interface. The exact boundary used here is therefore recorded below.

## Source priority

When the two sources differ, this project uses:

1. the public repository for the VAE module topology and tensor operations;
2. the paper for the 2,048/512 cardinalities, variational posterior,
   reconstruction objective, and EDM hyperparameters;
3. an explicitly causal implementation for information the released inference
   code omits.

This is a runnable, source-traceable interpretation. It must not be described
as the authors' exact unpublished training setup.

## Implemented specification

| Component | Source | Implementation |
| --- | --- | --- |
| Gaussian source | Paper and code | Frozen Splatt3R |
| VAE input | Paper Appendix B.2 | 2,048 Gaussians, sampled from the dense DROID point map |
| Encoder queries | Paper and code | center-only FPS, 2,048 to 512 |
| Encoder | Public code | one query-to-input cross-attention/FFN block |
| Posterior | Paper | 512 mean/log-variance tokens, reparameterized to `[512,64]` |
| Decoder trunk | Public code | four latent self-attention/FFN blocks |
| Decoder output | Public code | 2,048 input-Gaussian queries cross-attend to 512 latent tokens |
| VAE objective | Paper Eq. (4) | center Chamfer + rendered RGB L1 + weighted KL |
| Dynamics | Paper | one-step conditional EDM in latent-token space |
| EDM constants | Paper Appendix B.1 | `sigma_data=0.5`, `P_mean=-0.4`, `P_std=1.2` |
| DiT | Paper | RoPE self-attention, RMSNorm, time AdaLN, action cross-attention |
| Temporal setup | Paper | sequence 12, context 2, one-step target |
| Long rollout | Required causal completion | recursively sample one latent and decode one Gaussian frame |

The public VAE entrypoint passes `dim=latent_dim`; both are 64 here. Decoder
depth remains four. The paper does not report these two implementation values.

## Exact training flow

For every RGB frame:

```text
RGB [3,H,W]
  -> frozen Splatt3R
  -> dense Gaussians [H*W,14]
  -> center FPS
  -> VAE input G_t [2048,14]

G_t
  -> center FPS queries [512,14]
  -> one cross-attention encoder block over G_t
  -> mean/logvar [512,64]
  -> reparameterization
  -> z_t [512,64]

z_t
  -> four self-attention decoder blocks
G_t used again as decoder queries [2048,14]
  -> query-to-latent cross-attention
  -> reconstructed Gaussians [2048,14]
```

This matches the released standalone VAE call `model(points, points)`: the
second `points` argument determines decoder output cardinality. The training
loss follows the paper rather than the released scaffold's parameter-wise MSE:

```text
L_VAE = Chamfer(center(G_hat), center(G))
      + L1(render(G_hat), render(G))
      + 1e-3 * KL(q(z|G) || N(0,I))
```

After VAE training, Splatt3R and the VAE are frozen. For every valid transition,
the DiT receives two clean context latent frames, one noised next-frame latent,
the EDM noise embedding, and the action aligned with that transition. It
predicts the EDM preconditioned target. Teacher forcing is used across all
one-step transitions in a 12-frame training window.

Two released-predictor statements are treated as execution defects rather than
model choices. Its KL encoder path selects tuple element zero, which is the
batch KL value rather than latent tokens, and it reshapes `num_latents` through
`int(sqrt(num_latents))`. The latter drops tokens when `num_latents=512`
because 512 is not a square. Here tuple element one is used and the DiT keeps a
direct `[B,T,512,64]` token tensor without fabricating an image grid.

## Exact inference flow

The released query-based KL decoder cannot reconstruct from `z` alone, yet the
released predictor calls `decode(z)` without its required second argument.
Future target Gaussians are unavailable in a real rollout. This implementation
therefore makes the missing causal contract explicit:

1. encode the two observed context frames to `[2,512,64]`;
2. sample the next `[512,64]` latent with EDM conditioned on the current action;
3. for the first prediction, use the last observed 2,048 Gaussians as decoder
   queries;
4. decode `[512,64] + [2048,14] -> [2048,14]`;
5. append the sampled latent to DiT context and use the decoded prediction as
   the next frame's 2,048 decoder queries;
6. repeat for the requested horizon.

No future frame, future Gaussian, or future-derived camera intrinsics enter the
prediction path. Ground-truth future queries are used only to measure the
single-frame VAE reconstruction baseline. Rollout rendering uses intrinsics
estimated once from the last observed context frame.

## Important limitation of the public decoder

The latent is not a self-contained tokenization under the public query
decoder: reconstruction also depends on a full 14-channel Gaussian query set.
During VAE training that query set is the target itself, whereas causal rollout
uses the previous frame. This creates a train/inference conditioning shift and
allows a shortcut through query features. The public release neither specifies
nor implements a different future-query generator.

Set `decoder_num_queries: null` to use the alternative latent-only decoder,
which emits 512 Gaussians and needs no external queries. That path is closer to
the paper's displayed self-attention decoder equation, but it is not the
standalone VAE training path in the public repository. The default configuration
now follows the public query decoder.

## Checkpoints and evaluation

Version-6 VAE checkpoints record the one-block encoder, 512-token posterior,
and 2,048-query decoder. Version-6 DiT checkpoints also record the causal
`previous_frame` query strategy and a serialization-independent SHA-256
identity of the frozen VAE weights used to create its latent training targets.
This prevents a DiT
from silently running with a same-shaped but semantically different VAE.
Older or mismatched checkpoints fail fast because their weights or runtime
interface are incompatible.

Latent MSE is retained only as a diagnostic because FPS latent points are a
set, not guaranteed correspondences across time. Rollout evaluation also
reports center Chamfer distance and rendered RGB MSE.

The paper evaluates MetaWorld, RoboCasa, and Franka PnP. This repository uses
DROID and estimated single-camera intrinsics, so its outputs are a DROID
adaptation rather than a reproduction of the paper's benchmark numbers.
Reward learning, behavior cloning, and model-based policy optimization are not
implemented because the current DROID pipeline supplies dummy rewards.
