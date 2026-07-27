# Pretraining

First, download the Droid dataset. You can download the full dataset (1.7TB) using:
```bash
mkdir -p $GWM_PATH/data
gsutil -m cp -r gs://gresearch/robotics/droid $GWM_PATH/data
```
If you'd like to download an example version of the dataset with 100 episodes first (2GB), run:

```bash
mkdir -p $GWM_PATH/data
gsutil -m cp -r gs://gresearch/robotics/droid_100 $GWM_PATH/data
```

From the repository root, train both stages in order:

```bash
./scripts/train.sh all
```

The stages can also be run separately. Train the 3D variational autoencoder:

```bash
./scripts/train.sh vae
```

This writes weights to `ckpt/vae/YYYY-MM-DD`, logs to
`logs/vae/YYYY-MM-DD/HH-MM-SS.log`, and reusable Gaussian features to
`cache/vae/YYYY-MM-DD`. Older VAE checkpoints used inconsistent Splatt3r
normalization and a decoder-query path that is unavailable at inference; do
not reuse them.

Train the diffusion model:

```bash
./scripts/train.sh dit
```

The diffusion stage uses `ckpt/vae/YYYY-MM-DD/checkpoint-99.pth`, writes its
weights to `ckpt/dit/YYYY-MM-DD`, and writes logs to
`logs/dit/YYYY-MM-DD/HH-MM-SS.log`. Raw-Gaussian DiT checkpoints are not
architecture-compatible with this latent DiT.

Run a qualitative rollout after training:

```bash
./scripts/infer.sh \
  ckpt/dit/2026-07-27/model_latest.pt \
  outputs/2026-07-27/rollout-01 \
  5
```

The positional arguments are the DiT checkpoint, output directory, and number
of samples. Set `CUDA_VISIBLE_DEVICES`, `MASTER_PORT`, `PYTHON_BIN`, or
`VAE_CHECKPOINT` in the environment when overriding their defaults.
`RUN_DATE=YYYY-MM-DD` selects a particular dated VAE/DiT directory; it defaults
to the current UTC date.

The top-level directories have one responsibility:

- `data`: datasets
- `ckpt`: VAE and DiT weights, grouped by date
- `cache`: reusable preprocessed Gaussian features
- `logs`: dated training and inference logs named `HH-MM-SS.log`
- `outputs`: inference images, GIFs, and metrics only
