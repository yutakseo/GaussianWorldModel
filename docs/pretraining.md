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

The stages can also be run separately. Train the dense 3D variational
autoencoder:

```bash
./scripts/train.sh vae
```

This writes weights to `ckpt/vae/YYYY-MM-DD`, logs and the live loss graph to
`logs/train/vae/YYYY-MM-DD`, and reusable Gaussian features to
`cache/vae/YYYY-MM-DD`. The encoder keeps 64 latent tokens while the decoder
uses 2,048 learned queries to reconstruct the complete Gaussian set.

Train the diffusion model:

```bash
./scripts/train.sh dit
```

The diffusion stage uses `ckpt/vae/YYYY-MM-DD/checkpoint-latest.pth`, writes its
weights to `ckpt/dit/YYYY-MM-DD`, and writes logs and the live loss graph to
`logs/train/dit/YYYY-MM-DD`. Raw-Gaussian DiT checkpoints are not architecture-
compatible with this latent DiT.

After stable one-step training, continue with autoregressive rollout
fine-tuning:

```bash
VAE_CHECKPOINT=ckpt/vae/2026-07-28/checkpoint-latest.pth \
./scripts/train.sh rollout ckpt/dit/2026-07-28/model_100000.pt 105000
```

`train.sh` and `infer.sh` read the VAE checkpoint metadata and automatically
select the dense or legacy decoder shape.

DiT atomically refreshes `model_latest.pt`, including at a non-periodic final
step. VAE likewise writes numbered checkpoints and atomically refreshes
`checkpoint-latest.pth`. Both include optimizer state and the exact resume
position.

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
- `logs/train/vae`: dated VAE logs, metrics, and loss graphs
- `logs/train/dit`: dated DiT logs, metrics, and loss graphs
- `logs/infer`: dated inference logs
- `outputs`: inference images, GIFs, and metrics only
