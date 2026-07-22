# Pretraining

First, download the Droid dataset. You can download the full dataset (1.7TB) using:
```bash
mkdir -p $GWM_PATH/data
gsutil -m cp -r gs://gresearch/robotics/droid $GWM_PATH/datasets
```
If you'd like to download an example version of the dataset with 100 episodes first (2GB), run:

```bash
mkdir -p $GWM_PATH/data
gsutil -m cp -r gs://gresearch/robotics/droid_100 $GWM_PATH/datasets
```

Train the 3D variational autoencoder:

```bash
bash scripts/pretrain/vae.sh
```

Train the diffusion model:

```bash
bash scripts/pretrain/dit.sh
```