#!/bin/bash

set -e

if [[ ! -f /var/cache/gaussianwm/ready ]]; then
    echo "Container setup is still running. Wait for '[gaussianwm-setup] Setup complete' in: docker compose logs -f gaussianwm" >&2
    exit 1
fi

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0
GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

DATASET=droid

# Run VAE training
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "${PYTHON_BIN:-python3}" -m torch.distributed.run \
    --nproc_per_node=$GPU_NUMS \
    --master_port 12345 \
    gaussianwm/train_vae.py \
    --config-name train_vae \
    dataset=$DATASET \
    train.batch_size=16 \
    dataset.traj_transform_threads=10 \
    dataset.traj_read_threads=10 \
    use_wandb=false
