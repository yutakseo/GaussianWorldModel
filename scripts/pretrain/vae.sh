#!/bin/bash

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0
GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

DATASET=droid

# Run VAE training
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$GPU_NUMS \
    --master_port 12345 \
    gaussianwm/train_vae.py \
    --config-name train_vae \
    dataset=$DATASET \
    train.batch_size=1 \
    dataset.traj_transform_threads=4 \
    dataset.traj_read_threads=4 \
    use_wandb=false
