#!/bin/bash

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=0
GPU_NUMS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

DATASET=droid
DATA_PATH=$GWM_PATH/data/

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$GPU_NUMS \
    --master_port 12345 \
    gaussianwm/train_diffusion.py \
    --config-name train_gwm \
    dataset=$DATASET \
    dataset.data_path=$DATA_PATH \
    world_model.observation.use_gs=true \
    world_model.reward.use_reward_model=false \
    world_model.vae.use_vae=false \
    use_wandb=true
    
