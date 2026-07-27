#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

RUN_DATE="${RUN_DATE:-$(date -u +%F)}"
RUN_TIME="${RUN_TIME:-$(date -u +%H-%M-%S)}"
CHECKPOINT="${1:-ckpt/dit/${RUN_DATE}/model_latest.pt}"
OUTPUT_DIR="${2:-outputs/${RUN_DATE}/${RUN_TIME}}"
NUM_SAMPLES="${3:-5}"

export GWM_PATH="${GWM_PATH:-${REPO_DIR}}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VAE_CHECKPOINT="${VAE_CHECKPOINT:-ckpt/vae/${RUN_DATE}/checkpoint-99.pth}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "DiT checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${VAE_CHECKPOINT}" ]]; then
    echo "VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2
    exit 1
fi
if ! [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SAMPLES must be a positive integer: ${NUM_SAMPLES}" >&2
    exit 2
fi

echo "[GaussianWM] Checkpoint: ${CHECKPOINT}"
echo "[GaussianWM] Output:     ${OUTPUT_DIR}"
echo "[GaussianWM] Log:        logs/infer/${RUN_DATE}/${RUN_TIME}.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" -m gaussianwm.demo \
    --config-name train_gwm \
    dataset=droid \
    paths.stage=infer \
    "paths.date=${RUN_DATE}" \
    "paths.time=${RUN_TIME}" \
    "resume=${CHECKPOINT}" \
    "demo.output_dir=${OUTPUT_DIR}" \
    "demo.num_samples=${NUM_SAMPLES}" \
    "world_model.vae.pretrained_path=${VAE_CHECKPOINT}" \
    world_model.vae.use_vae=true \
    world_model.observation.use_gs=true \
    world_model.reward.use_reward_model=false \
    use_wandb=false
