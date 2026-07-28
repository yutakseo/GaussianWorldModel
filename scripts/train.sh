#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${MODE}" in
    vae|dit|all|rollout) ;;
    *)
        echo "Usage:" >&2
        echo "  $0 [vae|dit|all] [Hydra overrides ...]" >&2
        echo "  $0 rollout CHECKPOINT FINAL_STEP [Hydra overrides ...]" >&2
        exit 2
        ;;
esac

export GWM_PATH="${GWM_PATH:-${REPO_DIR}}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-12345}"
RUN_DATE="${RUN_DATE:-$(date -u +%F)}"
RUN_TIME="${RUN_TIME:-$(date -u +%H-%M-%S)}"
IFS=',' read -r -a GPU_LIST <<< "${CUDA_DEVICES}"
GPU_COUNT="${#GPU_LIST[@]}"
EXTRA_OVERRIDES=("$@")

run_distributed() {
    local stage="$1"
    local entrypoint="$2"
    shift 2
    local log_dir="logs/train/${stage}/${RUN_DATE}"
    local log_file="${log_dir}/${RUN_TIME}.log"
    mkdir -p "${log_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nproc_per_node="${GPU_COUNT}" \
        --master_port="${MASTER_PORT}" \
        --module "${entrypoint}" "$@" 2>&1 | tee -a "${log_file}"
}

train_vae() {
    local vae_overrides=()
    if [[ -n "${VAE_RESUME:-}" ]]; then
        vae_overrides+=("resume=${VAE_RESUME}")
    fi
    echo "[GaussianWM] VAE weights -> ckpt/vae/${RUN_DATE}"
    echo "[GaussianWM] VAE log     -> logs/train/vae/${RUN_DATE}/${RUN_TIME}.log"
    run_distributed vae gaussianwm.train_vae \
        --config-name train_vae \
        dataset=droid \
        "paths.date=${RUN_DATE}" \
        "paths.time=${RUN_TIME}" \
        vae.decoder_num_queries=2048 \
        use_wandb=false \
        "${vae_overrides[@]}" \
        "${EXTRA_OVERRIDES[@]}"
}

train_dit() {
    local default_vae_checkpoint="ckpt/vae/${RUN_DATE}/checkpoint-latest.pth"
    if [[ ! -f "${default_vae_checkpoint}" ]]; then
        default_vae_checkpoint="ckpt/vae/${RUN_DATE}/checkpoint-99.pth"
    fi
    local vae_checkpoint="${VAE_CHECKPOINT:-${default_vae_checkpoint}}"
    if [[ ! -f "${vae_checkpoint}" ]]; then
        echo "VAE checkpoint not found: ${vae_checkpoint}" >&2
        echo "Run '$0 vae' first or set VAE_CHECKPOINT." >&2
        exit 1
    fi
    echo "[GaussianWM] DiT weights -> ckpt/dit/${RUN_DATE}"
    echo "[GaussianWM] DiT log     -> logs/train/dit/${RUN_DATE}/${RUN_TIME}.log"
    run_distributed dit gaussianwm.train_diffusion \
        --config-name train_gwm \
        dataset=droid \
        "paths.date=${RUN_DATE}" \
        "paths.time=${RUN_TIME}" \
        "world_model.vae.pretrained_path=${vae_checkpoint}" \
        world_model.vae.use_vae=true \
        world_model.observation.use_gs=true \
        world_model.reward.use_reward_model=false \
        use_wandb=false \
        "${EXTRA_OVERRIDES[@]}"
}

if [[ "${MODE}" == "rollout" ]]; then
    if [[ ${#EXTRA_OVERRIDES[@]} -lt 2 ]]; then
        echo "Usage: $0 rollout CHECKPOINT FINAL_STEP [Hydra overrides ...]" >&2
        exit 2
    fi
    DIT_CHECKPOINT="${EXTRA_OVERRIDES[0]}"
    FINAL_STEP="${EXTRA_OVERRIDES[1]}"
    EXTRA_OVERRIDES=("${EXTRA_OVERRIDES[@]:2}")
    if [[ ! -f "${DIT_CHECKPOINT}" ]]; then
        echo "DiT checkpoint not found: ${DIT_CHECKPOINT}" >&2
        exit 1
    fi
    if ! [[ "${FINAL_STEP}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FINAL_STEP must be a positive integer: ${FINAL_STEP}" >&2
        exit 2
    fi
    EXTRA_OVERRIDES+=(
        "resume=${DIT_CHECKPOINT}"
        "train.max_steps=${FINAL_STEP}"
        "world_model.diffusion.autoregressive_training=true"
    )
    train_dit
    exit 0
fi

if [[ "${MODE}" == "vae" || "${MODE}" == "all" ]]; then
    train_vae
fi
if [[ "${MODE}" == "dit" || "${MODE}" == "all" ]]; then
    train_dit
fi
