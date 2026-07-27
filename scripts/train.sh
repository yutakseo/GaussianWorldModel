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
    vae|dit|all) ;;
    *)
        echo "Usage: $0 [vae|dit|all] [Hydra overrides ...]" >&2
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
    local log_dir="logs/${stage}/${RUN_DATE}"
    local log_file="${log_dir}/${RUN_TIME}.log"
    mkdir -p "${log_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nproc_per_node="${GPU_COUNT}" \
        --master_port="${MASTER_PORT}" \
        --module "${entrypoint}" "$@" 2>&1 | tee -a "${log_file}"
}

train_vae() {
    echo "[GaussianWM] VAE weights -> ckpt/vae/${RUN_DATE}"
    echo "[GaussianWM] VAE log     -> logs/vae/${RUN_DATE}/${RUN_TIME}.log"
    run_distributed vae gaussianwm.train_vae \
        --config-name train_vae \
        dataset=droid \
        "paths.date=${RUN_DATE}" \
        "paths.time=${RUN_TIME}" \
        use_wandb=false \
        "${EXTRA_OVERRIDES[@]}"
}

train_dit() {
    local vae_checkpoint="${VAE_CHECKPOINT:-ckpt/vae/${RUN_DATE}/checkpoint-99.pth}"
    if [[ ! -f "${vae_checkpoint}" ]]; then
        echo "VAE checkpoint not found: ${vae_checkpoint}" >&2
        echo "Run '$0 vae' first or set VAE_CHECKPOINT." >&2
        exit 1
    fi

    echo "[GaussianWM] DiT weights -> ckpt/dit/${RUN_DATE}"
    echo "[GaussianWM] DiT log     -> logs/dit/${RUN_DATE}/${RUN_TIME}.log"
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

if [[ "${MODE}" == "vae" || "${MODE}" == "all" ]]; then
    train_vae
fi
if [[ "${MODE}" == "dit" || "${MODE}" == "all" ]]; then
    train_dit
fi
