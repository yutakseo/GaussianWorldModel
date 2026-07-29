#!/usr/bin/env bash

set -euo pipefail

# 직접 사용할 체크포인트 경로를 여기서 지정합니다.
DEFAULT_CHECKPOINT="ckpt/dit/2026-07-29/model_latest.pt"
DEFAULT_VAE_CHECKPOINT="ckpt/vae/2026-07-29/checkpoint-latest.pth"

DEFAULT_OUTPUT_ROOT="outputs"
DEFAULT_NUM_SAMPLES="5"
DEFAULT_PYTHON_BIN="python3"
DEFAULT_CUDA_DEVICES="0"

validate_args() {
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "DiT checkpoint not found: ${CHECKPOINT}" >&2
        return 1
    fi

    if [[ ! -f "${VAE_CHECKPOINT}" ]]; then
        echo "VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2
        return 1
    fi

    if ! [[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_SAMPLES must be a positive integer: ${NUM_SAMPLES}" >&2
        return 2
    fi
}

run_inference() {
    echo "[GaussianWM] Checkpoint: ${CHECKPOINT}"
    echo "[GaussianWM] VAE:        ${VAE_CHECKPOINT}"
    echo "[GaussianWM] Output:     ${OUTPUT_DIR}"
    echo "[GaussianWM] Log:        ${LOG_FILE}"

    mkdir -p "$(dirname -- "${LOG_FILE}")"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        "${PYTHON_BIN}" -m gaussianwm.demo \
        --config-name train_gwm \
        dataset=droid \
        paths.stage=infer \
        paths.log_stage=infer \
        "paths.date=${RUN_DATE}" \
        "paths.time=${RUN_TIME}" \
        "resume=${CHECKPOINT}" \
        "demo.output_dir=${OUTPUT_DIR}" \
        "demo.num_samples=${NUM_SAMPLES}" \
        "world_model.vae.pretrained_path=${VAE_CHECKPOINT}" \
        world_model.vae.use_vae=true \
        world_model.observation.use_gs=true \
        world_model.reward.use_reward_model=false \
        use_wandb=false 2>&1 | tee -a "${LOG_FILE}"
}

main() {
    local script_dir
    local repo_dir

    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_dir="$(cd -- "${script_dir}/.." && pwd)"
    cd "${repo_dir}"

    RUN_DATE="${RUN_DATE:-$(date -u +%F)}"
    RUN_TIME="${RUN_TIME:-$(date -u +%H-%M-%S)}"
    CHECKPOINT="${1:-${CHECKPOINT:-${DEFAULT_CHECKPOINT}}}"
    VAE_CHECKPOINT="${VAE_CHECKPOINT:-${DEFAULT_VAE_CHECKPOINT}}"
    OUTPUT_DIR="${2:-${OUTPUT_DIR:-${DEFAULT_OUTPUT_ROOT}/${RUN_DATE}/${RUN_TIME}}}"
    NUM_SAMPLES="${3:-${NUM_SAMPLES:-${DEFAULT_NUM_SAMPLES}}}"
    PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
    CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_DEVICES}}"
    LOG_FILE="logs/infer/${RUN_DATE}/${RUN_TIME}.log"

    export GWM_PATH="${GWM_PATH:-${repo_dir}}"
    export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

    validate_args
    run_inference
}

main "$@"
