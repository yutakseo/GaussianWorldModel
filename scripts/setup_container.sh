#!/usr/bin/env bash
set -Eeuo pipefail

GWM_ROOT="${GWM_PATH:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-7.5}"
SETUP_STATE_DIR="${GWM_SETUP_STATE_DIR:-/var/cache/gaussianwm}"
SPLATT3R_CHECKPOINT="${SPLATT3R_CHECKPOINT:-${GWM_ROOT}/third_party/splatt3r/checkpoints/splatt3r_v1.0/epoch=19-step=1200.ckpt}"
SPLATT3R_CHECKPOINT_URL="${SPLATT3R_CHECKPOINT_URL:-https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt}"

log() {
    printf '[gaussianwm-setup] %s\n' "$*"
}

if [[ ! -f "${GWM_ROOT}/pyproject.toml" ]]; then
    log "ERROR: ${GWM_ROOT} does not contain pyproject.toml"
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    log "ERROR: nvcc is missing. Build with the CUDA devel Dockerfile."
    exit 1
fi

export GWM_PATH="${GWM_ROOT}"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_SYSTEM_PYTHON=1
export PYTHONPATH="${GWM_ROOT}:${GWM_ROOT}/third_party/splatt3r:${GWM_ROOT}/third_party/splatt3r/src/pixelsplat_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SETUP_STATE_DIR}"
cd "${GWM_ROOT}"

if [[ -f .gitmodules ]] && [[ ! -f third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope/setup.py ]]; then
    log "Initializing Git submodules"
    git submodule update --init --recursive
fi

setup_hash="$({
    sha256sum pyproject.toml scripts/setup_container.sh
    sha256sum third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope/setup.py 2>/dev/null || true
    printf '%s\n' "${CUDA_ARCH}"
} | sha256sum | cut -d' ' -f1)"
setup_marker="${SETUP_STATE_DIR}/python-${setup_hash}.done"

python_ready=0
if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import cv2
import dlimp
import gaussianwm
import pytorch3d
import torch
import torchvision
import diff_gaussian_rasterization
assert torch.version.cuda == "12.1", torch.version.cuda
PY
then
    python_ready=1
fi

if [[ ! -f "${setup_marker}" || "${python_ready}" -ne 1 ]]; then
    log "Installing Python dependencies into the container's system Python"
    uv pip install --system --python "${PYTHON_BIN}" \
        'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' \
        --index-url https://download.pytorch.org/whl/cu121
    uv pip install --system --python "${PYTHON_BIN}" -e "${GWM_ROOT}"

    log "Building diff-gaussian-rasterization for CUDA ${CUDA_ARCH}"
    uv pip install --system --python "${PYTHON_BIN}" \
        'git+https://github.com/dcharatan/diff-gaussian-rasterization-modified@1250c420ebb945f0dce9945086e22faab9157c92' \
        --no-build-isolation

    log "Building PyTorch3D for CUDA ${CUDA_ARCH}"
    uv pip install --system --python "${PYTHON_BIN}" \
        'git+https://github.com/facebookresearch/pytorch3d.git@4daa00b41c52455440b938d1b676e00935b204d7' \
        --no-build-isolation

    touch "${setup_marker}"
else
    log "Python and CUDA dependencies are already installed"
fi

CUROPE_DIR="${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope"
curope_marker="${SETUP_STATE_DIR}/curope-${CUDA_ARCH//./_}.done"
if [[ -d "${CUROPE_DIR}" ]] && { [[ ! -f "${curope_marker}" ]] || ! find "${CUROPE_DIR}" -maxdepth 1 -name 'curope*.so' -print -quit | grep -q .; }; then
    log "Compiling the optional Splatt3r cuRoPE CUDA kernel"
    (cd "${CUROPE_DIR}" && "${PYTHON_BIN}" setup.py build_ext --inplace)
    touch "${curope_marker}"
else
    log "Splatt3r cuRoPE CUDA kernel is already compiled"
fi

if [[ "${GWM_DOWNLOAD_SPLATT3R:-1}" == "1" && ! -s "${SPLATT3R_CHECKPOINT}" ]]; then
    log "Downloading the Splatt3r checkpoint (about 3.1 GB)"
    mkdir -p "$(dirname "${SPLATT3R_CHECKPOINT}")"
    checkpoint_tmp="${SPLATT3R_CHECKPOINT}.part"
    curl -fL --retry 5 --continue-at - --output "${checkpoint_tmp}" "${SPLATT3R_CHECKPOINT_URL}"
    mv "${checkpoint_tmp}" "${SPLATT3R_CHECKPOINT}"
fi

if [[ -n "${DROID_DATASET_DIR:-}" ]]; then
    if [[ ! -d "${DROID_DATASET_DIR}" ]]; then
        log "DROID dataset not found yet at ${DROID_DATASET_DIR}; skipping the data link"
    else
        mkdir -p "${GWM_ROOT}/data"
        ln -sfn "${DROID_DATASET_DIR}" "${GWM_ROOT}/data/droid_100"
        log "DROID dataset linked from ${DROID_DATASET_DIR}"
    fi
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import pytorch3d
import diff_gaussian_rasterization
print(f"[gaussianwm-setup] torch={torch.__version__}, CUDA runtime={torch.version.cuda}, GPU available={torch.cuda.is_available()}")
PY

touch "${SETUP_STATE_DIR}/ready"
log "Setup complete"
