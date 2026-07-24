#!/usr/bin/env bash
set -Eeuo pipefail

GWM_ROOT="${GWM_PATH:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
CUDA_ARCH_REQUEST="${TORCH_CUDA_ARCH_LIST:-auto}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MAX_JOBS="${MAX_JOBS:-8}"
SETUP_STATE_DIR="${GWM_SETUP_STATE_DIR:-/var/cache/gaussianwm}"
SPLATT3R_REPOSITORY="${SPLATT3R_REPOSITORY:-https://github.com/btsmart/splatt3r}"
SPLATT3R_CHECKPOINT="${SPLATT3R_CHECKPOINT:-${GWM_ROOT}/third_party/splatt3r/checkpoints/splatt3r_v1.0/epoch=19-step=1200.ckpt}"
SPLATT3R_CHECKPOINT_URL="${SPLATT3R_CHECKPOINT_URL:-https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt}"

log() {
    printf '[gaussianwm-setup] %s\n' "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

resolve_cuda_arch_list() {
    if [[ "${CUDA_ARCH_REQUEST,,}" != "auto" ]]; then
        printf '%s\n' "${CUDA_ARCH_REQUEST}"
        return
    fi

    command -v nvidia-smi >/dev/null 2>&1 \
        || die "TORCH_CUDA_ARCH_LIST=auto requires nvidia-smi and a visible GPU"

    local detected
    detected="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | sed '/^[[:space:]]*$/d' \
            | sort -Vu \
            | paste -sd ';' -
    )"
    [[ -n "${detected}" ]] \
        || die "Could not detect GPU compute capability; set TORCH_CUDA_ARCH_LIST explicitly"
    printf '%s\n' "${detected}"
}

ensure_splatt3r_source() {
    local source_probe="${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope/setup.py"
    [[ -f "${source_probe}" ]] && return

    log "Splatt3r source is missing; restoring it recursively"
    if git ls-files --stage third_party/splatt3r 2>/dev/null | awk '$1 == "160000" { found=1 } END { exit !found }'; then
        git submodule update --init --recursive third_party/splatt3r
        return
    fi

    # Some source archives retain .gitmodules but omit the gitlink. Clone into
    # a temporary directory and merge it with any already-downloaded checkpoint.
    local clone_root
    clone_root="$(mktemp -d "${GWM_ROOT}/third_party/.splatt3r-clone.XXXXXX")"
    git clone --recursive "${SPLATT3R_REPOSITORY}" "${clone_root}/repo"
    mkdir -p "${GWM_ROOT}/third_party/splatt3r"
    cp -a "${clone_root}/repo/." "${GWM_ROOT}/third_party/splatt3r/"
    rm -rf -- "${clone_root}"

    [[ -f "${source_probe}" ]] || die "Splatt3r clone completed without the expected DUSt3R source"
}

validate_python_environment() {
    "${PYTHON_BIN}" - <<'PY'
import cv2
import dlimp
import diff_gaussian_rasterization
import gaussianwm
import hydra
import pytorch3d
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision
import wandb
from pytorch3d.ops import sample_farthest_points

assert torch.version.cuda == "12.1", f"expected PyTorch CUDA 12.1, got {torch.version.cuda}"
assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

# Importing the extension is insufficient: a wheel compiled for another GPU
# architecture imports successfully and fails only when its kernel launches.
points = torch.randn(1, 32, 3, device="cuda")
sampled, indices = sample_farthest_points(points, K=8)
torch.cuda.synchronize()
assert sampled.shape == (1, 8, 3)
assert indices.shape == (1, 8)
PY
}

[[ -f "${GWM_ROOT}/pyproject.toml" ]] \
    || die "${GWM_ROOT} does not contain pyproject.toml"
[[ -f "${GWM_ROOT}/uv.lock" ]] \
    || die "${GWM_ROOT} does not contain uv.lock"
[[ -x "${PYTHON_BIN}" ]] \
    || die "Python executable not found: ${PYTHON_BIN}"
command -v uv >/dev/null 2>&1 || die "uv is not installed"
command -v nvcc >/dev/null 2>&1 || die "nvcc is missing; use the CUDA devel image"
[[ -d "${CUDA_HOME}" ]] || die "CUDA_HOME does not exist: ${CUDA_HOME}"

export GWM_PATH="${GWM_ROOT}"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST="$(resolve_cuda_arch_list)"
export MAX_JOBS
export FORCE_CUDA=1
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--include cstdint}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_SYSTEM_PYTHON=1
export PYTHONPATH="${GWM_ROOT}:${GWM_ROOT}/third_party/splatt3r:${GWM_ROOT}/third_party/splatt3r/src/pixelsplat_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SETUP_STATE_DIR}" "${GWM_ROOT}/third_party"
rm -f "${SETUP_STATE_DIR}/ready"
cd "${GWM_ROOT}"
git config --global --add safe.directory "${GWM_ROOT}"

log "Python=$("${PYTHON_BIN}" --version 2>&1), CUDA_HOME=${CUDA_HOME}, CUDA architectures=${TORCH_CUDA_ARCH_LIST}"
ensure_splatt3r_source

setup_hash="$({
    sha256sum pyproject.toml uv.lock scripts/setup_container.sh
    git -C third_party/splatt3r rev-parse HEAD 2>/dev/null || true
    nvcc --version
    "${PYTHON_BIN}" --version
    printf '%s\n' "${TORCH_CUDA_ARCH_LIST}"
} | sha256sum | cut -d' ' -f1)"
setup_marker="${SETUP_STATE_DIR}/python-${setup_hash}.done"

python_ready=0
if [[ -f "${setup_marker}" ]] && validate_python_environment >/dev/null 2>&1; then
    python_ready=1
fi

if [[ "${python_ready}" -ne 1 ]]; then
    log "Installing the exact Python dependency set from uv.lock"
    requirements_file="$(mktemp "${SETUP_STATE_DIR}/requirements.XXXXXX.txt")"
    uv export \
        --locked \
        --no-dev \
        --no-emit-project \
        --emit-index-url \
        --no-hashes \
        --output-file "${requirements_file}" \
        >/dev/null
    uv pip sync \
        --system \
        --python "${PYTHON_BIN}" \
        --index-strategy unsafe-best-match \
        "${requirements_file}"
    rm -f -- "${requirements_file}"
    uv pip install \
        --system \
        --python "${PYTHON_BIN}" \
        --no-deps \
        --editable "${GWM_ROOT}"

    log "Building diff-gaussian-rasterization for CUDA architectures ${TORCH_CUDA_ARCH_LIST}"
    uv pip install \
        --system \
        --python "${PYTHON_BIN}" \
        --reinstall \
        --no-cache \
        --no-deps \
        --no-build-isolation \
        'git+https://github.com/dcharatan/diff-gaussian-rasterization-modified@1250c420ebb945f0dce9945086e22faab9157c92'

    log "Building PyTorch3D for CUDA architectures ${TORCH_CUDA_ARCH_LIST}"
    uv pip install \
        --system \
        --python "${PYTHON_BIN}" \
        --reinstall \
        --no-cache \
        --no-deps \
        --no-build-isolation \
        'git+https://github.com/facebookresearch/pytorch3d.git@4daa00b41c52455440b938d1b676e00935b204d7'

    validate_python_environment
    "${PYTHON_BIN}" -m pip check
    touch "${setup_marker}"
else
    log "Locked Python dependencies and CUDA extensions are ready"
fi

CUROPE_DIR="${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r/croco/models/curope"
curope_hash="$({
    sha256sum "${CUROPE_DIR}/setup.py"
    nvcc --version
    printf '%s\n' "${TORCH_CUDA_ARCH_LIST}"
} | sha256sum | cut -d' ' -f1)"
curope_marker="${SETUP_STATE_DIR}/curope-${curope_hash}.done"
if [[ ! -f "${curope_marker}" ]] || ! find "${CUROPE_DIR}" -maxdepth 1 -name 'curope*.so' -print -quit | grep -q .; then
    log "Compiling the Splatt3r cuRoPE CUDA kernel for ${TORCH_CUDA_ARCH_LIST}"
    (cd "${CUROPE_DIR}" && "${PYTHON_BIN}" setup.py build_ext --inplace)
    find "${CUROPE_DIR}" -maxdepth 1 -name 'curope*.so' -print -quit | grep -q . \
        || die "cuRoPE build finished without producing a shared library"
    touch "${curope_marker}"
else
    log "Splatt3r cuRoPE CUDA kernel is ready"
fi

if [[ "${GWM_DOWNLOAD_SPLATT3R:-1}" == "1" && ! -s "${SPLATT3R_CHECKPOINT}" ]]; then
    log "Downloading the Splatt3r checkpoint"
    mkdir -p "$(dirname "${SPLATT3R_CHECKPOINT}")"
    checkpoint_tmp="${SPLATT3R_CHECKPOINT}.part"
    curl -fL --retry 5 --continue-at - --output "${checkpoint_tmp}" "${SPLATT3R_CHECKPOINT_URL}"
    mv "${checkpoint_tmp}" "${SPLATT3R_CHECKPOINT}"
fi

if [[ -n "${DROID_DATASET_DIR:-}" ]]; then
    if [[ ! -d "${DROID_DATASET_DIR}" ]]; then
        log "DROID dataset not found at ${DROID_DATASET_DIR}; skipping the data link"
    else
        dataset_link="${GWM_ROOT}/data/droid_100"
        mkdir -p "$(dirname "${dataset_link}")"
        if [[ -e "${dataset_link}" && ! -L "${dataset_link}" ]]; then
            die "${dataset_link} exists and is not a symlink; refusing to overwrite it"
        fi
        ln -sfn "${DROID_DATASET_DIR}" "${dataset_link}"
        log "DROID dataset linked: ${dataset_link} -> ${DROID_DATASET_DIR}"
    fi
fi

"${PYTHON_BIN}" - <<'PY'
import torch
from pytorch3d.ops import sample_farthest_points

gpu = torch.cuda.get_device_name(0)
capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
points = torch.randn(1, 32, 3, device="cuda")
sample_farthest_points(points, K=8)
torch.cuda.synchronize()
print(
    f"[gaussianwm-setup] torch={torch.__version__}, CUDA runtime={torch.version.cuda}, "
    f"GPU={gpu}, compute capability={capability}, PyTorch3D CUDA kernel=OK"
)
PY

touch "${SETUP_STATE_DIR}/ready"
log "Setup complete"
