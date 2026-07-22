#!/usr/bin/env bash
set -Eeuo pipefail

GWM_ROOT="${GWM_PATH:-/workspace}"
export GWM_PATH="${GWM_ROOT}"
export PYTHONPATH="${GWM_ROOT}:${GWM_ROOT}/third_party/splatt3r:${GWM_ROOT}/third_party/splatt3r/src/pixelsplat_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r${PYTHONPATH:+:${PYTHONPATH}}"

"${GWM_ROOT}/scripts/setup_container.sh"

cd "${GWM_ROOT}"
exec "$@"
