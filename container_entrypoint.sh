#!/usr/bin/env bash
set -Eeuo pipefail

GWM_ROOT="${GWM_PATH:-/workspace}"
export GWM_PATH="${GWM_ROOT}"
export PYTHONPATH="${GWM_ROOT}:${GWM_ROOT}/third_party/splatt3r:${GWM_ROOT}/third_party/splatt3r/src/pixelsplat_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src:${GWM_ROOT}/third_party/splatt3r/src/mast3r_src/dust3r${PYTHONPATH:+:${PYTHONPATH}}"

PERSISTED_CODEX_BIN=/root/.codex/packages/standalone/current/codex
if [[ -x "${PERSISTED_CODEX_BIN}" ]]; then
    # /root/.codex is persisted, while /usr/local/bin belongs to each new
    # container. Restore the command link whenever the container is recreated.
    ln -sfn "${PERSISTED_CODEX_BIN}" /usr/local/bin/codex
fi

if [[ "${CODEX_REMOTE_CONTROL:-1}" == "1" ]]; then
    CODEX_BIN="$(command -v codex 2>/dev/null || true)"
    if [[ -z "${CODEX_BIN}" && -x "${PERSISTED_CODEX_BIN}" ]]; then
        # The persisted managed CLI is available before editor integrations
        # have had a chance to add `codex` to PATH in a fresh container.
        CODEX_BIN="${PERSISTED_CODEX_BIN}"
    fi

    if [[ -n "${CODEX_BIN}" ]]; then
        # PID and lock files refer to processes in the previous container and
        # must not survive into the new PID namespace. Login and installation
        # identity remain persisted elsewhere under /root/.codex.
        rm -f \
            /root/.codex/app-server-control/app-server-startup.lock \
            /root/.codex/app-server-control/app-server-control.sock \
            /root/.codex/app-server-daemon/app-server.pid \
            /root/.codex/app-server-daemon/app-server.pid.lock \
            /root/.codex/app-server-daemon/app-server-updater.pid \
            /root/.codex/app-server-daemon/app-server-updater.pid.lock \
            /root/.codex/app-server-daemon/daemon.lock

        if ! timeout 30s "${CODEX_BIN}" remote-control start --json; then
            echo "warning: Codex remote-control daemon failed to start" >&2
        elif ! "${CODEX_BIN}" app-server daemon version >/dev/null; then
            echo "warning: Codex remote-control daemon started but did not remain healthy" >&2
        fi
    else
        echo "warning: Codex CLI is unavailable; remote control was not started" >&2
    fi
fi

# Make the remote-control protocol available immediately after container start.
# Project setup can take several minutes when CUDA extensions need rebuilding.
"${GWM_ROOT}/container_setup.sh"

cd "${GWM_ROOT}"
exec "$@"
