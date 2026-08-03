#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-g1_bjutech_remote}"
REMOTE_CAMERA_HOST="${REMOTE_CAMERA_HOST:-127.0.0.1}"
REMOTE_CAMERA_PORT="${REMOTE_CAMERA_PORT:-5555}"
LOCAL_CAMERA_PORT="${LOCAL_CAMERA_PORT:-15555}"
CONDA_ENV="${CONDA_ENV:-decoupled_vla_collection}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ykj/miniconda3/bin/conda}"

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "ERROR: conda executable not found: ${CONDA_BIN}" >&2
    exit 1
fi

if ! ssh -- "${SSH_HOST}" \
    "ss -ltn | awk '{print \$4}' | grep -Eq '(^|:)'${REMOTE_CAMERA_PORT}'$'"; then
    echo "ERROR: robot camera server is not listening on port ${REMOTE_CAMERA_PORT}." >&2
    echo "Start the robot camera server first, then rerun this command." >&2
    exit 1
fi

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${LOCAL_CAMERA_PORT}$"; then
    echo "ERROR: local port ${LOCAL_CAMERA_PORT} is already in use." >&2
    echo "Choose another port, for example: LOCAL_CAMERA_PORT=15556 $0" >&2
    exit 1
fi

# The host alias may define unrelated RemoteForward entries.  Do not let a
# collision on one of those ports tear down the camera's local tunnel; the
# listener check below validates the local forward explicitly.
ssh \
    -o ExitOnForwardFailure=no \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -N \
    -L "127.0.0.1:${LOCAL_CAMERA_PORT}:${REMOTE_CAMERA_HOST}:${REMOTE_CAMERA_PORT}" \
    -- "${SSH_HOST}" &
TUNNEL_PID=$!

cleanup() {
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1
if ! kill -0 "${TUNNEL_PID}" 2>/dev/null \
    || ! ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${LOCAL_CAMERA_PORT}$"; then
    echo "ERROR: SSH camera tunnel failed to start." >&2
    exit 1
fi

echo "Camera tunnel: 127.0.0.1:${LOCAL_CAMERA_PORT} -> ${SSH_HOST}:${REMOTE_CAMERA_PORT}"
echo "Close the viewer with Q; the SSH tunnel will be closed automatically."

cd "${REPO_ROOT}"
"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python gear_sonic/scripts/run_camera_viewer.py \
    --camera-host 127.0.0.1 \
    --camera-port "${LOCAL_CAMERA_PORT}"
