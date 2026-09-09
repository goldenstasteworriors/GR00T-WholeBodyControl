#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
UV_BIN="${UV_BIN:-$REPO_ROOT/.tools/ref-bridge/uv}"
BASE_PYTHON="${BASE_PYTHON:-/home/data-collection/miniconda3/envs/decoupled_vla_collection/bin/python}"
if [[ ! -x "$UV_BIN" || ! -x "$BASE_PYTHON" ]]; then
    echo "Set UV_BIN to uv and BASE_PYTHON to the existing collection Conda Python." >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/.cache/ref-bridge"
export UV_CACHE_DIR="$REPO_ROOT/.cache/ref-bridge"
export TMPDIR="$REPO_ROOT/.cache/ref-bridge"
if [[ ! -d .venv_ref_bridge ]]; then
    "$UV_BIN" venv --system-site-packages --python "$BASE_PYTHON" .venv_ref_bridge
fi
"$UV_BIN" pip install --python .venv_ref_bridge/bin/python --no-deps -r install_scripts/requirements_thinkpad_ref_bridge.txt
bash scripts/run_thinkpad_ref_bridge.sh --preflight
