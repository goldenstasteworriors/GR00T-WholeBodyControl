#!/usr/bin/env bash
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${ONBOARD_CONDA_ROOT:-/home/unitree/miniconda3}"
export PATH="$conda_root/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
exec "${ONBOARD_PYTHON:-$conda_root/envs/decoupled_vla_collection/bin/python}" "$script_root/gear_sonic/scripts/launch_split_collection_onboard.py" "$@"
