#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.venv_ref_bridge/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv_ref_bridge; see docs/source/tutorials/thinkpad_ref_bridge.md" >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-$REPO_ROOT/gear_sonic/config/data_collection/thinkpad_ref_fastdds.xml}"
exec "$PYTHON" -m gear_sonic.scripts.run_thinkpad_ref_bridge "$@"
