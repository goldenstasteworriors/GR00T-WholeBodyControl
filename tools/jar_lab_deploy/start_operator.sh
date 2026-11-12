#!/usr/bin/env bash
set -euo pipefail
deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deploy_root="$(cd -- "$deploy_dir/../.." && pwd)"
export TMPDIR="$deploy_root/runtime/tmp" XDG_CACHE_HOME="$deploy_root/runtime/cache"
export ROS_LOG_DIR="$deploy_root/runtime/ros" ROS_DOMAIN_ID="${JAR_ROS_DOMAIN_ID:-74}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$ROS_LOG_DIR"
exec "$deploy_root/.venv-control/bin/python" "$deploy_dir/operator_entry.py" \
    --interface "${JAR_G1_INTERFACE:-enp132s0}" --start-policy "$@"
