#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/thinkpad_collection_env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" ROS_LOCALHOST_ONLY=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python "$REPO/deploy/navila/bridge.py" "$@"
