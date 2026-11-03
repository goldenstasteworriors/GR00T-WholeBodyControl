#!/usr/bin/env bash
# Starts the real WBC process; initialization itself commands joints.
# Use only with an operator at the G1. Standing still needs a manual ] press.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" != --real && "${1:-}" != --read-only ]]; then
    echo 'Usage: bash deploy/navila/run_control.sh {--read-only|--real}'
    echo 'Real joint initialization; no automatic standing or navigation enable.'
    exit 2
fi
extra=()
if [[ "$1" == --read-only ]]; then extra+=(--diagnostic-read-only); fi
shift
if [[ $# != 0 ]]; then echo 'Unexpected extra arguments' >&2; exit 2; fi
source "$REPO/scripts/thinkpad_collection_env.sh"
export ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1
export NAVILA_ESTOP_DIR="$REPO/.runtime/navila/estop"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
IFACE=enx00e04c36c4e8
exec python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface "$IFACE" --wbc-version gear_wbc \
    --wbc-model-path 'policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx' \
    --control-frequency 50 --no-startup-t-pose --no-enable-waist \
    --hand-type dex1 --no-with-hands --keyboard-dispatcher-type raw "${extra[@]}"
