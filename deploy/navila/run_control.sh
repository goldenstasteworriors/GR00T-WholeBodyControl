#!/usr/bin/env bash
# Starts the real WBC process; initialization itself commands joints.
# Use only with an operator at the G1. Standing still needs a manual ] press.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" != --real ]]; then
    echo 'Usage: bash deploy/navila/run_control.sh --real'
    echo 'Real joint initialization; no automatic standing or navigation enable.'
    exit 2
fi
shift
if [[ $# != 0 ]]; then echo 'Unexpected extra arguments' >&2; exit 2; fi
source "$REPO/scripts/thinkpad_collection_env.sh"
export ROS_DOMAIN_ID=73 ROS_LOCALHOST_ONLY=1 NAVILA_NAVIGATION=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
IFACE=enx00e04c36c4e8
[[ -e "/sys/class/net/$IFACE" ]] || { echo "Missing wired interface $IFACE" >&2; exit 1; }
[[ "$(cat "/sys/class/net/$IFACE/carrier")" == 1 ]] || { echo 'Ethernet link is down' >&2; exit 1; }
# Do not overlap the established collection/control session. This also checks
# non-tmux body controllers; no existing process is killed.
if pgrep -af '[p]ython.*(run_g1_control_loop|run_teleop_policy_loop)' ; then
    echo 'Existing control/teleop process found; close it before navigation' >&2
    exit 1
fi
mkdir -p "$REPO/.runtime/navila"
exec 9>"$REPO/.runtime/navila/control.lock"
flock -n 9 || { echo 'NaVILA controller already owns the lock' >&2; exit 1; }
exec python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface "$IFACE" --wbc-version gear_wbc \
    --wbc-model-path 'policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx' \
    --control-frequency 50 --no-startup-t-pose --no-enable-waist \
    --hand-type dex1 --no-with-hands --keyboard-dispatcher-type raw
