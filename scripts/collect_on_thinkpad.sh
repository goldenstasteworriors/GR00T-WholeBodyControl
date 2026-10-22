#!/usr/bin/env bash
set -euo pipefail
[[ "$#" == 3 ]] || { echo "Usage: bash $0 GRIPPER_TASK PROMPT PROJECT_NAME" >&2; exit 2; }
GRIPPER_TASK="$1"
TASK_PROMPT="$2"
PROJECT_NAME="$3"
source "$(dirname "${BASH_SOURCE[0]}")/thinkpad_collection_env.sh"
# Resolve the preset before starting remote services.
python -c 'import sys; from gear_sonic.utils.dex1_profile import load_profile,preset_path; print(load_profile(preset_path(".runtime/dex1_profiles",sys.argv[1])))' "$GRIPPER_TASK"
if tmux has-session -t dex1_gripper_web 2>/dev/null; then
    tmux send-keys -t dex1_gripper_web C-c
fi
ssh -o BatchMode=yes -- g1 'cd /home/unitree/GR00T-WholeBodyControl && bash scripts/robot_collection_services.sh start'
bash scripts/thinkpad_pico_services.sh start
diagnostic_args=()
# Used for acceptance testing only; the public entry has exactly three arguments.
if [[ "${COLLECTION_READ_ONLY:-0}" == 1 ]]; then
    diagnostic_args+=(--diagnostic-read-only --no-pico-preflight)
fi
if [[ "${COLLECTION_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    diagnostic_args+=(--preflight-only)
fi
exec python gear_sonic/scripts/launch_decoupled_vla_collection.py \
    --interface enx00e04c36c4e8 \
    --hand-type dex1 --dex1-preset "$GRIPPER_TASK" \
    --camera-host 192.168.123.164 --record-wrist-cameras \
    --camera-sync-tolerance-ms 30 --camera-max-hold-ms 250 \
    --data-exporter-frequency 30 \
    --root-output-dir "$COLLECTION_REPO/outputs" \
    --task-prompt "$TASK_PROMPT" \
    --dataset-name "${PROJECT_NAME}_$(date +%Y%m%d_%H%M%S)" \
    --hand-task grab_middle_beaker --no-pico-data-streamer --no-startup-t-pose \
    --profile-timing --camera-viewer --ros-domain-id 73 --save-profile-logs \
    "${diagnostic_args[@]}"
