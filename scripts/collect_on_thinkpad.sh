#!/usr/bin/env bash
set -euo pipefail
[[ "$#" == 3 ]] || { echo "Usage: bash $0 GRIPPER_TASK PROMPT PROJECT_NAME" >&2; exit 2; }
GRIPPER_TASK="$1"
TASK_PROMPT="$2"
PROJECT_NAME="$3"
source "$(dirname "${BASH_SOURCE[0]}")/thinkpad_collection_env.sh"
ROBOT_NETWORK=enx00e04c36c4e8
if [[ ! -e "/sys/class/net/$ROBOT_NETWORK" ]]; then
    echo "Robot Ethernet adapter $ROBOT_NETWORK is missing; reconnect the ThinkPad USB adapter." >&2
    exit 1
fi
if [[ "$(cat "/sys/class/net/$ROBOT_NETWORK/carrier")" != 1 ]]; then
    echo "Robot Ethernet link is down; check the cable and robot power." >&2
    exit 1
fi
ROBOT_ROUTE="$(ip -4 route get 192.168.123.164)"
if [[ " $ROBOT_ROUTE " != *" dev $ROBOT_NETWORK "* ]]; then
    echo "Wrong robot route: $ROBOT_ROUTE; expected $ROBOT_NETWORK." >&2
    exit 1
fi
echo '[Startup 1/4] Ethernet ready; checking collection configuration'
hand_args=(--hand-type dex1)
if [[ "${NAVILA_COLLECTION:-0}" == 1 ]]; then
    hand_args+=(--no-with-hands)
    robot_start='cd /home/unitree/GR00T-WholeBodyControl && sudo -n /usr/bin/systemctl restart sonic_d435i_camera.service'
else
# Resolve the preset before starting remote services.
python -c 'import sys; from gear_sonic.utils.dex1_profile import load_profile,preset_path; print(load_profile(preset_path(".runtime/dex1_profiles",sys.argv[1])))' "$GRIPPER_TASK"
hand_args+=(--dex1-preset "$GRIPPER_TASK")
if tmux has-session -t dex1_gripper_web 2>/dev/null; then
    tmux send-keys -t dex1_gripper_web C-c
fi
robot_start='cd /home/unitree/GR00T-WholeBodyControl && bash scripts/robot_collection_services.sh start'
fi
echo '[Startup 2/4] Starting robot camera service (navigation does not start grippers)'
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 -- g1 "$robot_start"
# The three USB cameras initialize sequentially before the server can publish.
echo 'Waiting for the three-camera startup to complete...'
python -c 'from gear_sonic.scripts.check_dex1_connection import check_camera; raise SystemExit(0 if check_camera("192.168.123.164",5555,timeout=30.0) else 1)'
echo '[Startup 3/4] Starting/reusing ThinkPad PICO PC Service'
bash scripts/thinkpad_pico_services.sh start
diagnostic_args=()
if [[ "${NAVILA_COLLECTION:-0}" == 1 ]]; then
    diagnostic_args+=(--navigation-collection)
fi
# Used for acceptance testing only; the public entry has exactly three arguments.
if [[ "${COLLECTION_READ_ONLY:-0}" == 1 ]]; then
    diagnostic_args+=(--diagnostic-read-only --no-pico-preflight)
fi
if [[ "${COLLECTION_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    diagnostic_args+=(--preflight-only)
fi
echo '[Startup 4/4] Checking DDS/PICO and launching collection panes'
exec python gear_sonic/scripts/launch_decoupled_vla_collection.py \
    --interface "$ROBOT_NETWORK" \
    "${hand_args[@]}" \
    --camera-host 192.168.123.164 --record-wrist-cameras \
    --camera-sync-tolerance-ms 30 --camera-max-hold-ms 250 \
    --data-exporter-frequency 30 \
    --root-output-dir "$COLLECTION_REPO/outputs" \
    --task-prompt "$TASK_PROMPT" \
    --dataset-name "${PROJECT_NAME}_$(date +%Y%m%d_%H%M%S)" \
    --hand-task grab_middle_beaker --no-pico-data-streamer --no-startup-t-pose \
    --profile-timing --camera-viewer --ros-domain-id 73 --save-profile-logs \
    "${diagnostic_args[@]}"
