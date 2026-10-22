#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/thinkpad_collection_env.sh"
ssh -o BatchMode=yes -- g1 'cd /home/unitree/GR00T-WholeBodyControl && bash scripts/robot_collection_services.sh grippers'
if ! tmux has-session -t dex1_gripper_web 2>/dev/null; then
    printf -v command 'source %q && exec python -m gear_sonic.scripts.run_dex1_gripper_web --network enx00e04c36c4e8 --profile .runtime/dex1_profiles/default.json --port 8766' \
        "$COLLECTION_REPO/scripts/thinkpad_collection_env.sh"
    tmux new-session -d -s dex1_gripper_web -c "$COLLECTION_REPO" "bash -lc $(printf %q "$command")"
fi
echo 'Dex1 web: http://127.0.0.1:8766'
if [[ -n "${DISPLAY:-}" ]]; then xdg-open http://127.0.0.1:8766 >/dev/null 2>&1 & fi

