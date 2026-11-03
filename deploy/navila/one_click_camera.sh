#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/thinkpad_collection_env.sh"
ssh -o BatchMode=yes -o ConnectTimeout=5 -- 'g1' 'systemctl is-active --quiet sonic_d435i_camera.service || sudo -n systemctl restart sonic_d435i_camera.service'
exec python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
