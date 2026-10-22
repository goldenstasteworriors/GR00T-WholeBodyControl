#!/usr/bin/env bash
# Robot-side camera and gripper services. No body controller is started here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RGB_DEVICE="/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.1:1.3-video-index0"
cd "$REPO"
case "${1:-start}" in
    camera)
        [[ -e "$RGB_DEVICE" ]] || { echo "RGB device missing: $RGB_DEVICE" >&2; exit 1; }
        exec "$REPO/.venv_camera/bin/python" -m gear_sonic.camera.composed_camera \
            --ego-view-camera usb --ego-view-device-id "$RGB_DEVICE" \
            --left-wrist-camera usb --left-wrist-device-id /dev/v4l/by-id/usb-JR0001_JR0001_JR0001-video-index0 \
            --right-wrist-camera usb --right-wrist-device-id /dev/v4l/by-id/usb-JR0002_JR0002_JR0002-video-index0 \
            --fps 30 --port 5555 --sync-max-skew-ms 30 --sync-max-age-ms 250 \
            --profile-timing --profile-interval 10
        ;;
    grippers)
        sudo -n /usr/bin/systemctl start sonic-dex1.service
        /usr/bin/systemctl is-active sonic-dex1.service
        ;;
    start)
        bash "$0" grippers
        sudo -n /usr/bin/systemctl restart sonic_d435i_camera.service
        /usr/bin/systemctl is-active sonic_d435i_camera.service
        ;;
    *) echo "Usage: bash $0 {start|grippers|camera}" >&2; exit 2 ;;
esac

