#!/usr/bin/env bash
# Robot-side camera and gripper services. No body controller is started here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RGB_DEVICE="/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.1:1.3-video-index0"
cd "$REPO"

wait_for_gripper_hardware() {
    local deadline=$((SECONDS + 20)) carrier=0
    local -a ports=()
    shopt -s nullglob
    echo 'Waiting for eth0 and the Dex1 communication USB serial ports (up to 20s)...'
    while true; do
        ports=(/dev/ttyUSB*)
        carrier=0
        if [[ -r /sys/class/net/eth0/carrier ]]; then
            read -r carrier < /sys/class/net/eth0/carrier || carrier=0
        fi
        if [[ "$carrier" == 1 && ${#ports[@]} -gt 0 ]]; then
            printf 'Dex1 hardware ready: eth0 link up; serial ports: %s\n' "${ports[*]}"
            return 0
        fi
        if (( SECONDS >= deadline )); then
            if [[ "$carrier" != 1 ]]; then
                echo 'FAIL: robot eth0 is missing or its Ethernet link is down.' >&2
            fi
            if (( ${#ports[@]} == 0 )); then
                echo 'FAIL: no /dev/ttyUSB* communication ports. Check the gripper communication adapter USB cable/power; wrist cameras are separate devices.' >&2
            fi
            return 1
        fi
        sleep 0.2
    done
}

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
        wait_for_gripper_hardware
        sudo -n /usr/bin/systemctl start sonic-dex1.service
        if ! /usr/bin/systemctl is-active --quiet sonic-dex1.service; then
            echo 'FAIL: Dex1 service exited during startup. Recent service output:' >&2
            journalctl -u sonic-dex1.service -n 15 --no-pager -o cat >&2
            exit 1
        fi
        echo 'Dex1 service is active; the ThinkPad DDS preflight will verify both grippers.'
        ;;
    start)
        bash "$0" grippers
        sudo -n /usr/bin/systemctl restart sonic_d435i_camera.service
        /usr/bin/systemctl is-active sonic_d435i_camera.service
        ;;
    *) echo "Usage: bash $0 {start|grippers|camera}" >&2; exit 2 ;;
esac

