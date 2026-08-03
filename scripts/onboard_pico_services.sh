#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="onboard_pico_services"
INTERFACE="eth0"
ROBOT_IP="192.168.123.164"
PICO_IP="192.168.123.200"
PICO_MAC="dc:04:5a:1d:93:3b"
RUNTIME_DIR="${REPO_ROOT}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/runtime/ubuntu20-aarch64"
BUILD_ENV="${REPO_ROOT}/.build_envs/xrobotoolkit-ubuntu20-arm64"
STATE_DIR="${REPO_ROOT}/.runtime/onboard_pico"
LEASE_FILE="${STATE_DIR}/dnsmasq.leases"

usage() {
    echo "Usage: $0 {start|status|attach|stop}"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

status() {
    echo "PICO services tmux session:"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "  running ($SESSION_NAME)"
        tmux list-windows -t "$SESSION_NAME" -F '  window #{window_index}: #{window_name} (#{pane_current_command})'
    else
        echo "  not running"
    fi

    echo "Network:"
    ip -4 -brief address show dev "$INTERFACE" || true
    ip neigh show "$PICO_IP" dev "$INTERFACE" || true

    echo "Listeners:"
    ss -lntup 2>/dev/null | grep -E '(:67 |:63901 |:60061 )' || true
}

start() {
    command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
    command -v dnsmasq >/dev/null || { echo "ERROR: dnsmasq is not installed" >&2; exit 1; }
    require_file "${RUNTIME_DIR}/RoboticsServiceProcess"
    require_file "${BUILD_ENV}/lib/libQt6Core.so.6.7.3"

    if ! ip -4 address show dev "$INTERFACE" | grep -q "${ROBOT_IP}/24"; then
        echo "ERROR: ${INTERFACE} does not have expected robot address ${ROBOT_IP}/24" >&2
        echo "Refusing to change the robot network configuration." >&2
        exit 1
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "PICO services are already running."
        status
        return
    fi

    if ss -lun 2>/dev/null | grep -qE '(^|[[:space:]])[^[:space:]]*:67[[:space:]]'; then
        echo "ERROR: UDP port 67 is already in use." >&2
        echo "Stop the manually started dnsmasq with Ctrl+C before using this script." >&2
        exit 1
    fi

    mkdir -p "$STATE_DIR"
    sudo -v

    local dhcp_cmd
    dhcp_cmd="exec sudo -n /usr/sbin/dnsmasq --no-daemon --interface=${INTERFACE} --bind-interfaces --port=0 --log-dhcp --dhcp-range=192.168.123.0,static,255.255.255.0,12h --dhcp-host=${PICO_MAC},${PICO_IP} --dhcp-leasefile=${LEASE_FILE}"

    local service_cmd
    service_cmd="cd '${RUNTIME_DIR}' && export LD_LIBRARY_PATH='${RUNTIME_DIR}:${RUNTIME_DIR}/lib:${BUILD_ENV}/lib' && exec ./RoboticsServiceProcess"

    tmux new-session -d -s "$SESSION_NAME" -n dhcp "$dhcp_cmd"
    tmux new-window -t "$SESSION_NAME" -n pc_service "$service_cmd"
    sleep 2

    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "ERROR: PICO service session exited during startup." >&2
        exit 1
    fi

    echo "PICO DHCP and PC Service started."
    echo "  Robot: ${ROBOT_IP}"
    echo "  PICO:  ${PICO_IP} (${PICO_MAC})"
    echo "  View:  $0 attach"
    echo "  Check: $0 status"
}

case "${1:-}" in
    start)
        start
        ;;
    status)
        status
        ;;
    attach)
        exec tmux attach -t "$SESSION_NAME"
        ;;
    stop)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux kill-session -t "$SESSION_NAME"
            echo "Stopped $SESSION_NAME."
        else
            echo "$SESSION_NAME is not running."
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
