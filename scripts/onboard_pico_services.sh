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
DHCP_PID_FILE="${STATE_DIR}/dnsmasq.pid"
DHCP_LOG_FILE="${STATE_DIR}/dnsmasq.log"

usage() {
    echo "Usage: $0 {start|status|attach|stop}"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found: $1" >&2
        exit 1
    fi
}

dhcp_pid() {
    [[ -f "$DHCP_PID_FILE" ]] || return 1
    local pid
    pid="$(<"$DHCP_PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" \
        | grep -Fq "/usr/sbin/dnsmasq" || return 1
    echo "$pid"
}

dhcp_running() {
    dhcp_pid >/dev/null
}

pc_service_running() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null || return 1
    tmux list-panes -t "$SESSION_NAME" -F '#{pane_current_command}' \
        | grep -Eq '(^|/)RoboticsServiceProcess$'
}

status() {
    echo "PICO services tmux session:"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "  running ($SESSION_NAME)"
        tmux list-windows -t "$SESSION_NAME" -F '  window #{window_index}: #{window_name} (#{pane_current_command})'
    else
        echo "  not running"
    fi

    echo "DHCP:"
    if dhcp_running; then
        echo "  running (pid $(dhcp_pid), log ${DHCP_LOG_FILE})"
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

    if pc_service_running && dhcp_running; then
        echo "PICO DHCP and PC Service are already running."
        status
        return
    fi

    if ! dhcp_running \
        && ss -lun 2>/dev/null | grep -qE '(^|[[:space:]])[^[:space:]]*:67[[:space:]]'; then
        echo "ERROR: UDP port 67 is already in use." >&2
        echo "Stop the manually started dnsmasq with Ctrl+C before using this script." >&2
        exit 1
    fi

    mkdir -p "$STATE_DIR"

    if ! dhcp_running; then
        # Authenticate in the invoking terminal.  Starting sudo inside a new
        # tmux PTY does not inherit the sudo timestamp on systems using
        # tty-scoped tickets, so let dnsmasq daemonize directly instead.
        sudo -v
        sudo -n /usr/sbin/dnsmasq \
            --interface="$INTERFACE" \
            --bind-interfaces \
            --port=0 \
            --log-dhcp \
            --log-facility="$DHCP_LOG_FILE" \
            --pid-file="$DHCP_PID_FILE" \
            --dhcp-range=192.168.123.0,static,255.255.255.0,12h \
            --dhcp-host="${PICO_MAC},${PICO_IP}" \
            --dhcp-leasefile="$LEASE_FILE"
        sleep 1
        if ! dhcp_running; then
            echo "ERROR: dnsmasq exited during startup. Check ${DHCP_LOG_FILE}" >&2
            exit 1
        fi
    fi

    local service_cmd
    service_cmd="cd '${RUNTIME_DIR}' && export LD_LIBRARY_PATH='${RUNTIME_DIR}:${RUNTIME_DIR}/lib:${BUILD_ENV}/lib' && exec ./RoboticsServiceProcess"

    if ! pc_service_running; then
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux kill-session -t "$SESSION_NAME"
        fi
        tmux new-session -d -s "$SESSION_NAME" -n pc_service "$service_cmd"
    fi
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
            echo "Stopped XRoboToolkit PC Service."
        else
            echo "XRoboToolkit PC Service is not running."
        fi
        if dhcp_running; then
            pid="$(dhcp_pid)"
            sudo -v
            sudo -n kill "$pid"
            for _ in {1..20}; do
                [[ ! -r "/proc/${pid}/cmdline" ]] && break
                sleep 0.1
            done
            if [[ -r "/proc/${pid}/cmdline" ]]; then
                echo "ERROR: dnsmasq pid ${pid} did not stop" >&2
                exit 1
            fi
            rm -f "$DHCP_PID_FILE"
            echo "Stopped PICO DHCP service."
        else
            echo "PICO DHCP service is not running."
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
