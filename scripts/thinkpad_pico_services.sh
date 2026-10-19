#!/usr/bin/env bash
# Start only PICO networking/services; never create a robot controller.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${REPO_ROOT}/.runtime/thinkpad_pico"
SERVICE_DIR="/opt/apps/roboticsservice"
SESSION="thinkpad_pico_service"
INTERFACE="enx00e04c36c4e8"
PC_IP="192.168.123.222"
PICO_IP="192.168.123.200"
PICO_MAC="${PICO_ETH_MAC:-dc:04:5a:1d:93:3b}"
PID_FILE="${STATE_DIR}/dnsmasq.pid"

service_pids() {
    pgrep -f '^/opt/apps/roboticsservice/RoboticsServiceProcess([[:space:]]|$)' || true
}

dhcp_pid() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq -- "--pid-file=${PID_FILE}" || return 1
    echo "$pid"
}

status() {
    echo "PC Service PIDs: $(service_pids)"
    echo "Managed DHCP PID: $(dhcp_pid || true)"
    echo "PICO Ethernet MAC: ${PICO_MAC}"
    echo "Wired PC Service IP: ${PC_IP}"
    echo "Wi-Fi IP (use only when headset is on the same reachable network):"
    ip -4 -brief address show || true
    ip neigh show "$PICO_IP" dev "$INTERFACE" || true
    [[ ! -f "${STATE_DIR}/dnsmasq.leases" ]] || cat "${STATE_DIR}/dnsmasq.leases"
    echo "Logs: ${STATE_DIR}"
}

start_service() {
    command -v tmux >/dev/null
    [[ -x "${SERVICE_DIR}/RoboticsServiceProcess" ]] || { echo "PC Service is not installed" >&2; exit 1; }
    mkdir -p "$STATE_DIR"
    if [[ -n "$(service_pids)" ]]; then
        echo "Reusing existing PC Service"
        return
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Managed tmux session exists without PC Service; inspect before restarting" >&2
        exit 1
    fi
    local command
    printf -v command 'cd %q && export LD_LIBRARY_PATH=%q && export QT_PLUGIN_PATH=%q && export QT_QML_PATH=%q && exec %q >> %q 2>&1' \
        "$SERVICE_DIR" "${SERVICE_DIR}:${SERVICE_DIR}/lib:${SERVICE_DIR}/SDK/x64:${LD_LIBRARY_PATH:-}" \
        "${SERVICE_DIR}/plugins" "${SERVICE_DIR}/qml" "${SERVICE_DIR}/RoboticsServiceProcess" "${STATE_DIR}/service.log"
    tmux new-session -d -s "$SESSION" -n pc_service "$command"
    sleep 2
    [[ -n "$(service_pids)" ]] || { echo "PC Service failed; inspect ${STATE_DIR}/service.log" >&2; exit 1; }
}

start_wired() {
    [[ "$PICO_MAC" =~ ^([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}$ ]] || { echo "Invalid PICO_ETH_MAC" >&2; exit 1; }
    ip -4 address show dev "$INTERFACE" | grep -Fq "${PC_IP}/24" || { echo "Expected ${PC_IP}/24 on ${INTERFACE}; no network changes made" >&2; exit 1; }
    command -v dnsmasq >/dev/null
    mkdir -p "$STATE_DIR"
    if ! dhcp_pid >/dev/null; then
        if ss -lun | grep -qE '[[:space:]][^[:space:]]*:67[[:space:]]'; then
            echo "UDP 67 already in use; refusing a second DHCP server" >&2
            exit 1
        fi
        sudo -v
        sudo dnsmasq --conf-file=/dev/null --port=0 --no-resolv --no-hosts \
            --interface="$INTERFACE" --except-interface=lo --bind-interfaces \
            --dhcp-range=192.168.123.0,static,255.255.255.0 \
            --dhcp-host="${PICO_MAC},set:pico,${PICO_IP}" --dhcp-ignore=tag:!pico \
            --dhcp-option=3 --dhcp-option=6 \
            --dhcp-leasefile="${STATE_DIR}/dnsmasq.leases" --pid-file="$PID_FILE" \
            --log-dhcp --log-facility="${STATE_DIR}/dnsmasq.log"
    fi
    start_service
    status
}

case "${1:-status}" in
    start) start_service; status ;;
    start-wired) start_wired ;;
    status) status ;;
    stop)
        if tmux has-session -t "$SESSION" 2>/dev/null; then tmux kill-session -t "$SESSION"; fi
        if pid="$(dhcp_pid)"; then sudo kill "$pid"; fi
        echo "Stopped managed PICO services; other services left running"
        ;;
    *) echo "Usage: bash $0 {start|start-wired|status|stop}" >&2; exit 2 ;;
esac
