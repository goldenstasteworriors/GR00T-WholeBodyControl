#!/usr/bin/env bash
# One controller connection per process. No automatic restart or reconnect.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PORT="${ONBOARD_BODY_PORT:-5560}"
STATE="$REPO/.runtime/onboard_body"
STATUS="$REPO/.runtime/onboard_body_hold_status.json"
if [[ "$PORT" != 5560 ]]; then
    STATE="${STATE}_$PORT"
    STATUS="$REPO/.runtime/onboard_body_hold_status_$PORT.json"
fi
mkdir -p "$STATE"
PID_FILE="$STATE/pid"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case "${1:-status}" in
    start)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            if [[ -f "$STATUS" ]] && grep -q '"phase": "waiting"' "$STATUS"; then
                echo 'Reusing onboard receiver; no controller has connected yet.'
                exit 0
            fi
            echo 'Onboard body process already exists; stop the collection and run stop before restarting.' >&2
            exit 1
        fi
        [[ -x .venv_body/bin/python ]] || { echo 'Install the onboard body environment first.' >&2; exit 1; }
        args=(--port "$PORT")
        if [[ "${ONBOARD_BODY_READ_ONLY:-0}" == 1 ]]; then args+=(--read-only); fi
        nohup .venv_body/bin/python -u -m gear_sonic.scripts.run_onboard_body_hold "${args[@]}" \
            > "$STATE/service.log" 2>&1 < /dev/null &
        echo "$!" > "$PID_FILE"
        for _ in {1..100}; do
            if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
                cat "$STATE/service.log" >&2
                exit 1
            fi
            if grep -q '\[BodyHold\] ready' "$STATE/service.log"; then
                echo 'Onboard body receiver ready; no actuator commands until collection connects.'
                exit 0
            fi
            sleep 0.1
        done
        echo "Receiver did not become ready; inspect $STATE/service.log" >&2
        exit 1
        ;;
    stop)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            PID="$(cat "$PID_FILE")"
            [[ "$(tr '\0' ' ' < "/proc/$PID/cmdline")" == *gear_sonic.scripts.run_onboard_body_hold* ]] || exit 1
            kill -TERM "$PID"
            for _ in {1..100}; do
                if ! kill -0 "$PID" 2>/dev/null; then break; fi
                sleep 0.02
            done
            echo 'Stop requested: robot process sends damping before exiting.'
        fi
        ;;
    status)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            cat "$STATUS" 2>/dev/null || tail -3 "$STATE/service.log"
        else
            echo 'Onboard body process is stopped.'
        fi
        ;;
    *) echo "Usage: bash $0 {start|stop|status}" >&2; exit 2 ;;
esac
