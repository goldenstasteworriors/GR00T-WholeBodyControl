#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE=${COLLECT_CONFIG:-$SCRIPT_DIR/session.env}
SESSION_NAME=sonic_data_collection
DATA_PY="$REPO_ROOT/.venv_data_collection/bin/python"
TELEOP_PY="$REPO_ROOT/.venv_teleop/bin/python"
DATASET_GUARD="$SCRIPT_DIR/dataset_guard.py"

die() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}

pass() {
    printf '[PASS] %s\n' "$*"
}

note() {
    printf '[INFO] %s\n' "$*"
}

port_is_listening() {
    local port=$1
    ss -ltnH | awk '{print $4}' | grep -Eq ":${port}$"
}

ping_zero_loss() {
    local host=$1 output
    if ! output=$(ping -c 5 -i 0.2 -W 1 "$host" 2>&1); then
        printf '%s\n' "$output"
        return 1
    fi
    printf '%s\n' "$output"
    grep -Eq '(^|[[:space:]])0% packet loss' <<<"$output"
}

tcp_stable() {
    local host=$1 port=$2 attempt
    for attempt in 1 2 3 4 5; do
        if ! nc -z -w 2 "$host" "$port"; then
            printf 'TCP check %s/5 failed for %s:%s\n' "$attempt" "$host" "$port" >&2
            return 1
        fi
        sleep 0.2
    done
    printf 'TCP checks 5/5 passed for %s:%s\n' "$host" "$port"
}

usage() {
    cat <<'EOF'
SONIC G1 standalone collection controller

Usage:
  ./collectctl.sh [--config FILE] COMMAND [OPTIONS]

Commands:
  init              Create session.env from naming/task parameters
  config            Print the resolved session configuration
  pc-service        Start/check RoboticsService; --restart restarts it safely
  pico-check        Verify PICO body/timestamp stability for 12 seconds
  pico-start        Restart only the PICO manager inside an existing tmux stack
  preflight         Run all read-only launch gates
  launch            Launch the verified tmux stack (runs preflight first)
  attach            Attach to the tmux collection session
  status            Print a compact stack and recording status
  health            Require fresh G1, camera, PICO, hand, route, and process state
  verify-latest     Validate the newest saved parquet and ego-view MP4
  stop              Finish an already-stopped/saved collection safely
  remote-stop       Stop a stale G1 deploy after physical e-stop
  emergency-stop    Stop software after physical e-stop and delete discarded tail
  help              Show this help

Read README.md in this directory before the first real-robot run.
EOF
}

set_defaults() {
    DATASET_NAME=
    TASK_PROMPT=
    RUN_MODE=new
    HAND_MODE=right-only

    ROBOT_INTERFACE=enx00e04c36c4e8
    PC_ROBOT_IP=192.168.123.222
    G1_IP=192.168.123.164
    G1_USER=unitree
    G1_REPO_ROOT=/home/unitree/VLA/GR00T-WholeBodyControl
    G1_INTERFACE=eth0
    LEFT_HAND_IP=192.168.123.210
    RIGHT_HAND_IP=192.168.123.211
    HAND_PORT=6000
    INSPIRE_HAND_POSE_CONFIG="$SCRIPT_DIR/inspire_hand_pose.json"
    WIFI_INTERFACE=wlp0s20f3

    CAMERA_HTTP_URL=http://222.29.157.156:8765/stream.mjpg
    CAMERA_HOST=192.168.123.164
    CAMERA_PORT=5555
    CAMERA_ENABLED=true

    DEPLOY_CHECKPOINT=policy/sonic_v1_1/model
    DEPLOY_OBS_CONFIG=policy/sonic_v1_1/observation_config.yaml
    DEPLOY_PLANNER=planner/target_vel/V2/planner_sonic_trt85_static_einsum5d.onnx
    DEPLOY_MOTION_DATA=reference/example/
    DATA_FREQUENCY=50
    HAND_STATE_FREQUENCY=50
    SONIC_ZMQ_PORT=5556
    STATE_ZMQ_PORT=5557
    MAX_RUNTIME_S=3600
    STREAMING_TIMEOUT_S=5.0
    LOWSTATE_TIMEOUT_S=0.2
    AUDIO_CUE_VOLUME=0.6
    LATENCY_SAMPLE_INTERVAL=0.1
    LATENCY_HISTORY_SIZE=180
}

validate_config() {
    [[ "$CAMERA_ENABLED" == true || "$CAMERA_ENABLED" == false ]] \
        || die "CAMERA_ENABLED must be true or false"
    [[ "$DATASET_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || die "DATASET_NAME must match [A-Za-z0-9][A-Za-z0-9._-]*"
    [[ -n "$TASK_PROMPT" ]] || die "TASK_PROMPT is empty; run 'init' first"
    [[ "$RUN_MODE" == new || "$RUN_MODE" == resume ]] \
        || die "RUN_MODE must be new or resume"
    [[ "$HAND_MODE" == right-only || "$HAND_MODE" == left-only || "$HAND_MODE" == both ]] \
        || die "HAND_MODE must be right-only, left-only, or both"
    [[ -z "$CAMERA_HTTP_URL" || "$CAMERA_HTTP_URL" == http://* || "$CAMERA_HTTP_URL" == https://* ]] \
        || die "CAMERA_HTTP_URL must be empty or start with http:// or https://"
    [[ "$DATA_FREQUENCY" =~ ^[0-9]+$ ]] && (( DATA_FREQUENCY > 0 && DATA_FREQUENCY <= 100 )) \
        || die "DATA_FREQUENCY must be an integer in [1, 100]"
    [[ "$HAND_STATE_FREQUENCY" =~ ^[0-9]+$ ]] \
        && (( HAND_STATE_FREQUENCY > 0 && HAND_STATE_FREQUENCY <= 50 )) \
        || die "HAND_STATE_FREQUENCY must be an integer in [1, 50]"
    [[ -f "$INSPIRE_HAND_POSE_CONFIG" ]] \
        || die "missing Inspire hand pose config: $INSPIRE_HAND_POSE_CONFIG"
}

load_config() {
    set_defaults
    [[ -f "$CONFIG_FILE" ]] || die "missing config: $CONFIG_FILE (run './collectctl.sh init ...')"
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    validate_config
    DATASET_DIR="$REPO_ROOT/outputs/$DATASET_NAME"
    LOG_TAG=$(printf '%s' "$DATASET_NAME" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^[._]+|[._]+$//g')
    EXPORTER_LOG="$REPO_ROOT/logs/data_exporter_${LOG_TAG}.log"
    LATENCY_LOG="$REPO_ROOT/logs/latency_exporter_${LOG_TAG}.jsonl"
}

write_assignment() {
    local key=$1 value=$2
    printf '%s=%q\n' "$key" "$value"
}

cmd_init() {
    set_defaults
    local dataset= date_part action object run task hand_mode camera_url run_mode
    dataset=
    date_part=$(date +%m_%d)
    action=
    object=
    run=1
    task=
    hand_mode=right-only
    camera_url=$CAMERA_HTTP_URL
    run_mode=new

    while (($#)); do
        case "$1" in
            --dataset) dataset=${2:?missing value for --dataset}; shift 2 ;;
            --date) date_part=${2:?missing value for --date}; shift 2 ;;
            --action) action=${2:?missing value for --action}; shift 2 ;;
            --object) object=${2:?missing value for --object}; shift 2 ;;
            --run) run=${2:?missing value for --run}; shift 2 ;;
            --task) task=${2:?missing value for --task}; shift 2 ;;
            --hand-mode) hand_mode=${2:?missing value for --hand-mode}; shift 2 ;;
            --camera-url) camera_url=${2:?missing value for --camera-url}; shift 2 ;;
            --resume) run_mode=resume; shift ;;
            --new) run_mode=new; shift ;;
            *) die "unknown init option: $1" ;;
        esac
    done

    if [[ -z "$dataset" ]]; then
        [[ "$date_part" =~ ^[0-9]{2}_[0-9]{2}$ ]] \
            || die "--date must use MM_DD, for example 09_06"
        [[ "$action" =~ ^[A-Za-z0-9-]+$ ]] || die "--action is required when --dataset is omitted"
        [[ "$object" =~ ^[A-Za-z0-9-]+$ ]] || die "--object is required when --dataset is omitted"
        [[ "$run" =~ ^[1-9][0-9]*$ ]] || die "--run must be a positive integer"
        dataset="${date_part}_${action}_${object}_${run}"
    fi

    DATASET_NAME=$dataset
    TASK_PROMPT=$task
    RUN_MODE=$run_mode
    HAND_MODE=$hand_mode
    CAMERA_HTTP_URL=$camera_url
    validate_config

    if [[ "$RUN_MODE" == new && -e "$REPO_ROOT/outputs/$DATASET_NAME" ]]; then
        die "new dataset already exists: $REPO_ROOT/outputs/$DATASET_NAME; choose another --run or use --resume"
    fi
    if [[ "$RUN_MODE" == resume && ! -d "$REPO_ROOT/outputs/$DATASET_NAME" ]]; then
        die "resume dataset does not exist: $REPO_ROOT/outputs/$DATASET_NAME"
    fi

    mkdir -p "$(dirname -- "$CONFIG_FILE")"
    {
        printf '# Generated by tools/0902collect/collectctl.sh init\n'
        write_assignment DATASET_NAME "$DATASET_NAME"
        write_assignment TASK_PROMPT "$TASK_PROMPT"
        write_assignment RUN_MODE "$RUN_MODE"
        write_assignment HAND_MODE "$HAND_MODE"
        printf '\n'
        write_assignment ROBOT_INTERFACE "$ROBOT_INTERFACE"
        write_assignment PC_ROBOT_IP "$PC_ROBOT_IP"
        write_assignment G1_IP "$G1_IP"
        write_assignment G1_USER "$G1_USER"
        write_assignment G1_REPO_ROOT "$G1_REPO_ROOT"
        write_assignment G1_INTERFACE "$G1_INTERFACE"
        write_assignment LEFT_HAND_IP "$LEFT_HAND_IP"
        write_assignment RIGHT_HAND_IP "$RIGHT_HAND_IP"
        write_assignment HAND_PORT "$HAND_PORT"
        write_assignment INSPIRE_HAND_POSE_CONFIG "$INSPIRE_HAND_POSE_CONFIG"
        write_assignment WIFI_INTERFACE "$WIFI_INTERFACE"
        printf '\n'
        write_assignment CAMERA_HTTP_URL "$CAMERA_HTTP_URL"
        write_assignment CAMERA_HOST "$CAMERA_HOST"
        write_assignment CAMERA_PORT "$CAMERA_PORT"
        write_assignment CAMERA_ENABLED "$CAMERA_ENABLED"
        printf '\n'
        write_assignment DEPLOY_CHECKPOINT "$DEPLOY_CHECKPOINT"
        write_assignment DEPLOY_OBS_CONFIG "$DEPLOY_OBS_CONFIG"
        write_assignment DEPLOY_PLANNER "$DEPLOY_PLANNER"
        write_assignment DEPLOY_MOTION_DATA "$DEPLOY_MOTION_DATA"
        write_assignment DATA_FREQUENCY "$DATA_FREQUENCY"
        write_assignment HAND_STATE_FREQUENCY "$HAND_STATE_FREQUENCY"
        write_assignment SONIC_ZMQ_PORT "$SONIC_ZMQ_PORT"
        write_assignment STATE_ZMQ_PORT "$STATE_ZMQ_PORT"
        write_assignment MAX_RUNTIME_S "$MAX_RUNTIME_S"
        write_assignment STREAMING_TIMEOUT_S "$STREAMING_TIMEOUT_S"
        write_assignment LOWSTATE_TIMEOUT_S "$LOWSTATE_TIMEOUT_S"
        write_assignment AUDIO_CUE_VOLUME "$AUDIO_CUE_VOLUME"
        write_assignment LATENCY_SAMPLE_INTERVAL "$LATENCY_SAMPLE_INTERVAL"
        write_assignment LATENCY_HISTORY_SIZE "$LATENCY_HISTORY_SIZE"
    } >"$CONFIG_FILE"

    pass "created $CONFIG_FILE"
    note "dataset=$DATASET_NAME mode=$RUN_MODE hand=$HAND_MODE"
    note "next: ./collectctl.sh config"
}

cmd_config() {
    load_config
    local wifi_ip
    wifi_ip=$(ip -br -4 addr show "$WIFI_INTERFACE" 2>/dev/null | awk 'NR==1 {sub(/\/.*/,"",$3); print $3}')
    cat <<EOF
dataset              $DATASET_NAME
task prompt          $TASK_PROMPT
collection mode      $RUN_MODE
hand mode            $HAND_MODE
robot LAN            $ROBOT_INTERFACE / $PC_ROBOT_IP
G1                   $G1_USER@$G1_IP ($G1_INTERFACE)
left/right hand      $LEFT_HAND_IP / $RIGHT_HAND_IP:$HAND_PORT
hand pose config     $INSPIRE_HAND_POSE_CONFIG
camera               ${CAMERA_HTTP_URL:-$CAMERA_HOST:$CAMERA_PORT}
camera enabled       $CAMERA_ENABLED
PICO PC Service IP   ${wifi_ip:-UNAVAILABLE}
checkpoint           $DEPLOY_CHECKPOINT
observation config   $DEPLOY_OBS_CONFIG
planner              $DEPLOY_PLANNER
data/hand frequency  $DATA_FREQUENCY / $HAND_STATE_FREQUENCY Hz
ZMQ pose/state       $SONIC_ZMQ_PORT / $STATE_ZMQ_PORT
runtime/timeout      $MAX_RUNTIME_S / $STREAMING_TIMEOUT_S / $LOWSTATE_TIMEOUT_S s
config file          $CONFIG_FILE
EOF
    hand_pose_config_check
}

cmd_pc_service() {
    load_config
    local wifi_ip restart=false service_pid
    local service_pids=()
    if [[ ${1:-} == --restart ]]; then
        restart=true
        shift
    fi
    (($# == 0)) || die "unknown pc-service option: $1"

    if [[ "$restart" == true ]]; then
        pgrep -f '[p]ico_manager_thread_server.py' >/dev/null \
            && die "PICO manager is running; stop the policy/manager before restarting PC Service"
        mapfile -t service_pids < <(pgrep -f '^\./RoboticsServiceProcess$' || true)
        ((${#service_pids[@]} <= 1)) || die "multiple RoboticsService processes are running"
        if ((${#service_pids[@]} == 1)); then
            service_pid=${service_pids[0]}
            kill -TERM "$service_pid"
            for _ in {1..40}; do
                kill -0 "$service_pid" 2>/dev/null || break
                sleep 0.25
            done
            kill -0 "$service_pid" 2>/dev/null \
                && die "RoboticsService PID $service_pid did not stop after SIGTERM"
            pass "old RoboticsService stopped"
        fi
    fi

    if pgrep -f '[R]oboticsServiceProcess' >/dev/null; then
        pass "RoboticsService is already running: $(pgrep -af '[R]oboticsServiceProcess' | head -n 1)"
    else
        [[ -x /opt/apps/roboticsservice/RoboticsServiceProcess ]] \
            || die "/opt/apps/roboticsservice/RoboticsServiceProcess is missing"
        (cd /opt/apps/roboticsservice && setsid ./RoboticsServiceProcess >/tmp/robotics_service.log 2>&1 </dev/null &)
        sleep 3
        pgrep -f '[R]oboticsServiceProcess' >/dev/null || die "RoboticsService failed; inspect /tmp/robotics_service.log"
        pass "RoboticsService started"
    fi
    port_is_listening 60061 || die "RoboticsService SDK port 60061 is not listening"
    port_is_listening 63901 || die "RoboticsService PICO port 63901 is not listening"
    wifi_ip=$(ip -br -4 addr show "$WIFI_INTERFACE" 2>/dev/null | awk 'NR==1 {sub(/\/.*/,"",$3); print $3}')
    [[ -n "$wifi_ip" ]] || die "no IPv4 address on $WIFI_INTERFACE"
    pass "set PICO PC Service to $wifi_ip, recalibrate, then press Send"
}

cmd_pico_check() {
    load_config
    [[ -x "$TELEOP_PY" ]] || die "missing $TELEOP_PY"
    if pgrep -f '[p]ico_manager_thread_server.py' >/dev/null; then
        die "PICO manager is already running; use 'status' instead of opening a second SDK client"
    fi
    if ! timeout --signal=TERM 25s "$TELEOP_PY" - <<'PY'
import time
import xrobotoolkit_sdk as x

x.init()
try:
    time.sleep(2.0)
    trackers = int(x.num_motion_data_available())
    timestamp_1 = previous = int(x.get_time_stamp_ns())
    started = last_change = time.monotonic()
    max_stale_s = 0.0
    body_ok = True
    while time.monotonic() - started < 12.0:
        time.sleep(0.1)
        now = time.monotonic()
        body_ok = body_ok and bool(x.is_body_data_available())
        current = int(x.get_time_stamp_ns())
        if current > previous:
            last_change = now
        max_stale_s = max(max_stale_s, now - last_change)
        previous = current
    timestamp_2 = previous
    advanced = timestamp_2 > timestamp_1
    print(
        f"body={body_ok} trackers={trackers} duration=12s "
        f"timestamp1={timestamp_1} timestamp2={timestamp_2} "
        f"advanced={advanced} max_timestamp_stale_ms={max_stale_s * 1000.0:.1f} "
        "max_allowed_stale_ms=800"
    )
    if not (body_ok and advanced and max_stale_s <= 0.80):
        raise SystemExit(1)
finally:
    x.close()
PY
    then
        printf '[FAIL] PICO body data is unavailable, timestamp is not advancing, or a timestamp stall exceeds 800 ms\n' >&2
        return 1
    fi
    pass "PICO body data is live"
}

camera_check() {
    if [[ -n "$CAMERA_HTTP_URL" ]]; then
        CAMERA_CHECK_URL=$CAMERA_HTTP_URL "$DATA_PY" - <<'PY'
import os
import time
import requests

url = os.environ["CAMERA_CHECK_URL"]
session = requests.Session()
session.trust_env = False
start = time.monotonic()
frames = 0
tail = b""
with session.get(url, stream=True, timeout=(2, 5)) as response:
    response.raise_for_status()
    for chunk in response.iter_content(65536):
        if not chunk:
            continue
        data = tail + chunk
        frames += data.count(b"\xff\xd8")
        tail = data[-1:]
        if time.monotonic() - start >= 3.0:
            break
elapsed = time.monotonic() - start
fps = frames / elapsed
print(f"HTTP camera frames={frames} elapsed={elapsed:.2f}s fps={fps:.2f}")
if fps < 15.0:
    raise SystemExit(1)
PY
    else
        nc -z -w 2 "$CAMERA_HOST" "$CAMERA_PORT"
    fi
}

hand_pose_config_check() {
    INSPIRE_HAND_POSE_CONFIG="$INSPIRE_HAND_POSE_CONFIG" "$DATA_PY" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["INSPIRE_HAND_POSE_CONFIG"])
with path.open("r", encoding="utf-8") as stream:
    data = json.load(stream)
if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a JSON object")

profiles = {}
for name in ("open", "grasp"):
    values = data.get(name)
    if not isinstance(values, list) or len(values) != 6:
        raise SystemExit(f"{path}: {name} must contain exactly 6 values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise SystemExit(f"{path}: {name} values must be numbers")
    if any(not 0.0 <= float(value) <= 1.0 for value in values):
        raise SystemExit(f"{path}: {name} values must be in [0.0, 1.0]")
    profiles[name] = [float(value) for value in values]

thumb = data.get("thumb_rotate_default")
if thumb is not None:
    if isinstance(thumb, bool) or not isinstance(thumb, (int, float)):
        raise SystemExit(f"{path}: thumb_rotate_default must be a number or null")
    thumb = float(thumb)
    if not 0.0 <= thumb <= 1.0:
        raise SystemExit(f"{path}: thumb_rotate_default must be in [0.0, 1.0]")

thumb_text = "profile" if thumb is None else f"fixed={thumb:.3f}"
print(
    f"hand pose open={profiles['open']} grasp={profiles['grasp']} "
    f"thumb_rotate={thumb_text}"
)
PY
}

remote_model_check() {
    local remote_deploy="$G1_REPO_ROOT/gear_sonic_deploy"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$G1_USER@$G1_IP" \
        "test -x '$remote_deploy/target/release/g1_deploy_onnx_ref' && test -f '$remote_deploy/${DEPLOY_CHECKPOINT}_decoder.onnx' && test -f '$remote_deploy/${DEPLOY_CHECKPOINT}_encoder.onnx' && test -f '$remote_deploy/$DEPLOY_OBS_CONFIG' && test -f '$remote_deploy/$DEPLOY_PLANNER'"
}

build_pico_manager_command() {
    printf -v PICO_MANAGER_COMMAND \
        'cd %q && source .venv_teleop/bin/activate && python gear_sonic/scripts/pico_manager_thread_server.py --input-source xrt --zmq_feedback_host %q --zmq_feedback_port %q --manager --vis_vr3pt --vis_smpl --slow_frame_log %q --slow_frame_threshold_ms 40.0 --stall_stack_threshold_ms 100.0' \
        "$REPO_ROOT" \
        "$G1_IP" \
        "$STATE_ZMQ_PORT" \
        "$REPO_ROOT/logs/pico_slow_frames_${LOG_TAG}.jsonl"
}

cmd_pico_start() {
    load_config
    tmux has-session -t "$SESSION_NAME" 2>/dev/null \
        || die "tmux session $SESSION_NAME does not exist; use launch instead"
    pgrep -f '[p]ico_manager_thread_server.py' >/dev/null \
        && die "PICO manager is already running"
    [[ $("$DATA_PY" "$DATASET_GUARD" recording --latency-log "$LATENCY_LOG" --value-only) == false ]] \
        || die "recording must be stopped before restarting the PICO manager"

    cmd_pico_check
    build_pico_manager_command
    tmux send-keys -t "$SESSION_NAME:data_collection.1" C-l
    tmux clear-history -t "$SESSION_NAME:data_collection.1" 2>/dev/null || true
    tmux send-keys -t "$SESSION_NAME:data_collection.1" "$PICO_MANAGER_COMMAND" C-m
    sleep 15
    pgrep -f '[p]ico_manager_thread_server.py' >/dev/null \
        || die "PICO manager failed to stay running"
    local recent
    recent=$(tmux capture-pane -p -t "$SESSION_NAME:data_collection.1" -S -60 | tail -n 60)
    grep -q 'Body visualization:' <<<"$recent" \
        || die "PICO manager is running but OFF preview is not ready; inspect pane 1"
    pass "PICO manager restored in robot control OFF preview"
}

run_dataset_gate() {
    local mode=$1
    local args=(
        check
        --dataset-dir "$DATASET_DIR"
        --mode "$mode"
        --task "$TASK_PROMPT"
        --fps "$DATA_FREQUENCY"
        --hand-mode "$HAND_MODE"
        --hand-pose-config "$INSPIRE_HAND_POSE_CONFIG"
        --checkpoint "$DEPLOY_CHECKPOINT"
        --obs-config "$DEPLOY_OBS_CONFIG"
        --planner "$DEPLOY_PLANNER"
        --motion-data "$DEPLOY_MOTION_DATA"
    )
    args+=(--camera-mode "$CAMERA_ENABLED")
    if [[ "$CAMERA_ENABLED" == true && -n "$CAMERA_HTTP_URL" ]]; then
        args+=(--camera-http-url "$CAMERA_HTTP_URL")
    fi
    "$DATA_PY" "$DATASET_GUARD" "${args[@]}"
}

cmd_preflight() {
    load_config
    local failures=0

    check_gate() {
        local label=$1
        shift
        if "$@"; then
            pass "$label"
        else
            printf '[FAIL] %s\n' "$label" >&2
            failures=$((failures + 1))
        fi
    }

    check_gate "repository and virtual environments" test -x "$DATA_PY"
    check_gate "teleop virtual environment" test -x "$TELEOP_PY"
    check_gate "Inspire hand pose config" hand_pose_config_check
    check_gate "tmux installed" command -v tmux
    check_gate "timeout command installed" command -v timeout
    check_gate "$ROBOT_INTERFACE carrier" bash -c "ip -br link show '$ROBOT_INTERFACE' | grep -q 'LOWER_UP'"
    check_gate "$ROBOT_INTERFACE has $PC_ROBOT_IP" bash -c "ip -br -4 addr show '$ROBOT_INTERFACE' | grep -q '$PC_ROBOT_IP/'"
    check_gate "route to G1 uses $ROBOT_INTERFACE" bash -c "ip route get '$G1_IP' | grep -q 'dev $ROBOT_INTERFACE'"
    check_gate "G1 zero-loss ping at $G1_IP" ping_zero_loss "$G1_IP"
    check_gate "G1 SSH key login" ssh -o BatchMode=yes -o ConnectTimeout=5 "$G1_USER@$G1_IP" true
    check_gate "G1 deploy models exist" remote_model_check

    if [[ "$HAND_MODE" == left-only || "$HAND_MODE" == both ]]; then
        check_gate "route to left hand uses $ROBOT_INTERFACE" bash -c "ip route get '$LEFT_HAND_IP' | grep -q 'dev $ROBOT_INTERFACE'"
        check_gate "left hand zero-loss ping" ping_zero_loss "$LEFT_HAND_IP"
        check_gate "left hand Modbus $HAND_PORT stable" tcp_stable "$LEFT_HAND_IP" "$HAND_PORT"
    fi
    if [[ "$HAND_MODE" == right-only || "$HAND_MODE" == both ]]; then
        check_gate "route to right hand uses $ROBOT_INTERFACE" bash -c "ip route get '$RIGHT_HAND_IP' | grep -q 'dev $ROBOT_INTERFACE'"
        check_gate "right hand zero-loss ping" ping_zero_loss "$RIGHT_HAND_IP"
        check_gate "right hand Modbus $HAND_PORT stable" tcp_stable "$RIGHT_HAND_IP" "$HAND_PORT"
    fi

    if [[ "$CAMERA_ENABLED" == true ]]; then
        check_gate "camera source" camera_check
    else
        note "camera disabled: no camera check or video recording"
    fi
    check_gate "RoboticsService running" pgrep -f '[R]oboticsServiceProcess'
    check_gate "RoboticsService SDK port 60061" port_is_listening 60061
    check_gate "RoboticsService PICO port 63901" port_is_listening 63901

    if pgrep -f '[p]ico_manager_thread_server.py' >/dev/null; then
        printf '[FAIL] PICO manager is already running\n' >&2
        failures=$((failures + 1))
    else
        if cmd_pico_check; then
            pass "PICO launch gate"
        else
            printf '[FAIL] PICO launch gate\n' >&2
            failures=$((failures + 1))
        fi
    fi

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        printf '[FAIL] tmux session $SESSION_NAME already exists; stop the old session first\n' >&2
        failures=$((failures + 1))
    else
        pass "no existing tmux collection session"
    fi
    if pgrep -f '[r]un_data_exporter.py|[r]un_camera_viewer.py|[i]nspire_modbus_hand.py' >/dev/null; then
        printf '[FAIL] local collection process already exists\n' >&2
        failures=$((failures + 1))
    else
        pass "no conflicting local collection process"
    fi
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$G1_USER@$G1_IP" "pgrep -f '^./target/release/g1_deploy_onnx_ref '" >/dev/null 2>&1; then
        printf '[FAIL] G1 deploy process already exists\n' >&2
        failures=$((failures + 1))
    else
        pass "no existing G1 deploy process"
    fi

    if run_dataset_gate "$RUN_MODE"; then
        pass "dataset gate"
    else
        printf '[FAIL] dataset gate\n' >&2
        failures=$((failures + 1))
    fi

    ((failures == 0)) || die "preflight failed: $failures gate(s)"
    pass "all launch gates passed"
    note "physical gate: workspace clear, emergency stop reachable, robot stable"
}

build_launcher_command() {
    LAUNCH_CMD=(
        "$DATA_PY" "$REPO_ROOT/gear_sonic/scripts/launch_data_collection.py"
        --deploy-onboard
        --deploy-onboard-host "$G1_IP"
        --deploy-onboard-user "$G1_USER"
        --deploy-onboard-repo-root "$G1_REPO_ROOT"
        --deploy-onboard-interface "$G1_INTERFACE"
        --offboard-zmq-host "$PC_ROBOT_IP"
        --deploy-checkpoint "$DEPLOY_CHECKPOINT"
        --deploy-obs-config "$DEPLOY_OBS_CONFIG"
        --deploy-planner "$DEPLOY_PLANNER"
        --deploy-motion-data "$DEPLOY_MOTION_DATA"
        --enable-sonic-publish
        --sonic-publish-max-runtime-s "$MAX_RUNTIME_S"
        --sonic-streaming-timeout-s "$STREAMING_TIMEOUT_S"
        --sonic-lowstate-timeout-s "$LOWSTATE_TIMEOUT_S"
        --camera-host "$CAMERA_HOST"
        --camera-port "$CAMERA_PORT"
        --task-prompt "$TASK_PROMPT"
        --dataset-name "$DATASET_NAME"
        --data-exporter-frequency "$DATA_FREQUENCY"
        --sonic-zmq-port "$SONIC_ZMQ_PORT"
        --state-zmq-host "$G1_IP"
        --state-zmq-port "$STATE_ZMQ_PORT"
        --inspire-hand-bridge
        --inspire-hand-network "$ROBOT_INTERFACE"
        --inspire-left-ip "$LEFT_HAND_IP"
        --inspire-right-ip "$RIGHT_HAND_IP"
        --inspire-hand-state-frequency "$HAND_STATE_FREQUENCY"
        --inspire-hand-pose-config "$INSPIRE_HAND_POSE_CONFIG"
        --audio-cue-volume "$AUDIO_CUE_VOLUME"
        --latency-sample-interval "$LATENCY_SAMPLE_INTERVAL"
        --latency-history-size "$LATENCY_HISTORY_SIZE"
    )
    if [[ "$CAMERA_ENABLED" == false ]]; then
        LAUNCH_CMD+=(--no-camera --no-camera-viewer)
    elif [[ -n "$CAMERA_HTTP_URL" ]]; then
        LAUNCH_CMD+=(--camera-http-url "$CAMERA_HTTP_URL")
    fi
    case "$HAND_MODE" in
        right-only) LAUNCH_CMD+=(--right-hand-only) ;;
        left-only) LAUNCH_CMD+=(--left-hand-only) ;;
        both) ;;
    esac
}

cmd_launch() {
    load_config
    local dry_run=false
    if [[ ${1:-} == --dry-run ]]; then
        dry_run=true
        shift
    fi
    (($# == 0)) || die "unknown launch option: $1"
    build_launcher_command
    if [[ "$dry_run" == true ]]; then
        printf 'cd %q\n' "$REPO_ROOT"
        printf '%q ' "${LAUNCH_CMD[@]}"
        printf '\n'
        return
    fi
    cmd_preflight
    cat <<EOF

[MANUAL SAFETY GATE]
  1. Keep the physical emergency stop in reach.
  2. Confirm the robot is stable and the workspace is clear.
  3. Pane 0 will ask: Proceed with REAL robot deployment? [Y/n]
  4. Press Enter there only after checking the other panes.

EOF
    cd "$REPO_ROOT"
    "${LAUNCH_CMD[@]}"
}

cmd_attach() {
    load_config
    tmux has-session -t "$SESSION_NAME" 2>/dev/null || die "tmux session $SESSION_NAME does not exist"
    exec tmux attach -t "$SESSION_NAME"
}

capture_markers() {
    local target=$1 pattern=$2
    tmux capture-pane -p -t "$target" -S -160 2>/dev/null | grep -E "$pattern" | tail -n 12 || true
}

cmd_status() {
    load_config
    printf '%s\n' 'Processes:'
    pgrep -af '[p]ico_manager_thread_server.py|[r]un_data_exporter.py|[r]un_camera_viewer.py|[i]nspire_modbus_hand.py' || true
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$G1_USER@$G1_IP" "pgrep -af '^./target/release/g1_deploy_onnx_ref ' || true" 2>/dev/null || true

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        printf '\nTmux panes:\n'
        tmux list-panes -a -F '#S:#I.#P cmd=#{pane_current_command} dead=#{pane_dead}' | grep "^$SESSION_NAME:" || true
        printf '\nManager markers:\n'
        capture_markers "$SESSION_NAME:data_collection.1" 'Body visualization|Calibration completed|StreamMode switch|PicoReader|timestamp.*paused|timestamp resumed|Shutdown'
        printf '\nExporter markers:\n'
        capture_markers "$SESSION_NAME:data_collection.2" '开始记录|停止记录|Finished saving|丢弃数据|Waiting for message|HTTP MJPEG.*failed'
        printf '\nHand errors (empty is good):\n'
        capture_markers "$SESSION_NAME:inspire_hand.0" 'Connection refused|No route to host|Modbus.*failed|timeout|interface.*not found|ddsi_udp_conn_write.*failed'
    else
        printf '\nTmux session: absent\n'
    fi

    printf '\nRecording state:\n'
    if pgrep -f '[r]un_data_exporter.py' >/dev/null; then
        "$DATA_PY" "$DATASET_GUARD" recording --latency-log "$LATENCY_LOG"
    else
        note "exporter is not running; last logged recording state is historical, not live"
    fi
    printf '\nDataset state:\n'
    run_dataset_gate resume || true
}

cmd_health() {
    load_config
    local failures=0 hand_side hand_errors

    health_gate() {
        local label=$1
        shift
        if "$@"; then
            pass "$label"
        else
            printf '[FAIL] %s\n' "$label" >&2
            failures=$((failures + 1))
        fi
    }

    all_collection_panes_live() {
        local panes
        panes=$(tmux list-panes -a -F '#S:#{pane_dead}' 2>/dev/null | grep "^$SESSION_NAME:" || true)
        [[ -n "$panes" ]] && ! grep -q ':1$' <<<"$panes"
    }

    tmux_session_exists() {
        tmux has-session -t "$SESSION_NAME" 2>/dev/null
    }

    process_exists() {
        pgrep -f "$1" >/dev/null
    }

    remote_deploy_exists() {
        ssh -o BatchMode=yes -o ConnectTimeout=3 "$G1_USER@$G1_IP" \
            "pgrep -f '^./target/release/g1_deploy_onnx_ref '" >/dev/null 2>&1
    }

    health_gate "tmux session exists" tmux_session_exists
    health_gate "all collection panes are live" all_collection_panes_live
    health_gate "route to G1 uses $ROBOT_INTERFACE" bash -c "ip route get '$G1_IP' | grep -q 'dev $ROBOT_INTERFACE'"
    health_gate "G1 zero-loss ping" ping_zero_loss "$G1_IP"
    health_gate "PICO manager process" process_exists '[p]ico_manager_thread_server.py'
    health_gate "data exporter process" process_exists '[r]un_data_exporter.py'
    if [[ "$CAMERA_ENABLED" == true ]]; then
        health_gate "camera viewer process" process_exists '[r]un_camera_viewer.py'
    fi
    health_gate "Inspire hand bridge process" process_exists '[i]nspire_modbus_hand.py'
    health_gate "RoboticsService process" process_exists '[R]oboticsServiceProcess'
    health_gate "G1 deploy process" remote_deploy_exists

    hand_side=both
    [[ "$HAND_MODE" == right-only ]] && hand_side=right
    [[ "$HAND_MODE" == left-only ]] && hand_side=left
    if pgrep -af '[i]nspire_modbus_hand.py' | grep -F -- "--side $hand_side" >/dev/null; then
        pass "Inspire bridge side is $hand_side"
    else
        printf '[FAIL] Inspire bridge is not running with --side %s\n' "$hand_side" >&2
        failures=$((failures + 1))
    fi

    if [[ "$HAND_MODE" == right-only || "$HAND_MODE" == both ]]; then
        health_gate "route to right hand uses $ROBOT_INTERFACE" bash -c "ip route get '$RIGHT_HAND_IP' | grep -q 'dev $ROBOT_INTERFACE'"
    fi
    if [[ "$HAND_MODE" == left-only || "$HAND_MODE" == both ]]; then
        health_gate "route to left hand uses $ROBOT_INTERFACE" bash -c "ip route get '$LEFT_HAND_IP' | grep -q 'dev $ROBOT_INTERFACE'"
    fi

    hand_errors=$(tmux capture-pane -p -t "$SESSION_NAME:inspire_hand.0" -S -80 2>/dev/null \
        | grep -Ei 'Connection refused|No route|Modbus.*(fail|timeout)|timed out|interface.*not found|ddsi_udp_conn_write.*failed' \
        | tail -n 10 || true)
    if [[ -n "$hand_errors" ]]; then
        printf '%s\n' "$hand_errors" >&2
        printf '[FAIL] recent Inspire hand errors detected\n' >&2
        failures=$((failures + 1))
    else
        pass "no recent Inspire hand errors"
    fi

    local camera_health_args=()
    [[ "$CAMERA_ENABLED" == true ]] || camera_health_args+=(--no-camera)
    if "$DATA_PY" "$DATASET_GUARD" health --latency-log "$LATENCY_LOG" "${camera_health_args[@]}"; then
        pass "live data freshness gate"
    else
        printf '[FAIL] live data freshness gate\n' >&2
        failures=$((failures + 1))
    fi

    ((failures == 0)) || die "health check failed: $failures gate(s); do not start/continue recording"
    pass "collection is healthy; recording may start or continue"
}

cmd_verify_latest() {
    load_config
    "$DATA_PY" "$DATASET_GUARD" latest --dataset-dir "$DATASET_DIR"
}

stop_remote_deploy() {
    tmux send-keys -t "$SESSION_NAME:data_collection.0" C-c 2>/dev/null || true
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$G1_USER@$G1_IP" \
        "pkill -INT -f '^./target/release/g1_deploy_onnx_ref ' || true" 2>/dev/null || true
}

stop_tmux_pane() {
    local target=$1
    tmux send-keys -t "$target" C-c 2>/dev/null || true
}

wait_for_exporter() {
    local max_seconds=$1
    local loops=$((max_seconds * 2))
    local i
    for ((i = 0; i < loops; i++)); do
        if ! pgrep -f '[r]un_data_exporter.py' >/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

require_exporter_matches_config() {
    local pids=() pid arg expect_value=false active_dataset=
    mapfile -t pids < <(pgrep -f '[r]un_data_exporter.py' || true)
    ((${#pids[@]} <= 1)) || die "multiple data exporters are running; stop and inspect them manually"
    ((${#pids[@]} == 1)) || return 0

    pid=${pids[0]}
    while IFS= read -r -d '' arg; do
        if [[ "$expect_value" == true ]]; then
            active_dataset=$arg
            break
        fi
        if [[ "$arg" == --dataset-name ]]; then
            expect_value=true
        fi
    done <"/proc/$pid/cmdline"

    [[ -n "$active_dataset" ]] || die "cannot identify dataset name for exporter PID $pid"
    [[ "$active_dataset" == "$DATASET_NAME" ]] \
        || die "config dataset=$DATASET_NAME but active exporter dataset=$active_dataset; refusing cleanup"
}

cmd_stop() {
    load_config
    local off_preview=false recording_state
    if [[ ${1:-} == --off-preview-confirmed ]]; then
        off_preview=true
        shift
    fi
    (($# == 0)) || die "unknown stop option: $1"

    require_exporter_matches_config

    if pgrep -f '[r]un_data_exporter.py' >/dev/null; then
        recording_state=$("$DATA_PY" "$DATASET_GUARD" recording --latency-log "$LATENCY_LOG" --value-only)
        case "$recording_state" in
            true)
                die "recording is active; press Left Grip + right A once, wait for Finished saving, then rerun stop"
                ;;
            false) ;;
            *)
                die "recording state is unknown; run 'status' and confirm the exporter log before stopping"
                ;;
        esac
    else
        note "exporter is not running; checking saved data instead of historical recording state"
        "$DATA_PY" "$DATASET_GUARD" check --dataset-dir "$DATASET_DIR" --mode resume \
            || die "saved dataset needs inspection; preserving all files and logs"
        note "saved dataset is consistent; this does not recover an interrupted in-memory episode"
    fi
    if pgrep -f '[p]ico_manager_thread_server.py' >/dev/null; then
        if [[ "$off_preview" != true ]]; then
            die "PICO manager is still running. If policy is active, press A+B+X+Y and release. If it never left OFF preview, rerun: stop --off-preview-confirmed"
        fi
        stop_tmux_pane "$SESSION_NAME:data_collection.1"
        pkill -INT -f '[p]ico_manager_thread_server.py' 2>/dev/null || true
        sleep 2
    fi

    stop_remote_deploy
    stop_tmux_pane "$SESSION_NAME:data_collection.2"
    stop_tmux_pane "$SESSION_NAME:data_collection.3"
    stop_tmux_pane "$SESSION_NAME:inspire_hand.0"
    sleep 3
    wait_for_exporter 60 || die "exporter is still finishing; do not kill it; rerun status"
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$G1_USER@$G1_IP" \
        "pgrep -f '^./target/release/g1_deploy_onnx_ref ' >/dev/null; test \$? -eq 1" \
        || die "local cleanup finished, but G1 deploy shutdown is unconfirmed; preserve physical e-stop and check G1 connectivity/processes"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    pass "collection stack stopped after saved/idle verification"
}

cmd_remote_stop() {
    load_config
    [[ ${1:-} == --physical-estop-pressed ]] \
        || die "press the physical emergency stop first, then pass --physical-estop-pressed"
    shift
    (($# == 0)) || die "unknown remote-stop option: $1"
    ip route get "$G1_IP" | grep -q "dev $ROBOT_INTERFACE" \
        || die "route to G1 does not use $ROBOT_INTERFACE"
    ping_zero_loss "$G1_IP" || die "G1 network is not stable"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$G1_USER@$G1_IP" \
        "pkill -INT -f '^./target/release/g1_deploy_onnx_ref ' || true"
    for _ in {1..40}; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=3 "$G1_USER@$G1_IP" \
            "pgrep -f '^./target/release/g1_deploy_onnx_ref '" >/dev/null 2>&1; then
            pass "no G1 deploy process remains"
            return 0
        fi
        sleep 0.25
    done
    die "G1 deploy process is still running"
}

cmd_emergency_stop() {
    load_config
    local estop=false delete_active=false
    while (($#)); do
        case "$1" in
            --physical-estop-pressed) estop=true; shift ;;
            --delete-active) delete_active=true; shift ;;
            *) die "unknown emergency-stop option: $1" ;;
        esac
    done
    [[ "$estop" == true ]] || die "press the physical emergency stop first, then pass --physical-estop-pressed"
    [[ "$delete_active" == true ]] || die "pass --delete-active to authorize deleting the current incomplete episode"

    note "stopping PICO manager, hand bridge, and G1 deploy"
    stop_tmux_pane "$SESSION_NAME:data_collection.1"
    pkill -INT -f '[p]ico_manager_thread_server.py' 2>/dev/null || true
    stop_tmux_pane "$SESSION_NAME:inspire_hand.0"
    pkill -INT -f '[i]nspire_modbus_hand.py' 2>/dev/null || true
    stop_remote_deploy
    sleep 2

    if pgrep -f '[r]un_data_exporter.py' >/dev/null; then
        require_exporter_matches_config
        note "asking exporter to save the active buffer as discarded; this may take a minute"
        stop_tmux_pane "$SESSION_NAME:data_collection.2"
        pkill -INT -f '[r]un_data_exporter.py' 2>/dev/null || true
        wait_for_exporter 120 || die "exporter is still finishing; do not SIGKILL it; rerun status later"
    fi

    "$DATA_PY" "$DATASET_GUARD" prune-discarded-tail \
        --dataset-dir "$DATASET_DIR" --confirm DELETE_DISCARDED_TAIL
    stop_tmux_pane "$SESSION_NAME:data_collection.3"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    pass "software emergency stop complete; run the full preflight before restarting"
}

while [[ ${1:-} == --config ]]; do
    [[ $# -ge 2 ]] || die "--config needs a file path"
    CONFIG_FILE=$2
    shift 2
done

command_name=${1:-help}
if (($#)); then
    shift
fi

case "$command_name" in
    init) cmd_init "$@" ;;
    config) cmd_config "$@" ;;
    pc-service) cmd_pc_service "$@" ;;
    pico-check) cmd_pico_check "$@" ;;
    pico-start) cmd_pico_start "$@" ;;
    preflight) cmd_preflight "$@" ;;
    launch) cmd_launch "$@" ;;
    attach) cmd_attach "$@" ;;
    status) cmd_status "$@" ;;
    health) cmd_health "$@" ;;
    verify-latest) cmd_verify_latest "$@" ;;
    stop) cmd_stop "$@" ;;
    remote-stop) cmd_remote_stop "$@" ;;
    emergency-stop) cmd_emergency_stop "$@" ;;
    help|-h|--help) usage ;;
    *) die "unknown command: $command_name (run './collectctl.sh help')" ;;
esac
