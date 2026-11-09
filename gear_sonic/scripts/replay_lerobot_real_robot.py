"""Replay recorded G1 joint trajectories through onboard SONIC on the real robot.

This launcher is intentionally gated by two interactive Enter confirmations:

1. Start the onboard deploy, wait for its INIT/default-pose ramp, feed and hold
   the first recorded G1 pose through encoder mode 0, then enter SONIC CONTROL.
2. Release the full recorded trajectory. No hand command is ever transmitted.

Run ``--validate-only`` for data/asset checks, or ``--dry-run`` to exercise the
same tmux layout and both confirmations without launching the deploy binary or
publishing robot commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "outputs" / "8_16_dun"
DEFAULT_SESSION = "sonic_real_replay"
SONIC_PUBLISH_CONFIRMATION = "I_ACKNOWLEDGE_SONIC_CAN_MOVE_G1"

G1_ISAACLAB_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


def _bootstrap_environment() -> None:
    try:
        import pandas  # noqa: F401
        import zmq  # noqa: F401

        return
    except ImportError as exc:
        env_python = REPO_ROOT / ".venv_data_collection" / "bin" / "python"
        if env_python.exists() and Path(sys.executable).resolve() != env_python.resolve():
            os.execv(
                str(env_python),
                [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        raise SystemExit(
            "ERROR: .venv_data_collection is missing pandas/pyzmq. "
            "Run bash install_scripts/install_data_collection.sh"
        ) from exc


_bootstrap_environment()

import numpy as np
import pandas as pd


@dataclass
class ReplayData:
    dataset_path: Path
    parquet_path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay LeRobot G1 body joints through onboard SONIC on the real robot."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--joint-column",
        choices=("observation.state", "action.wbc"),
        default="observation.state",
    )
    parser.add_argument(
        "--velocity-source",
        choices=("finite_difference", "zeros"),
        default="finite_difference",
    )

    parser.add_argument("--onboard-host", default="192.168.123.164")
    parser.add_argument("--onboard-user", default="unitree")
    parser.add_argument(
        "--onboard-repo-root",
        default="/home/unitree/VLA/GR00T-WholeBodyControl",
    )
    parser.add_argument("--onboard-interface", default="eth0")
    parser.add_argument("--onboard-binary", default="target/release/g1_deploy_onnx_ref")
    parser.add_argument("--offboard-zmq-host", default="192.168.123.222")
    parser.add_argument("--zmq-bind", default="tcp://*:5556")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--deploy-checkpoint", default="policy/sonic_v1_1/model")
    parser.add_argument(
        "--deploy-obs-config",
        default="policy/sonic_v1_1/observation_config.yaml",
    )
    parser.add_argument(
        "--deploy-planner",
        default="planner/target_vel/V2/planner_sonic_trt85_static_einsum5d.onnx",
    )
    parser.add_argument("--deploy-motion-data", default="reference/example/")
    parser.add_argument("--deploy-output-type", default="all")
    parser.add_argument("--publish-max-runtime-s", type=float, default=3600.0)
    parser.add_argument("--streaming-timeout-s", type=float, default=5.0)
    parser.add_argument("--lowstate-timeout-s", type=float, default=0.2)

    parser.add_argument("--subscriber-settle-time", type=float, default=1.0)
    parser.add_argument("--warmup-frames", type=int, default=25)
    parser.add_argument("--init-timeout", type=float, default=60.0)
    parser.add_argument("--streaming-timeout", type=float, default=15.0)
    parser.add_argument("--control-timeout", type=float, default=15.0)
    parser.add_argument("--stop-timeout", type=float, default=10.0)

    parser.add_argument("--session-name", default=DEFAULT_SESSION)
    parser.add_argument("--attach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-session", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--controller", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_dataset_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _episode_path(dataset_path: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_path / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _feature_names(info: dict[str, Any], column: str) -> list[str]:
    names = info.get("features", {}).get(column, {}).get("names", [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"meta/info.json does not provide names for {column}")
    return [str(name) for name in names]


def _normalize_name(name: str) -> str:
    return name if name.endswith("_joint") else f"{name}_joint"


def _load_replay_data(args: argparse.Namespace) -> ReplayData:
    dataset_path = _resolve_dataset_path(args.dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes and not 0 <= args.episode_index < total_episodes:
        raise ValueError(
            f"episode_index {args.episode_index} is outside [0, {total_episodes - 1}]"
        )
    fps = float(info.get("fps", 50.0))
    if fps <= 0 or args.speed <= 0:
        raise ValueError("Dataset fps and --speed must be positive")

    parquet_path = _episode_path(dataset_path, info, args.episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    table = pd.read_parquet(parquet_path)
    required = [args.joint_column, "observation.root_orientation", "timestamp"]
    missing_columns = [column for column in required if column not in table.columns]
    if missing_columns:
        raise ValueError(
            f"Episode is missing required columns {missing_columns}; available={list(table.columns)}"
        )

    start = args.start_frame
    end = len(table) if args.end_frame is None else args.end_frame
    if start < 0 or end <= start or end > len(table):
        raise ValueError(f"Invalid frame range [{start}, {end}) for {len(table)} frames")
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        end = min(end, start + args.max_frames)
    table = table.iloc[start:end].reset_index(drop=True)

    source_names = [_normalize_name(name) for name in _feature_names(info, args.joint_column)]
    source_index = {name: index for index, name in enumerate(source_names)}
    missing_joints = [name for name in G1_ISAACLAB_ORDER if name not in source_index]
    if missing_joints:
        raise ValueError(f"{args.joint_column} is missing G1 body joints: {missing_joints}")
    indices = np.asarray([source_index[name] for name in G1_ISAACLAB_ORDER], dtype=np.int64)

    source_values = np.stack(
        [np.asarray(value, dtype=np.float32).reshape(-1) for value in table[args.joint_column]],
        axis=0,
    )
    if source_values.shape[1] != len(source_names):
        raise ValueError(
            f"{args.joint_column} width {source_values.shape[1]} != metadata names {len(source_names)}"
        )
    joint_pos = source_values[:, indices]
    if not np.all(np.isfinite(joint_pos)):
        raise ValueError(f"{args.joint_column} contains non-finite G1 joint values")

    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamp contains non-finite values")
    if args.velocity_source == "zeros":
        joint_vel = np.zeros_like(joint_pos)
    else:
        fallback_dt = 1.0 / fps
        delta_t = np.diff(timestamps)
        delta_t = np.where(delta_t > 1e-8, delta_t, fallback_dt)
        joint_vel = np.empty_like(joint_pos)
        if len(joint_pos) == 1:
            joint_vel[0] = 0.0
        else:
            joint_vel[0] = (joint_pos[1] - joint_pos[0]) / delta_t[0]
            joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / delta_t[:, None]
    if not np.all(np.isfinite(joint_vel)):
        raise ValueError("Computed G1 joint velocities contain non-finite values")

    root_quat = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in table["observation.root_orientation"]
        ],
        axis=0,
    )
    if root_quat.shape != (len(table), 4) or not np.all(np.isfinite(root_quat)):
        raise ValueError("observation.root_orientation must be finite [frames, 4]")
    quat_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(quat_norm < 1e-8):
        raise ValueError("observation.root_orientation contains a zero quaternion")
    root_quat = (root_quat / quat_norm).astype(np.float32)

    result = ReplayData(
        dataset_path=dataset_path,
        parquet_path=parquet_path,
        fps=fps,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        root_quat=root_quat,
    )
    print(
        f"Loaded episode {args.episode_index}: {result.num_frames} frames, fps={fps:g}, "
        f"duration={result.num_frames / fps:.2f}s, path={parquet_path}"
    )
    print(
        f"G1 input: {args.joint_column} -> 29 body joints in IsaacLab encoder order; "
        f"velocity={args.velocity_source}; max_abs_velocity={np.abs(result.joint_vel).max():.4f}"
    )
    print("Hand input: disabled (no hand fields will be transmitted).")
    return result


def _ssh_target(args: argparse.Namespace) -> str:
    return f"{args.onboard_user}@{args.onboard_host}"


def _required_onboard_paths(args: argparse.Namespace) -> list[str]:
    return [
        args.onboard_binary,
        f"{args.deploy_checkpoint}_decoder.onnx",
        f"{args.deploy_checkpoint}_encoder.onnx",
        args.deploy_obs_config,
        args.deploy_planner,
        args.deploy_motion_data,
    ]


def _asset_check_command(args: argparse.Namespace, success_marker: str) -> str:
    deploy_root = f"{args.onboard_repo_root}/gear_sonic_deploy"
    checks = " && ".join(f"test -e {shlex.quote(path)}" for path in _required_onboard_paths(args))
    return (
        f"cd {shlex.quote(deploy_root)} && {checks} && "
        f"test -x {shlex.quote(args.onboard_binary)} && echo {shlex.quote(success_marker)}"
    )


def _verify_onboard_assets(args: argparse.Namespace) -> None:
    marker = "ONBOARD_REPLAY_ASSETS_OK"
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            _ssh_target(args),
            _asset_check_command(args, marker),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or marker not in result.stdout:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown SSH/asset error"
        raise RuntimeError(f"Onboard replay asset check failed: {detail}")
    print(f"Onboard asset check passed: {_ssh_target(args)}")


def _deploy_command(args: argparse.Namespace) -> str:
    deploy_root = f"{args.onboard_repo_root}/gear_sonic_deploy"
    deploy_args = [
        f"./{args.onboard_binary}",
        args.onboard_interface,
        f"{args.deploy_checkpoint}_decoder.onnx",
        args.deploy_motion_data,
        "--obs-config",
        args.deploy_obs_config,
        "--encoder-file",
        f"{args.deploy_checkpoint}_encoder.onnx",
        "--planner-file",
        args.deploy_planner,
        "--input-type",
        "zmq_manager",
        "--output-type",
        args.deploy_output_type,
        "--zmq-host",
        args.offboard_zmq_host,
        "--zmq-port",
        str(args.zmq_port),
        "--zmq-topic",
        "pose",
        "--enable-sonic-publish",
        "--max-runtime-s",
        str(args.publish_max_runtime_s),
        "--streaming-timeout-s",
        str(args.streaming_timeout_s),
        "--lowstate-timeout-s",
        str(args.lowstate_timeout_s),
        "--sonic-publish-confirm",
        SONIC_PUBLISH_CONFIRMATION,
    ]
    remote_command = f"cd {shlex.quote(deploy_root)} && exec {shlex.join(deploy_args)}"
    return shlex.join(["ssh", "-tt", _ssh_target(args), remote_command])


def _tmux(*tmux_args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *tmux_args],
        check=check,
        capture_output=capture,
        text=capture,
    )


def _pane_text(target: str) -> str:
    return _tmux("capture-pane", "-p", "-t", target, "-S", "-3000", capture=True).stdout


def _wait_for_pane(target: str, needle: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if needle in _pane_text(target):
                return True
        except subprocess.CalledProcessError:
            return False
        time.sleep(0.25)
    return False


def _pack_g1_message(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_quat: np.ndarray,
    frame_index: int,
) -> bytes:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    return pack_pose_message(
        {
            "joint_pos": np.asarray(joint_pos, dtype=np.float32).reshape(1, 29),
            "joint_vel": np.asarray(joint_vel, dtype=np.float32).reshape(1, 29),
            "body_quat_w": np.asarray(root_quat, dtype=np.float32).reshape(1, 4),
            "frame_index": np.asarray([frame_index], dtype=np.int64),
        },
        topic="pose",
        version=1,
    )


def _send_command(socket: Any, start: bool) -> None:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    message = build_command_message(start=start, stop=not start, planner=False)
    for _ in range(3):
        socket.send(message)
        time.sleep(0.02)


def _hold_first_pose(
    socket: Any,
    message: bytes,
    period: float,
    stop_event: threading.Event,
) -> None:
    target_time = time.monotonic()
    while not stop_event.is_set():
        socket.send(message)
        target_time += period
        stop_event.wait(max(0.0, target_time - time.monotonic()))


def _stream_trajectory(args: argparse.Namespace, data: ReplayData, socket: Any) -> None:
    period = 1.0 / (data.fps * args.speed)
    target_time = time.monotonic()
    for frame_index in range(data.num_frames):
        socket.send(
            _pack_g1_message(
                data.joint_pos[frame_index],
                data.joint_vel[frame_index],
                data.root_quat[frame_index],
                frame_index,
            )
        )
        if args.verbose and frame_index % max(1, int(data.fps)) == 0:
            print(f"Replay frame {frame_index}/{data.num_frames - 1}", flush=True)
        target_time += period
        remaining = target_time - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def _stop_real_deploy(
    args: argparse.Namespace,
    deploy_target: str,
    socket: Any | None,
) -> None:
    if socket is not None:
        try:
            _send_command(socket, start=False)
        except Exception as exc:
            print(f"WARNING: failed to send network stop: {exc}")
    try:
        _tmux("send-keys", "-t", deploy_target, "O")
    except subprocess.CalledProcessError:
        return
    if _wait_for_pane(deploy_target, "Stop", args.stop_timeout):
        print("Onboard deploy reported Stop; damping command requested.")
        return
    print("WARNING: onboard deploy did not report Stop before timeout; sending Ctrl+C.")
    try:
        _tmux("send-keys", "-t", deploy_target, "C-c")
    except subprocess.CalledProcessError:
        pass


def _run_dry_controller(
    args: argparse.Namespace, data: ReplayData, deploy_target: str
) -> None:
    input(
        "[DRY RUN] Press Enter for stage 1: verify onboard command/assets "
        "(deploy binary will NOT run)..."
    )
    marker = "DRY_RUN_DEPLOY_READY"
    remote_check = _asset_check_command(args, marker)
    dry_command = shlex.join(["ssh", "-T", _ssh_target(args), remote_check])
    _tmux("send-keys", "-t", deploy_target, dry_command, "C-m")
    if not _wait_for_pane(deploy_target, marker, args.init_timeout):
        raise RuntimeError(f"Dry-run onboard asset pane failed; inspect {deploy_target}")
    print("[DRY RUN] Stage 1 passed; no deploy process or robot command was started.")

    input("[DRY RUN] Press Enter for stage 2: validate the complete trajectory stream...")
    first = _pack_g1_message(data.joint_pos[0], data.joint_vel[0], data.root_quat[0], 0)
    last_index = data.num_frames - 1
    last = _pack_g1_message(
        data.joint_pos[last_index],
        data.joint_vel[last_index],
        data.root_quat[last_index],
        last_index,
    )
    if not first or not last:
        raise RuntimeError("Packed G1 replay message is empty")
    print(
        f"DRY_RUN_COMPLETE frames={data.num_frames} first_bytes={len(first)} "
        f"last_bytes={len(last)} hand_fields=0"
    )


def _run_real_controller(
    args: argparse.Namespace, data: ReplayData, deploy_target: str
) -> None:
    import zmq

    socket = None
    context = None
    hold_stop = threading.Event()
    hold_thread = None
    deploy_started = False
    try:
        print()
        print("Robot stage: onboard deploy is NOT running; no replay command is being published.")
        input(
            "Press Enter to start onboard SONIC, run its default-pose INIT, "
            "and enter CONTROL while holding the first recorded pose..."
        )

        _verify_onboard_assets(args)
        _tmux("send-keys", "-t", deploy_target, _deploy_command(args), "C-m")
        deploy_started = True
        print("Onboard deploy started; waiting for Init Done...")
        if not _wait_for_pane(deploy_target, "Init Done", args.init_timeout):
            raise RuntimeError(f"Onboard deploy did not reach Init Done; inspect {deploy_target}")

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 3)
        socket.bind(args.zmq_bind)
        print(f"G1 replay publisher bound to {args.zmq_bind}; waiting for subscriber...")
        time.sleep(args.subscriber_settle_time)

        first_message = _pack_g1_message(
            data.joint_pos[0], data.joint_vel[0], data.root_quat[0], 0
        )
        frame_period = 1.0 / data.fps
        for _ in range(args.warmup_frames):
            socket.send(first_message)
            time.sleep(frame_period)
        _send_command(socket, start=True)
        if not _wait_for_pane(deploy_target, "ZMQ streaming enabled", args.streaming_timeout):
            raise RuntimeError("Onboard deploy did not enable ZMQ streaming")

        _tmux("send-keys", "-t", deploy_target, "]")
        if not _wait_for_pane(
            deploy_target, "transitioning to CONTROL", args.control_timeout
        ):
            raise RuntimeError("Onboard SONIC did not enter CONTROL")

        hold_thread = threading.Thread(
            target=_hold_first_pose,
            args=(socket, first_message, frame_period, hold_stop),
            daemon=True,
        )
        hold_thread.start()
        print("SONIC CONTROL is active; first recorded body pose is being held at 50 Hz.")
        print("Hands remain untouched by this replay script.")
        input("Press Enter to START the recorded trajectory replay...")

        hold_stop.set()
        hold_thread.join(timeout=2.0)
        hold_thread = None
        print(
            f"Starting real-robot replay: {data.num_frames} frames at "
            f"{data.fps * args.speed:g} Hz"
        )
        _stream_trajectory(args, data, socket)
        print("Trajectory finished; requesting SONIC stop/damping.")
    except KeyboardInterrupt:
        print("Interrupted; requesting immediate SONIC stop/damping.")
    finally:
        hold_stop.set()
        if hold_thread is not None:
            hold_thread.join(timeout=2.0)
        if deploy_started:
            _stop_real_deploy(args, deploy_target, socket)
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()


def _controller(args: argparse.Namespace) -> None:
    data = _load_replay_data(args)
    deploy_target = f"{args.session_name}:replay.0"
    print(f"Deploy pane: {deploy_target}")
    if args.dry_run:
        _run_dry_controller(args, data, deploy_target)
    else:
        _run_real_controller(args, data, deploy_target)


def _controller_command(args: argparse.Namespace) -> str:
    argv = [arg for arg in sys.argv[1:] if arg != "--controller"]
    argv.extend(("--session-name", args.session_name, "--controller"))
    python = REPO_ROOT / ".venv_data_collection" / "bin" / "python"
    return shlex.join([str(python), str(Path(__file__).resolve()), *argv])


def _launch_tmux(args: argparse.Namespace) -> None:
    session = args.session_name
    if args.dry_run and session == DEFAULT_SESSION:
        session = f"{DEFAULT_SESSION}_dry_run"
        args.session_name = session
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session], capture_output=True
    ).returncode == 0
    if existing:
        if not args.replace_session:
            raise RuntimeError(
                f"tmux session {session!r} already exists; inspect it or pass --replace-session"
            )
        _tmux("kill-session", "-t", session)

    _tmux("new-session", "-d", "-s", session, "-n", "replay")
    _tmux("set-option", "-t", session, "-g", "mouse", "on")
    _tmux("set-option", "-t", session, "-g", "history-limit", "20000")
    _tmux("split-window", "-h", "-t", f"{session}:replay")
    controller_target = f"{session}:replay.1"
    _tmux("send-keys", "-t", controller_target, _controller_command(args), "C-m")
    _tmux("select-pane", "-t", controller_target)

    print(f"Created tmux session: {session}")
    print(f"  pane 0: onboard deploy (starts only after the first Enter; dry-run never starts it)")
    print("  pane 1: gated replay controller (selected)")
    print(f"  reattach: tmux attach -t {session}")
    print("  use Ctrl+C in pane 1 for the graceful SONIC stop/damping path")
    if args.attach:
        subprocess.run(["tmux", "attach", "-t", session], check=False)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "--publish-max-runtime-s": args.publish_max_runtime_s,
        "--lowstate-timeout-s": args.lowstate_timeout_s,
        "--init-timeout": args.init_timeout,
        "--streaming-timeout": args.streaming_timeout,
        "--control-timeout": args.control_timeout,
        "--stop-timeout": args.stop_timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.subscriber_settle_time < 0 or args.warmup_frames < 0:
        parser.error("--subscriber-settle-time and --warmup-frames must be non-negative")
    if args.validate_only and args.controller:
        parser.error("--validate-only cannot be combined with --controller")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.controller:
        _controller(args)
        return

    data = _load_replay_data(args)
    _verify_onboard_assets(args)
    if args.validate_only:
        first = _pack_g1_message(data.joint_pos[0], data.joint_vel[0], data.root_quat[0], 0)
        print(
            f"Validation passed: {data.num_frames} G1 frames; protocol-v1 message "
            f"size={len(first)} bytes; hand_fields=0; no robot command was sent."
        )
        return
    _launch_tmux(args)


def _signal_handler(_signal: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main()
