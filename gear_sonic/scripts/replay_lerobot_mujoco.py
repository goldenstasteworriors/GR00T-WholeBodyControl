"""Replay a collected SONIC LeRobot episode in MuJoCo.

Two replay paths are supported:

``joint``
    Kinematic playback. Joint values from ``observation.state`` (or
    ``action.wbc``) are written to MuJoCo by joint name. This path does not run
    physics or the SONIC policy; it is intended for inspecting the recorded
    robot pose exactly.

``token``
    Closed-loop SONIC playback. Recorded ``action.motion_token`` values are
    published with SONIC's ZMQ protocol v4. The existing C++ deploy process
    consumes the tokens, builds observations from the live MuJoCo state, runs
    the local SONIC decoder, and sends its joint targets through the normal
    Unitree DDS/PD simulation path.

The token path can launch MuJoCo, C++ deploy, and this publisher in one tmux
session. It is deliberately simulation-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_ROOT = REPO_ROOT / "gear_sonic_deploy"
SESSION_NAME = "sonic_dataset_replay"


def _requested_mode(argv: Sequence[str]) -> str:
    for index, arg in enumerate(argv):
        if arg == "--mode" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return "joint"


def _bootstrap_environment() -> None:
    """Re-exec in the project environment appropriate for the selected mode."""
    mode = _requested_mode(sys.argv[1:])
    env_name = ".venv_sim" if mode == "joint" else ".venv_data_collection"
    env_python = REPO_ROOT / env_name / "bin" / "python"
    required_modules = ("mujoco", "pandas") if mode == "joint" else ("pandas", "zmq")

    try:
        for module in required_modules:
            __import__(module)
        return
    except ImportError as exc:
        if env_python.exists() and Path(sys.executable).resolve() != env_python.resolve():
            script_path = str(Path(__file__).resolve())
            os.execv(str(env_python), [str(env_python), script_path, *sys.argv[1:]])

        if mode == "joint":
            install_hint = "bash install_scripts/install_mujoco_sim.sh"
        else:
            install_hint = "bash install_scripts/install_data_collection.sh"
        raise SystemExit(
            f"ERROR: {env_name} is missing a replay dependency: {exc}.\n"
            f"Run from the repository root: {install_hint}"
        ) from exc


_bootstrap_environment()

import numpy as np
import pandas as pd


@dataclass
class EpisodeData:
    dataset_path: Path
    parquet_path: Path
    info: dict[str, Any]
    frame_table: pd.DataFrame
    fps: float
    state_joint_names: list[str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay SONIC LeRobot data in MuJoCo (joint angles or latent tokens)."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--mode", choices=("joint", "token"), default="joint")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the selected frame count. Primarily useful for smoke tests.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    joint_group = parser.add_argument_group("joint replay")
    joint_group.add_argument(
        "--joint-column",
        choices=("observation.state", "action.wbc"),
        default="observation.state",
    )
    joint_group.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="MuJoCo scene XML. By default it is selected from dataset joint names.",
    )
    joint_group.add_argument("--root-position", type=float, nargs=3, default=(0.0, 0.0, 0.8))
    joint_group.add_argument(
        "--recorded-root-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    joint_group.add_argument("--headless", action="store_true")

    token_group = parser.add_argument_group("token replay")
    token_group.add_argument(
        "--launch-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch MuJoCo and C++ deploy in a tmux session.",
    )
    token_group.add_argument("--attach", action=argparse.BooleanOptionalAction, default=True)
    token_group.add_argument(
        "--replace-session", action=argparse.BooleanOptionalAction, default=True
    )
    token_group.add_argument("--zmq-bind", default="tcp://*:5556")
    token_group.add_argument("--deploy-zmq-host", default="localhost")
    token_group.add_argument(
        "--deploy-checkpoint",
        default="",
        help="Model prefix relative to gear_sonic_deploy. Empty uses dataset script_config.",
    )
    token_group.add_argument(
        "--deploy-obs-config",
        default="",
        help="Observation config relative to gear_sonic_deploy. Empty uses dataset script_config.",
    )
    token_group.add_argument(
        "--deploy-planner",
        default="",
        help="Planner path relative to gear_sonic_deploy. Empty uses dataset script_config.",
    )
    token_group.add_argument(
        "--deploy-motion-data", default="reference/example/"
    )
    token_group.add_argument("--deploy-output-type", default="all")
    token_group.add_argument("--sim-headless", action="store_true")
    token_group.add_argument("--send-hands", action=argparse.BooleanOptionalAction, default=True)
    token_group.add_argument("--subscriber-settle-time", type=float, default=1.0)
    token_group.add_argument("--warmup-frames", type=int, default=25)
    token_group.add_argument(
        "--start-hold-time",
        type=float,
        default=1.0,
        help="Repeat the first token briefly after requesting control start.",
    )
    token_group.add_argument("--deploy-ready-timeout", type=float, default=90.0)
    token_group.add_argument("--sim-startup-delay", type=float, default=3.0)
    return parser


def _resolve_dataset_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _episode_parquet_path(dataset_path: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_path / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _load_episode(args: argparse.Namespace) -> EpisodeData:
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

    parquet_path = _episode_parquet_path(dataset_path, info, args.episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    frame_table = pd.read_parquet(parquet_path)

    required = [args.joint_column] if args.mode == "joint" else ["action.motion_token"]
    missing = [name for name in required if name not in frame_table.columns]
    if missing:
        raise ValueError(
            f"Episode is missing required columns {missing}. Available: {list(frame_table.columns)}"
        )

    start = args.start_frame
    end = len(frame_table) if args.end_frame is None else args.end_frame
    if start < 0 or end < 0 or start >= end or end > len(frame_table):
        raise ValueError(
            f"Invalid frame range [{start}, {end}) for episode with {len(frame_table)} frames"
        )
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        end = min(end, start + args.max_frames)
    frame_table = frame_table.iloc[start:end].reset_index(drop=True)

    features = info.get("features", {})
    state_feature = features.get("observation.state", {})
    state_names = state_feature.get("names", [])
    if not isinstance(state_names, list):
        state_names = []

    fps = float(info.get("fps", 50))
    if fps <= 0 or args.speed <= 0:
        raise ValueError("Dataset fps and --speed must both be positive")

    episode = EpisodeData(
        dataset_path=dataset_path,
        parquet_path=parquet_path,
        info=info,
        frame_table=frame_table,
        fps=fps,
        state_joint_names=state_names,
    )
    print(
        f"Loaded episode {args.episode_index}: {len(frame_table)} selected frames, "
        f"dataset fps={fps:g}, path={parquet_path}"
    )
    return episode


def _default_scene(state_joint_names: Sequence[str]) -> Path:
    is_inspire = any("hand_little" in name or "hand_ring" in name for name in state_joint_names)
    filename = "scene_29dof_inspire.xml" if is_inspire else "scene_43dof.xml"
    return REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1" / filename


def _resolve_scene(args: argparse.Namespace, episode: EpisodeData) -> Path:
    scene = args.scene if args.scene is not None else _default_scene(episode.state_joint_names)
    if not scene.is_absolute():
        scene = REPO_ROOT / scene
    scene = scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
    return scene


def _joint_name_candidates(name: str) -> tuple[str, ...]:
    if name.endswith("_joint"):
        return (name,)
    return (name, f"{name}_joint")


def _joint_value_for_mujoco(name: str, value: float) -> float:
    """Convert recorded Inspire normalized positions to the scene's legacy qpos."""
    normalized_name = name.removesuffix("_joint")
    inspire_value = float(np.clip(value, 0.0, 1.0))
    if normalized_name.endswith(("_hand_little", "_hand_ring", "_hand_middle", "_hand_index")):
        return 1.7 * (1.0 - inspire_value)
    if normalized_name.endswith("_hand_thumb_bend"):
        return 0.5 * (1.0 - inspire_value)
    if normalized_name.endswith("_hand_thumb_rotate"):
        return 1.3 - 1.4 * inspire_value
    return float(value)


def _build_qpos_mapping(model: Any, joint_names: Sequence[str]) -> list[tuple[int, int, str]]:
    import mujoco

    mapping: list[tuple[int, int, str]] = []
    missing: list[str] = []
    for source_index, source_name in enumerate(joint_names):
        joint_id = -1
        resolved_name = source_name
        for candidate in _joint_name_candidates(source_name):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if joint_id >= 0:
                resolved_name = candidate
                break
        if joint_id < 0:
            missing.append(source_name)
            continue
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"Recorded joint {source_name} maps to non-scalar joint {resolved_name}"
            )
        mapping.append((source_index, int(model.jnt_qposadr[joint_id]), resolved_name))

    if missing:
        raise ValueError(f"Dataset joints are missing from MuJoCo model: {missing}")
    return mapping


def _joint_names_for_column(args: argparse.Namespace, episode: EpisodeData) -> list[str]:
    feature = episode.info.get("features", {}).get(args.joint_column, {})
    names = feature.get("names", [])
    if not isinstance(names, list) or not names:
        if args.joint_column == "action.wbc" and episode.state_joint_names:
            names = episode.state_joint_names
        else:
            raise ValueError(
                f"meta/info.json does not provide joint names for {args.joint_column}"
            )
    return names


def _set_free_root_pose(model: Any, data: Any, position: Sequence[float], quaternion: Any) -> None:
    import mujoco

    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos_adr = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(position, dtype=np.float64)
            quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
            if quat.shape != (4,) or not np.all(np.isfinite(quat)):
                raise ValueError(f"Invalid root quaternion: {quat}")
            norm = np.linalg.norm(quat)
            if norm < 1e-8:
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                quat = quat / norm
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
            return


def _run_joint_replay(args: argparse.Namespace, episode: EpisodeData) -> None:
    import mujoco
    import mujoco.viewer

    scene = _resolve_scene(args, episode)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    joint_names = _joint_names_for_column(args, episode)
    mapping = _build_qpos_mapping(model, joint_names)

    first_values = np.asarray(episode.frame_table.iloc[0][args.joint_column]).reshape(-1)
    if len(first_values) != len(joint_names):
        raise ValueError(
            f"{args.joint_column} has {len(first_values)} values but metadata has "
            f"{len(joint_names)} joint names"
        )

    print(f"Joint replay scene: {scene}")
    print(f"Mapped {len(mapping)} recorded joints by name")
    if args.verbose:
        for source_index, qpos_index, name in mapping:
            print(f"  data[{source_index:02d}] -> qpos[{qpos_index:02d}] {name}")
    if args.validate_only:
        print("Validation passed: joint replay inputs and MuJoCo mapping are compatible.")
        return

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.2
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.7])

    root_column_available = "observation.root_orientation" in episode.frame_table.columns
    period = 1.0 / (episode.fps * args.speed)
    try:
        while True:
            target_time = time.monotonic()
            for frame_offset, row in episode.frame_table.iterrows():
                if viewer is not None and not viewer.is_running():
                    return
                values = np.asarray(row[args.joint_column], dtype=np.float64).reshape(-1)
                if values.shape != (len(joint_names),):
                    raise ValueError(
                        f"Frame {frame_offset}: {args.joint_column} shape {values.shape} does not "
                        f"match metadata joint count {len(joint_names)}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Frame {frame_offset} contains non-finite joint values")
                for source_index, qpos_index, _ in mapping:
                    data.qpos[qpos_index] = _joint_value_for_mujoco(
                        joint_names[source_index], values[source_index]
                    )

                root_quat = (
                    row["observation.root_orientation"]
                    if args.recorded_root_orientation and root_column_available
                    else np.array([1.0, 0.0, 0.0, 0.0])
                )
                _set_free_root_pose(model, data, args.root_position, root_quat)
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                if viewer is not None:
                    viewer.sync()

                if args.verbose and frame_offset % max(1, int(episode.fps)) == 0:
                    print(f"Joint replay frame {frame_offset}/{len(episode.frame_table) - 1}")
                if viewer is not None:
                    target_time += period
                    remaining = target_time - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
            if not args.loop:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print("Joint replay complete.")


def _inspire_to_legacy_hand(values: np.ndarray) -> np.ndarray:
    """Convert Inspire's six normalized joints to the legacy seven-value hand layout."""
    values = np.clip(np.asarray(values, dtype=np.float32).reshape(6), 0.0, 1.0)
    result = np.zeros(7, dtype=np.float32)
    result[0:4] = 1.7 * (1.0 - values[0:4])
    result[4] = 1.3 - values[5] * 1.4
    result[5] = 0.5 * (1.0 - values[4])
    return result


def _normalize_hand(values: Any, side: str) -> np.ndarray:
    hand = np.asarray(values, dtype=np.float32).reshape(-1)
    if hand.shape == (6,):
        return _inspire_to_legacy_hand(hand)
    if hand.shape == (7,):
        return hand
    raise ValueError(f"{side} hand must contain 6 Inspire or 7 legacy joints, got {hand.shape}")


def _validate_token_episode(args: argparse.Namespace, episode: EpisodeData) -> None:
    for frame_offset, row in episode.frame_table.iterrows():
        token = np.asarray(row["action.motion_token"], dtype=np.float32).reshape(-1)
        if token.shape != (64,):
            raise ValueError(f"Frame {frame_offset}: expected 64D token, got {token.shape}")
        if not np.all(np.isfinite(token)):
            raise ValueError(f"Frame {frame_offset}: token contains non-finite values")
        if args.send_hands:
            for side in ("left", "right"):
                column = f"teleop.{side}_hand_joints"
                if column not in episode.frame_table.columns:
                    raise ValueError(f"--send-hands requires dataset column {column}")
                _normalize_hand(row[column], side)


def _pack_token_message(row: pd.Series, send_hands: bool) -> bytes:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    frame_index = row.get("frame_index", 0)
    pose_data: dict[str, np.ndarray] = {
        "token_state": np.asarray(row["action.motion_token"], dtype=np.float32).reshape(1, 64),
        "frame_index": np.asarray([frame_index], dtype=np.int64),
    }
    if send_hands:
        pose_data["left_hand_joints"] = _normalize_hand(
            row["teleop.left_hand_joints"], "left"
        ).reshape(1, 7)
        pose_data["right_hand_joints"] = _normalize_hand(
            row["teleop.right_hand_joints"], "right"
        ).reshape(1, 7)
    return pack_pose_message(pose_data, topic="pose", version=4)


def _send_command(socket: Any, start: bool) -> None:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    message = build_command_message(start=start, stop=not start, planner=False)
    for _ in range(3):
        socket.send(message)
        time.sleep(0.02)


def _run_token_publisher(args: argparse.Namespace, episode: EpisodeData) -> None:
    import zmq

    _validate_token_episode(args, episode)
    if args.validate_only:
        print("Validation passed: all selected frames contain compatible 64D SONIC tokens.")
        return

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 3)
    socket.bind(args.zmq_bind)
    print(f"Token publisher bound to {args.zmq_bind}")
    print(f"Waiting {args.subscriber_settle_time:g}s for C++ subscribers...")
    time.sleep(args.subscriber_settle_time)

    first_message = _pack_token_message(episode.frame_table.iloc[0], args.send_hands)
    warmup_period = 1.0 / episode.fps
    for _ in range(args.warmup_frames):
        socket.send(first_message)
        time.sleep(warmup_period)
    _send_command(socket, start=True)
    print("Sent C++ start command in streamed/token mode.")
    hold_deadline = time.monotonic() + args.start_hold_time
    while time.monotonic() < hold_deadline:
        socket.send(first_message)
        time.sleep(warmup_period)

    period = 1.0 / (episode.fps * args.speed)
    try:
        while True:
            target_time = time.monotonic()
            for frame_offset, row in episode.frame_table.iterrows():
                socket.send(_pack_token_message(row, args.send_hands))
                if args.verbose and frame_offset % max(1, int(episode.fps)) == 0:
                    print(f"Token replay frame {frame_offset}/{len(episode.frame_table) - 1}")
                target_time += period
                remaining = target_time - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("Token replay interrupted.")
    finally:
        try:
            _send_command(socket, start=False)
        finally:
            socket.close(linger=0)
            context.term()
    print("Token replay complete; sent C++ stop command.")


def _script_config(episode: EpisodeData) -> dict[str, Any]:
    config = episode.info.get("script_config", {})
    return config if isinstance(config, dict) else {}


def _checkpoint_prefix_from_decoder(path: str) -> str:
    suffix = "_decoder.onnx"
    if not path.endswith(suffix):
        raise ValueError(
            "Dataset script_config.model_path must end with '_decoder.onnx', or pass "
            "--deploy-checkpoint explicitly"
        )
    return path[: -len(suffix)]


def _deploy_paths(args: argparse.Namespace, episode: EpisodeData) -> tuple[str, str, str]:
    script_config = _script_config(episode)
    checkpoint = args.deploy_checkpoint
    if not checkpoint:
        model_path = str(script_config.get("model_path", ""))
        if not model_path:
            raise ValueError(
                "Dataset has no script_config.model_path; pass --deploy-checkpoint"
            )
        checkpoint = _checkpoint_prefix_from_decoder(model_path)

    obs_config = args.deploy_obs_config or str(script_config.get("obs_config_path", ""))
    planner = args.deploy_planner or str(script_config.get("planner_path", ""))
    if not obs_config:
        raise ValueError("Dataset has no obs_config_path; pass --deploy-obs-config")
    if not planner:
        raise ValueError("Dataset has no planner_path; pass --deploy-planner")

    required_paths = {
        "decoder": f"{checkpoint}_decoder.onnx",
        "encoder": f"{checkpoint}_encoder.onnx",
        "observation config": obs_config,
        "planner": planner,
        "motion data": args.deploy_motion_data,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_paths.items()
        if not (DEPLOY_ROOT / path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Token replay deployment assets are missing under gear_sonic_deploy:\n  "
            + "\n  ".join(missing)
        )
    return checkpoint, obs_config, planner


def _tmux(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args],
        check=check,
        capture_output=capture,
        text=capture,
    )


def _pane_text(target: str) -> str:
    result = _tmux("capture-pane", "-p", "-t", target, "-S", "-2000", capture=True)
    return result.stdout


def _wait_for_pane_text(target: str, needle: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in _pane_text(target):
            return True
        time.sleep(0.5)
    return False


def _create_token_session(args: argparse.Namespace, episode: EpisodeData) -> None:
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required for --launch-stack")
    if not (REPO_ROOT / ".venv_sim" / "bin" / "python").exists():
        raise RuntimeError(
            ".venv_sim is missing; run: bash install_scripts/install_mujoco_sim.sh"
        )
    checkpoint, obs_config, planner = _deploy_paths(args, episode)

    existing = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME], capture_output=True
    ).returncode == 0
    if existing:
        if not args.replace_session:
            raise RuntimeError(
                f"tmux session {SESSION_NAME!r} already exists; use --replace-session or remove it"
            )
        _tmux("kill-session", "-t", SESSION_NAME)

    _tmux("new-session", "-d", "-s", SESSION_NAME, "-n", "replay")
    _tmux("set-option", "-t", SESSION_NAME, "-g", "mouse", "on")
    _tmux("set-option", "-t", SESSION_NAME, "-g", "history-limit", "10000")
    _tmux("bind-key", "-T", "root", "C-\\", "kill-session")
    _tmux("split-window", "-h", "-t", f"{SESSION_NAME}:replay")
    _tmux("new-window", "-t", SESSION_NAME, "-n", "sim")

    sim_args = [
        "python",
        "gear_sonic/scripts/run_sim_loop.py",
    ]
    if args.sim_headless:
        sim_args.append("--no-enable-onscreen")
    sim_cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        "source .venv_sim/bin/activate && "
        f"{shlex.join(sim_args)}"
    )
    _tmux("send-keys", "-t", f"{SESSION_NAME}:sim", sim_cmd, "C-m")
    print(f"Started MuJoCo; waiting {args.sim_startup_delay:g}s...")
    time.sleep(args.sim_startup_delay)

    deploy_args = [
        "./deploy.sh",
        "--input-type",
        "zmq_manager",
        "--zmq-host",
        args.deploy_zmq_host,
        "--cp",
        checkpoint,
        "--obs-config",
        obs_config,
        "--planner",
        planner,
        "--motion-data",
        args.deploy_motion_data,
        "--output-type",
        args.deploy_output_type,
        "sim",
    ]
    deploy_cmd = f"cd {shlex.quote(str(DEPLOY_ROOT))} && {shlex.join(deploy_args)}"
    deploy_target = f"{SESSION_NAME}:replay.0"
    _tmux("send-keys", "-t", deploy_target, deploy_cmd, "C-m")
    print("Started C++ deploy; waiting for its confirmation prompt...")
    if not _wait_for_pane_text(deploy_target, "Proceed with deployment?", 30.0):
        raise RuntimeError(
            f"C++ deploy did not reach confirmation prompt. Inspect: tmux attach -t {SESSION_NAME}"
        )
    _tmux("send-keys", "-t", deploy_target, "C-m")
    print("Confirmed simulation deployment; waiting for SONIC initialization...")
    if not _wait_for_pane_text(
        deploy_target, "Init Done", args.deploy_ready_timeout
    ):
        raise RuntimeError(
            f"C++ deploy did not finish its initial pose within "
            f"{args.deploy_ready_timeout:g}s. "
            f"Inspect: tmux attach -t {SESSION_NAME}"
        )

    replay_args = [
        "python",
        str(Path(__file__).resolve()),
        "--dataset-path",
        str(episode.dataset_path),
        "--episode-index",
        str(args.episode_index),
        "--mode",
        "token",
        "--start-frame",
        str(args.start_frame),
        "--speed",
        str(args.speed),
        "--no-launch-stack",
        "--zmq-bind",
        args.zmq_bind,
        "--subscriber-settle-time",
        str(args.subscriber_settle_time),
        "--warmup-frames",
        str(args.warmup_frames),
        "--start-hold-time",
        str(args.start_hold_time),
    ]
    if args.end_frame is not None:
        replay_args.extend(("--end-frame", str(args.end_frame)))
    if args.max_frames is not None:
        replay_args.extend(("--max-frames", str(args.max_frames)))
    replay_args.append("--loop" if args.loop else "--no-loop")
    replay_args.append("--send-hands" if args.send_hands else "--no-send-hands")
    if args.verbose:
        replay_args.append("--verbose")
    replay_cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        "source .venv_data_collection/bin/activate && "
        f"{shlex.join(replay_args)}"
    )
    replay_target = f"{SESSION_NAME}:replay.1"
    _tmux("send-keys", "-t", replay_target, replay_cmd, "C-m")

    publisher_timeout = max(
        30.0,
        args.subscriber_settle_time
        + args.warmup_frames / episode.fps
        + args.start_hold_time
        + 10.0,
    )
    if not _wait_for_pane_text(
        replay_target, "Sent C++ start command", publisher_timeout
    ):
        raise RuntimeError(
            "Token publisher did not request streamed control in time. "
            f"Inspect: tmux attach -t {SESSION_NAME}"
        )
    if not _wait_for_pane_text(deploy_target, "ZMQ streaming enabled", 10.0):
        raise RuntimeError(
            "C++ deploy did not switch to streamed mode. "
            f"Inspect: tmux attach -t {SESSION_NAME}"
        )

    # ZMQManager currently consumes the network start flag while switching modes
    # instead of forwarding it to ZMQEndpointInterface. Its documented `]` key
    # starts policy control, so inject that key after the streamed-mode switch.
    _tmux("send-keys", "-t", deploy_target, "]")
    if not _wait_for_pane_text(deploy_target, "transitioning to CONTROL", 10.0):
        raise RuntimeError(
            "C++ SONIC policy did not enter CONTROL after the start key. "
            f"Inspect: tmux attach -t {SESSION_NAME}"
        )

    _tmux("select-window", "-t", f"{SESSION_NAME}:replay")
    _tmux("select-pane", "-t", replay_target)

    print()
    print("SONIC token replay stack is ready:")
    print(f"  tmux attach -t {SESSION_NAME}")
    print("  window replay, pane 0: C++ SONIC decoder")
    print("  window replay, pane 1: dataset token publisher")
    print("  window sim: MuJoCo")
    print("  C++ SONIC policy state: CONTROL")
    print("  Ctrl+b then n/p switches windows; Ctrl+\\ terminates this replay session.")
    if args.attach:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME], check=False)


def _run_token_mode(args: argparse.Namespace, episode: EpisodeData) -> None:
    _validate_token_episode(args, episode)
    if args.validate_only:
        if args.launch_stack:
            checkpoint, obs_config, planner = _deploy_paths(args, episode)
            print(f"Resolved checkpoint: {checkpoint}")
            print(f"Resolved observation config: {obs_config}")
            print(f"Resolved planner: {planner}")
        print("Validation passed: token replay data is compatible.")
        return
    if args.launch_stack:
        _create_token_session(args, episode)
    else:
        _run_token_publisher(args, episode)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.subscriber_settle_time < 0 or args.start_hold_time < 0:
        parser.error("--subscriber-settle-time and --start-hold-time must be non-negative")
    if args.warmup_frames < 0 or args.deploy_ready_timeout <= 0:
        parser.error("--warmup-frames must be non-negative and --deploy-ready-timeout positive")
    episode = _load_episode(args)
    if args.mode == "joint":
        _run_joint_replay(args, episode)
    else:
        _run_token_mode(args, episode)


def _signal_handler(_signal: int, _frame: Any) -> None:
    print("Replay launcher interrupted; the tmux session is left running for inspection.")
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main()
