"""
tmux launcher for decoupled WBC + Sonic VLA data collection.

This launcher is intentionally non-Docker. It starts the decoupled ROS control
loop, decoupled PICO teleop loop, the Sonic-VLA-compatible decoupled exporter,
and optionally a PICO pose streamer for SMPL/VR3PT metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


DEFAULT_CONDA_ENV = "decoupled_vla_collection"
SESSION_NAME = "decoupled_vla_collection"


def _bootstrap_tyro() -> None:
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    conda = shutil.which("conda")
    env_name = os.environ.get("DECOUPLED_VLA_CONDA_ENV", DEFAULT_CONDA_ENV)
    if conda is None or os.environ.get("CONDA_DEFAULT_ENV") == env_name:
        print(
            "ERROR: tyro is not available. Install the collection environment first:\n"
            "  bash install_scripts/install_decoupled_vla_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with conda env {env_name!r} ...")
    os.execvp(
        conda,
        [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            env_name,
            "python",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
    )


_bootstrap_tyro()

import tyro


def _sanitize_log_name(name: str) -> str:
    name = name.strip() or time.strftime("%Y%m%d_%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or time.strftime("%Y%m%d_%H%M%S")


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _bool_arg(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _conda_base() -> Path:
    try:
        out = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
    except Exception as exc:
        raise RuntimeError("conda is required for this launcher") from exc
    return Path(out)


def _conda_prefix(repo_root: Path, env_name: str) -> str:
    conda_hook = _conda_base() / "etc" / "profile.d" / "conda.sh"
    return (
        # The G1 login shell sources ROS 1 Noetic from ~/.bashrc.  tmux panes
        # inherit those paths before this ROS 2 environment is activated,
        # which can make ROS 2 messages look like ROS 1 types and can exhaust
        # aarch64 static TLS while importing torch.  Clean only this pane's
        # process environment; do not change the user's shell configuration.
        "unset ROS_ROOT ROS_PACKAGE_PATH ROS_MASTER_URI "
        "ROSLISP_PACKAGE_DIRECTORIES ROS_DISTRO ROS_VERSION "
        "ROS_PYTHON_VERSION ROS_ETC_DIR AMENT_PREFIX_PATH "
        "COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH && "
        'export PATH="${PATH//\\/opt\\/ros\\/noetic\\/bin:/}" && '
        'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\\/opt\\/ros\\/noetic\\/lib\\/aarch64-linux-gnu:/}" && '
        'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//\\/opt\\/ros\\/noetic\\/lib:/}" && '
        f"source {shlex.quote(str(conda_hook))} && "
        f"conda activate {shlex.quote(env_name)} && "
        'if [ -f "$CONDA_PREFIX/setup.bash" ]; then source "$CONDA_PREFIX/setup.bash"; fi && '
        f"cd {shlex.quote(str(repo_root))} && "
        "export PYTHONUNBUFFERED=1 && "
    )


def _default_hand_task_config(repo_root: Path) -> Path:
    return repo_root / "gear_sonic/config/data_collection/inspire_hand_tasks.json"


def _aarch64_control_runtime_prefix() -> str:
    """Select the ARM64 DDS runtime and preload PyTorch's OpenMP library."""
    return (
        'if [ "$(uname -m)" = aarch64 ]; then '
        'if [ -f "$HOME/cyclonedds/install/lib/libddsc.so" ]; then '
        'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"; '
        "fi; "
        'SONIC_TORCH_LIBGOMP="$(find "$CONDA_PREFIX/lib" '
        "-path '*/torch.libs/libgomp*.so*' -print -quit)\"; "
        'if [ -n "$SONIC_TORCH_LIBGOMP" ]; then '
        'export LD_PRELOAD="$SONIC_TORCH_LIBGOMP${LD_PRELOAD:+:$LD_PRELOAD}"; '
        "fi; fi && "
    )


def _runtime_env_prefix(repo_root: Path, config: "DecoupledVLACollectionLaunchConfig") -> str:
    hand_task_config = Path(config.hand_task_config).expanduser() if config.hand_task_config else _default_hand_task_config(repo_root)
    return (
        f"export SONIC_HAND_TASK={shlex.quote(config.hand_task)} && "
        f"export SONIC_HAND_TASK_CONFIG={shlex.quote(str(hand_task_config))} && "
    )


def _allocate_profile_log_dir(repo_root: Path, dataset_name: str, base_dir: str) -> Path:
    safe_dataset = _sanitize_log_name(dataset_name)
    root = repo_root / base_dir
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(10000):
        candidate = root / f"{safe_dataset}_{idx:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not allocate profile log directory under {root}")


def _pipe_pane_to_log(target: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tmux", "pipe-pane", "-o", "-t", target, f"cat >> {shlex.quote(str(log_file))}"],
        check=True,
    )


@dataclass
class DecoupledVLACollectionLaunchConfig:
    """Launch decoupled WBC data collection into the Sonic VLA dataset schema."""

    sim: bool = False
    """Run against MuJoCo sim. Default is real robot."""

    conda_env: str = DEFAULT_CONDA_ENV
    """Conda environment that contains decoupled_wbc, gear_sonic, ROS2 and PICO deps."""

    interface: str = ""
    """Control interface. Defaults to sim when --sim is set, otherwise real."""

    simulator: str = "mujoco"
    """Simulator backend for run_sim_loop.py."""

    env_name: str = "default"
    """Decoupled WBC environment name."""

    control_frequency: int = 50
    """Decoupled control-loop frequency."""

    teleop_frequency: float = 50.0
    """Decoupled teleop-loop frequency."""

    wbc_version: str = "gear_wbc"
    """Decoupled WBC version."""

    lower_body_controller: str = "decoupled"
    """Use ``decoupled`` ONNX control or Unitree's official ``unitree_loco`` service."""

    unitree_loco_start_fsm_id: int = 500
    """Official Start FSM used by the G1 sport service; negative skips the explicit transition."""

    unitree_loco_damp_fsm_id: int = 1
    """Official damping FSM used to begin startup."""

    unitree_loco_stand_fsm_id: int = 4
    """Official StandUp FSM used by the G1 sport service."""

    unitree_loco_service_name: str = "sport"
    """Official loco RPC service name."""

    unitree_loco_damp_duration: float = 0.5
    """Minimum Damp hold time in seconds."""

    unitree_loco_stand_duration: float = 4.0
    """Minimum StandUp transition time in seconds."""

    unitree_loco_stability_duration: float = 0.5
    """Continuous stability time required before confirmation."""

    unitree_loco_activation_timeout: float = 15.0
    """Overall official-loco startup timeout in seconds."""

    unitree_loco_state_timeout: float = 0.5
    """Maximum accepted age of measured robot state in seconds."""

    unitree_loco_max_leg_velocity: float = 0.35
    """Maximum settled leg speed in rad/s."""

    unitree_loco_max_torso_tilt: float = 0.35
    """Maximum upright torso tilt in radians."""

    wbc_model_path: str = (
        "policy/GR00T-WholeBodyControl-Balance.onnx,"
        "policy/GR00T-WholeBodyControl-Walk.onnx"
    )
    """WBC ONNX model path passed to decoupled control."""

    enable_waist: bool = False
    """Enable waist joints in decoupled IK."""

    with_hands: bool = True
    """Enable hand functionality in decoupled WBC."""

    high_elbow_pose: bool = False
    """Use high elbow pose in robot model."""

    upper_body_joint_speed: float = 1000.0
    """Upper-body speed limit for decoupled WBC."""

    startup_t_pose: bool = False
    """Route the startup arm ramp through a verified G1 T-pose waypoint."""

    startup_t_pose_duration: float = 4.0
    """Seconds from the measured startup pose to the T-pose waypoint."""

    startup_elbow_pose_duration: float = 4.0
    """Seconds to move only the elbows from T-pose to their final startup angles."""

    startup_final_pose_duration: float = 4.0
    """Seconds to move all remaining joints to the normal initial pose."""

    startup_final_elbow_angle: float = -0.34906585
    """Final startup elbow angle; raises both elbows 20 degrees from zero."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher type for control loop."""

    keyboard_lower_body_control: bool = False
    """Control official lower-body startup and navigation from the control pane."""

    keyboard_loco_command_timeout: float = 0.5
    """Keyboard movement deadman timeout in seconds."""

    body_control_device: str = "pico"
    """Decoupled body teleop device."""

    hand_control_device: str = "pico"
    """Decoupled hand teleop device."""

    enable_visualization: bool = False
    """Enable decoupled teleop visualization."""

    enable_real_device: bool = True
    """Connect the teleop loop to the real PICO/device stream."""

    body_streamer_ip: str = "10.110.67.24"
    """Body streamer IP for non-PICO devices; kept for config compatibility."""

    body_streamer_keyword: str = "foot"
    """Body streamer keyword for non-PICO devices; kept for config compatibility."""

    # Sim camera process
    sim_separate_process: bool = True
    """Start run_sim_loop.py separately for sim, enabling image publishing."""

    sim_enable_offscreen: bool = True
    """Enable offscreen rendering in the sim loop."""

    sim_enable_onscreen: bool = True
    """Enable onscreen rendering in the sim loop."""

    sim_mp_start_method: str = "spawn"
    """Multiprocessing start method for sim image publishing."""

    # Dataset/exporter
    task_prompt: str = "demo"
    """Language task prompt for the dataset."""

    dataset_name: str = ""
    """Dataset name. Empty means timestamp."""

    root_output_dir: str = "outputs"
    """Root output dir for the dataset."""

    data_exporter_frequency: int = 50
    """Exporter frequency in Hz."""

    overwrite_existing_dataset: bool = False
    """Delete existing dataset dir before recording."""

    require_sonic_pose: bool = False
    """Skip frames until fresh PICO/SMPL ZMQ pose is available."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams if provided by the camera server."""

    text_to_speech: bool = False
    """Enable optional voice feedback in the exporter; tone cues remain enabled by default."""

    audio_cues: bool = True
    """Enable local tone cues in the exporter."""

    audio_cue_volume: float = 0.35
    """Volume for exporter start/stop tone cues."""

    discard_audio_cue_volume: float = 0.9
    """Volume for exporter discard tone cue."""

    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    camera_viewer: bool = False
    """Start camera viewer pane."""

    # Optional PICO metadata streamer
    pico_data_streamer: bool = True
    """Start gear_sonic PICO pose streamer for SMPL/VR3PT dataset fields."""

    pico_data_manager: bool = False
    """Run pico_manager_thread_server.py with --manager."""

    pico_input_source: str = "xrt"
    """PICO metadata streamer input source: xrt or isaac-teleop."""

    hand_task: str = "pick_up_pipette"
    """Hand task passed to pico_manager_thread_server.py."""

    hand_task_config: str = ""
    """Optional inspire_hand_tasks.json path passed to pico_manager_thread_server.py."""

    pico_vis_vr3pt: bool = False
    """Enable VR3PT visualization in the PICO metadata streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization in the PICO metadata streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking in the PICO metadata streamer."""

    pico_zmq_feedback_host: str = "localhost"
    """Feedback host for PICO manager mode. Usually unused in legacy pose mode."""

    pico_zmq_feedback_port: int = 5557
    """Feedback port for PICO manager mode."""

    sonic_zmq_host: str = "localhost"
    """Exporter host for PICO/SMPL ZMQ data."""

    sonic_zmq_port: int = 5556
    """PICO/SMPL ZMQ port."""

    # Logging
    profile_timing: bool = False
    """Enable exporter/viewer timing logs."""

    profile_interval: float = 1.0
    """Seconds between profile logs."""

    save_profile_logs: bool = False
    """Pipe tmux pane output to logs."""

    profile_log_dir: str = "logs/decoupled_vla_collection"
    """Base directory for launcher logs."""


def _check_prerequisites(config: DecoupledVLACollectionLaunchConfig, repo_root: Path) -> None:
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")
    if not shutil.which("conda"):
        errors.append("conda is not on PATH")

    if shutil.which("conda"):
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
        )
        if config.conda_env not in result.stdout:
            errors.append(
                f"conda env {config.conda_env!r} not found. Run: "
                "bash install_scripts/install_decoupled_vla_collection.sh"
            )

    required_files = [
        repo_root / "decoupled_wbc/control/main/teleop/run_g1_control_loop.py",
        repo_root / "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py",
        repo_root / "gear_sonic/scripts/run_decoupled_vla_data_exporter.py",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"required file missing: {path}")

    if config.sim and config.sim_separate_process:
        sim_path = repo_root / "decoupled_wbc/control/main/teleop/run_sim_loop.py"
        if not sim_path.exists():
            errors.append(f"required sim loop missing: {sim_path}")

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be xrt or isaac-teleop")

    if config.lower_body_controller not in {"decoupled", "unitree_loco"}:
        errors.append("--lower-body-controller must be decoupled or unitree_loco")
    if config.lower_body_controller == "unitree_loco" and config.sim:
        errors.append("--lower-body-controller unitree_loco is only available on the real robot")
    if config.lower_body_controller == "unitree_loco" and config.enable_waist:
        errors.append("unitree_loco owns the waist; do not use --enable-waist")
    if config.keyboard_lower_body_control:
        if config.lower_body_controller != "unitree_loco":
            errors.append(
                "--keyboard-lower-body-control requires --lower-body-controller unitree_loco"
            )
        if config.keyboard_dispatcher_type != "raw":
            errors.append("--keyboard-lower-body-control requires --keyboard-dispatcher-type raw")
        if config.keyboard_loco_command_timeout <= 0.0:
            errors.append("--keyboard-loco-command-timeout must be positive")
    if config.lower_body_controller == "unitree_loco":
        positive_values = {
            "--unitree-loco-activation-timeout": config.unitree_loco_activation_timeout,
            "--unitree-loco-state-timeout": config.unitree_loco_state_timeout,
            "--unitree-loco-max-leg-velocity": config.unitree_loco_max_leg_velocity,
            "--unitree-loco-max-torso-tilt": config.unitree_loco_max_torso_tilt,
        }
        errors.extend(f"{name} must be positive" for name, value in positive_values.items() if value <= 0.0)
        nonnegative_values = {
            "--unitree-loco-damp-duration": config.unitree_loco_damp_duration,
            "--unitree-loco-stand-duration": config.unitree_loco_stand_duration,
            "--unitree-loco-stability-duration": config.unitree_loco_stability_duration,
        }
        errors.extend(
            f"{name} must be nonnegative"
            for name, value in nonnegative_values.items()
            if value < 0.0
        )

    hand_task_config = Path(config.hand_task_config).expanduser() if config.hand_task_config else _default_hand_task_config(repo_root)
    if not hand_task_config.exists():
        errors.append(f"hand task config missing: {hand_task_config}")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        sys.exit(1)


def _kill_existing_session() -> None:
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)


def _create_tmux_session() -> None:
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION_NAME], check=True)
    subprocess.run(["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"])
    subprocess.run(["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"])
    subprocess.run(["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "collection"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-h"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.1", "-v"])
    subprocess.run(["tmux", "select-layout", "-t", f"{SESSION_NAME}:collection", "tiled"])
    subprocess.run(["tmux", "set-option", "-t", SESSION_NAME, "-g", "pane-border-status", "top"])
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:collection.0", "-T", "control"])
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:collection.1", "-T", "exporter"])
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:collection.2", "-T", "teleop"])
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:collection.3", "-T", "camera_viewer"])
    time.sleep(2)


def _send_to_target(target: str, cmd: str, wait: float = 1.0) -> None:
    subprocess.run(["tmux", "send-keys", "-t", target, cmd, "C-m"])
    time.sleep(wait)


def _build_control_args(config: DecoupledVLACollectionLaunchConfig, interface: str) -> list[str]:
    simulator = "none" if config.sim and config.sim_separate_process else config.simulator
    args = [
        "python",
        "decoupled_wbc/control/main/teleop/run_g1_control_loop.py",
        "--interface",
        interface,
        "--simulator",
        simulator,
        "--env-name",
        config.env_name,
        "--wbc-version",
        config.wbc_version,
        "--lower-body-controller",
        config.lower_body_controller,
        "--unitree-loco-start-fsm-id",
        str(config.unitree_loco_start_fsm_id),
        "--unitree-loco-damp-fsm-id",
        str(config.unitree_loco_damp_fsm_id),
        "--unitree-loco-stand-fsm-id",
        str(config.unitree_loco_stand_fsm_id),
        "--unitree-loco-service-name",
        config.unitree_loco_service_name,
        "--unitree-loco-damp-duration",
        str(config.unitree_loco_damp_duration),
        "--unitree-loco-stand-duration",
        str(config.unitree_loco_stand_duration),
        "--unitree-loco-stability-duration",
        str(config.unitree_loco_stability_duration),
        "--unitree-loco-activation-timeout",
        str(config.unitree_loco_activation_timeout),
        "--unitree-loco-state-timeout",
        str(config.unitree_loco_state_timeout),
        "--unitree-loco-max-leg-velocity",
        str(config.unitree_loco_max_leg_velocity),
        "--unitree-loco-max-torso-tilt",
        str(config.unitree_loco_max_torso_tilt),
        "--wbc-model-path",
        config.wbc_model_path,
        "--control-frequency",
        str(config.control_frequency),
        "--upper-body-joint-speed",
        str(config.upper_body_joint_speed),
        *_bool_arg("startup-t-pose", config.startup_t_pose),
        "--startup-t-pose-duration",
        str(config.startup_t_pose_duration),
        "--startup-elbow-pose-duration",
        str(config.startup_elbow_pose_duration),
        "--startup-final-pose-duration",
        str(config.startup_final_pose_duration),
        "--startup-final-elbow-angle",
        str(config.startup_final_elbow_angle),
        "--keyboard-dispatcher-type",
        config.keyboard_dispatcher_type,
        *_bool_arg("keyboard-lower-body-control", config.keyboard_lower_body_control),
        "--keyboard-loco-command-timeout",
        str(config.keyboard_loco_command_timeout),
        *_bool_arg("enable-waist", config.enable_waist),
        *_bool_arg("with-hands", config.with_hands),
        *_bool_arg("high-elbow-pose", config.high_elbow_pose),
    ]
    return args


def _build_teleop_args(config: DecoupledVLACollectionLaunchConfig, interface: str) -> list[str]:
    simulator = "none" if config.sim and config.sim_separate_process else config.simulator
    args = [
        "python",
        "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py",
        "--interface",
        interface,
        "--simulator",
        simulator,
        "--env-name",
        config.env_name,
        "--wbc-version",
        config.wbc_version,
        "--body-control-device",
        config.body_control_device,
        "--hand-control-device",
        config.hand_control_device,
        "--body-streamer-ip",
        config.body_streamer_ip,
        "--body-streamer-keyword",
        config.body_streamer_keyword,
        "--teleop-frequency",
        str(config.teleop_frequency),
        *_bool_arg("enable-waist", config.enable_waist),
        *_bool_arg("with-hands", config.with_hands),
        *_bool_arg("high-elbow-pose", config.high_elbow_pose),
        *_bool_arg("enable-visualization", config.enable_visualization),
        *_bool_arg("pico-vis-smpl", config.pico_vis_vr3pt or config.pico_vis_smpl),
        *_bool_arg("enable-real-device", config.enable_real_device),
    ]
    return args


def _build_exporter_args(config: DecoupledVLACollectionLaunchConfig) -> list[str]:
    args = [
        "python",
        "gear_sonic/scripts/run_decoupled_vla_data_exporter.py",
        "--task-prompt",
        config.task_prompt,
        "--root-output-dir",
        config.root_output_dir,
        "--data-collection-frequency",
        str(config.data_exporter_frequency),
        "--camera-host",
        config.camera_host,
        "--camera-port",
        str(config.camera_port),
        "--sonic-zmq-host",
        config.sonic_zmq_host,
        "--sonic-zmq-port",
        str(config.sonic_zmq_port),
        "--robot-config-timeout",
        "30",
        *_bool_arg("require-sonic-pose", config.require_sonic_pose),
        *_bool_arg("record-wrist-cameras", config.record_wrist_cameras),
        *_bool_arg("with-hands", config.with_hands),
        *_bool_arg("text-to-speech", config.text_to_speech),
        *_bool_arg("audio-cues", config.audio_cues),
        "--audio-cue-volume",
        str(config.audio_cue_volume),
        "--discard-audio-cue-volume",
        str(config.discard_audio_cue_volume),
        *_bool_arg("overwrite-existing-dataset", config.overwrite_existing_dataset),
        *_bool_arg("profile-timing", config.profile_timing),
        "--profile-interval",
        str(config.profile_interval),
    ]
    if config.dataset_name:
        args += ["--dataset-name", config.dataset_name]
    return args


def _build_sim_args(config: DecoupledVLACollectionLaunchConfig, interface: str) -> list[str]:
    return [
        "python",
        "decoupled_wbc/control/main/teleop/run_sim_loop.py",
        "--interface",
        interface,
        "--simulator",
        config.simulator,
        "--env-name",
        config.env_name,
        "--mp-start-method",
        config.sim_mp_start_method,
        "--camera-port",
        str(config.camera_port),
        *_bool_arg("enable-image-publish", True),
        *_bool_arg("enable-offscreen", config.sim_enable_offscreen),
        *_bool_arg("enable-onscreen", config.sim_enable_onscreen),
        *_bool_arg("enable-waist", config.enable_waist),
        *_bool_arg("with-hands", config.with_hands),
        *_bool_arg("high-elbow-pose", config.high_elbow_pose),
    ]


def _build_pico_data_args(config: DecoupledVLACollectionLaunchConfig) -> list[str]:
    args = [
        "python",
        "gear_sonic/scripts/pico_manager_thread_server.py",
        "--input-source",
        config.pico_input_source,
        "--port",
        str(config.sonic_zmq_port),
        "--hand-task",
        config.hand_task,
        "--zmq_feedback_host",
        config.pico_zmq_feedback_host,
        "--zmq_feedback_port",
        str(config.pico_zmq_feedback_port),
    ]
    if config.hand_task_config:
        args += ["--hand-task-config", config.hand_task_config]
    if config.pico_data_manager:
        args.append("--manager")
    # Visualization consumes the teleop process's PICO frames, so this second
    # XR client remains headless and cannot compete for the display input.
    if config.pico_waist_tracking:
        args.append("--waist_tracking")
    return args


def main(config: DecoupledVLACollectionLaunchConfig) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    interface = config.interface or ("sim" if config.sim else "real")

    _check_prerequisites(config, repo_root)
    _kill_existing_session()

    prefix = _conda_prefix(repo_root, config.conda_env)
    runtime_env = _runtime_env_prefix(repo_root, config)

    print("=" * 72)
    print("  Decoupled WBC -> Sonic VLA Data Collection")
    print("=" * 72)
    print(f"  Mode:          {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Lower body:    {config.lower_body_controller}")
    print(
        "  Keyboard loco: "
        f"{'enabled' if config.keyboard_lower_body_control else 'disabled'}"
    )
    print(f"  Interface:     {interface}")
    print(f"  Dataset:       {config.dataset_name or '(timestamp)'}")
    print(f"  Task:          {config.task_prompt}")
    print(f"  Conda env:     {config.conda_env}")
    print(f"  Camera:        {config.camera_host}:{config.camera_port}")
    print(f"  PICO fields:   {'streamer window enabled' if config.pico_data_streamer else 'disabled'}")
    print(f"  PICO ZMQ:      {config.sonic_zmq_host}:{config.sonic_zmq_port}")
    print(f"  Hand task:     {config.hand_task}")
    print(
        "  Hand config:   "
        f"{config.hand_task_config or _default_hand_task_config(repo_root)}"
    )
    print(f"  Export freq:   {config.data_exporter_frequency} Hz")
    print("=" * 72)

    _create_tmux_session()

    profile_log_dir = None
    if config.save_profile_logs:
        profile_log_dir = _allocate_profile_log_dir(
            repo_root,
            config.dataset_name or config.task_prompt,
            config.profile_log_dir,
        )
        _pipe_pane_to_log(f"{SESSION_NAME}:collection.0", profile_log_dir / "control.log")
        _pipe_pane_to_log(f"{SESSION_NAME}:collection.1", profile_log_dir / "exporter.log")
        _pipe_pane_to_log(f"{SESSION_NAME}:collection.2", profile_log_dir / "teleop.log")
        if config.camera_viewer:
            _pipe_pane_to_log(f"{SESSION_NAME}:collection.3", profile_log_dir / "viewer.log")

    if config.sim and config.sim_separate_process:
        subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"])
        sim_cmd = prefix + runtime_env + _shell_join(_build_sim_args(config, interface))
        _send_to_target(f"{SESSION_NAME}:sim", sim_cmd, wait=3.0)
        subprocess.run(["tmux", "select-window", "-t", f"{SESSION_NAME}:collection"])

    control_cmd = (
        prefix
        + runtime_env
        + _aarch64_control_runtime_prefix()
        + _shell_join(_build_control_args(config, interface))
    )
    teleop_cmd = prefix + runtime_env + _shell_join(_build_teleop_args(config, interface))
    exporter_cmd = prefix + runtime_env + _shell_join(_build_exporter_args(config))
    viewer_cmd = (
        prefix
        + runtime_env
        + _shell_join(
            [
                "python",
                "gear_sonic/scripts/run_camera_viewer.py",
                "--camera-host",
                config.camera_host,
                "--camera-port",
                str(config.camera_port),
                *_bool_arg("profile-timing", config.profile_timing),
                "--profile-interval",
                str(config.profile_interval),
            ]
        )
    )

    _send_to_target(f"{SESSION_NAME}:collection.0", control_cmd, wait=3.0)
    _send_to_target(f"{SESSION_NAME}:collection.2", teleop_cmd, wait=2.0)
    if config.pico_data_streamer:
        subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, "-n", "pico_data"])
        pico_cmd = prefix + runtime_env + _shell_join(_build_pico_data_args(config))
        _send_to_target(f"{SESSION_NAME}:pico_data", pico_cmd, wait=2.0)
        subprocess.run(["tmux", "select-window", "-t", f"{SESSION_NAME}:collection"])
        if profile_log_dir is not None:
            _pipe_pane_to_log(f"{SESSION_NAME}:pico_data.0", profile_log_dir / "pico_data.log")
    if config.camera_viewer:
        _send_to_target(f"{SESSION_NAME}:collection.3", viewer_cmd, wait=1.0)
    _send_to_target(f"{SESSION_NAME}:collection.1", exporter_cmd, wait=1.0)
    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:collection.1"])

    print()
    print("=" * 72)
    print("  All components launched")
    print(f"  tmux attach: tmux attach -t {SESSION_NAME}")
    print()
    print("  collection window:")
    print("    pane 0 top-left:     decoupled control loop")
    print("    pane 1 top-right:    VLA data exporter")
    print("    pane 2 bottom-left:  decoupled teleop loop")
    if config.camera_viewer:
        print("    pane 3 bottom-right: camera viewer")
    if config.sim and config.sim_separate_process:
        print("  sim window:            MuJoCo sim + image publisher")
    if config.pico_data_streamer:
        print("  pico_data window:      optional PICO SMPL/VR3PT streamer")
    if profile_log_dir is not None:
        print(f"  logs:                  {profile_log_dir}")
    print()
    print("  Controls:")
    if config.keyboard_lower_body_control:
        print("    control pane: G start/toggle emergency, Space emergency stop")
        print("    control pane: hold W/S forward, A/D lateral, Q/E yaw; Z zero")
        print("    control pane: C start/save recording, X discard recording")
        print(
            "    movement deadman: zero after "
            f"{config.keyboard_loco_command_timeout:.2f}s without a movement key"
        )
    print("    Ctrl+b, arrows  switch panes")
    print("    Ctrl+b, d       detach")
    print("    Ctrl+\\          kill session")
    print("=" * 72)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME], capture_output=True)
    if result.returncode == 0:
        print(f"\nSession {SESSION_NAME!r} is still running.")
        print(f"  Reattach: tmux attach -t {SESSION_NAME}")
        print(f"  Kill:     tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame) -> None:
    print("\nShutdown requested...")
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main(tyro.cli(DecoupledVLACollectionLaunchConfig))
