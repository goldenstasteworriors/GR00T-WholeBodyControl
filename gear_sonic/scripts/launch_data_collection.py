"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in a single tmux session:

    Window 0 — data_collection (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    ├───────────────────────┼───────────────────────┤
    │ Pane 1: Teleop        │ Pane 3: Camera Viewer │
    │ (.venv_teleop)        │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_pico.sh            -> .venv_teleop
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_data_collection.py                          # real robot (default)
    python gear_sonic/scripts/launch_data_collection.py --sim                    # MuJoCo sim
    python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer       # skip viewer
    python gear_sonic/scripts/launch_data_collection.py --pico-input-source isaac-teleop  # in-process CloudXR / DeviceIO
    python gear_sonic/scripts/launch_data_collection.py --deploy-onboard --camera-host 192.168.123.164
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time


DEFAULT_INSPIRE_HAND_POSE_CONFIG = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "data_collection"
    / "inspire_hand_pose.json"
)


def _bootstrap_venv():
    """Re-exec with the .venv_data_collection Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_data_collection not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    deploy_onboard: bool = False
    """Run C++ deploy over SSH on the G1 onboard computer instead of on this PC."""

    deploy_onboard_host: str = ""
    """G1 onboard host/IP for SSH and robot-state ZMQ. Defaults to camera_host."""

    deploy_onboard_user: str = "unitree"
    """SSH user for onboard deployment."""

    deploy_onboard_repo_root: str = "/home/unitree/data_collection/GR00T-WholeBodyControl"
    """Repository root on the G1 onboard computer."""

    offboard_zmq_host: str = ""
    """PC host/IP that onboard deploy should use to reach the PICO ZMQ server.
    Defaults to this PC's detected LAN IP when --deploy-onboard is set."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    pico_zmq_feedback_host: str = ""
    """Host for PICO frozen-target feedback (g1_debug). Defaults to state_zmq_host."""

    pico_zmq_feedback_port: int = 5557
    """Port for PICO frozen-target feedback (g1_debug)."""

    # Inspire hand bridge options
    inspire_hand_bridge: bool = True
    """Start decoupled_wbc/scripts/inspire_modbus_hand.py DDS->Modbus bridge."""

    restart_inspire_hand_bridge: bool = True
    """Kill existing inspire_modbus_hand.py --mode dds processes before launching the bridge."""

    inspire_hand_network: str = "enp7s0"
    """DDS network interface for the Inspire hand bridge."""

    inspire_left_ip: str = "192.168.123.210"
    """Left Inspire hand Modbus TCP IP."""

    inspire_right_ip: str = "192.168.123.211"
    """Right Inspire hand Modbus TCP IP."""

    inspire_hand_pose_config: str = str(DEFAULT_INSPIRE_HAND_POSE_CONFIG)
    """JSON file controlling PICO-trigger hand open/grasp poses for the Inspire bridge."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    overwrite_existing_dataset: bool = False
    """Delete and recreate the dataset directory if it already exists."""

    sonic_zmq_host: str = "localhost"
    """Host for SMPL/pose ZMQ from pico_manager_thread_server."""

    sonic_zmq_port: int = 5556
    """Port for SMPL/pose ZMQ from pico_manager_thread_server."""

    state_zmq_host: str = ""
    """Host for robot state/config ZMQ from C++ deploy. Defaults to localhost, or onboard host."""

    state_zmq_port: int = 5557
    """Port for robot state/config ZMQ from C++ deploy."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback via espeak (data exporter)."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host (shared by data exporter and viewer)."""

    camera_port: int = 5555
    """Camera server port (shared by data exporter and viewer)."""

    profile_timing: bool = False
    """Enable periodic Python-side timing logs in camera viewer and data exporter."""

    profile_interval: float = 1.0
    """Seconds between timing profile log lines."""


SESSION_NAME = "sonic_data_collection"


def _check_prerequisites(config: DataCollectionLaunchConfig):
    """Verify that required tools and venvs exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if not (repo_root / ".venv_data_collection" / "bin" / "activate").exists():
        errors.append(
            ".venv_data_collection not found. Run: "
            "bash install_scripts/install_data_collection.sh"
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not config.deploy_onboard and not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if config.deploy_onboard and config.sim:
        errors.append("--deploy-onboard is only supported for the real robot, not --sim")

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(see install instructions)."
        )

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _kill_existing_inspire_hand_bridge():
    """Stop stale DDS->Modbus bridge processes so the new hand pose config is used."""
    subprocess.run(
        [
            "pkill",
            "-f",
            "decoupled_wbc/scripts/inspire_modbus_hand.py.*--mode dds",
        ],
        capture_output=True,
    )


def _create_tmux_session():
    """Create a 4-pane tmux layout."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "data_collection"],
    )

    # Split into 4 panes:
    #   0 | 1
    #   -----
    #   2 | 3

    # Split horizontally: pane 0 (left) and pane 1 (right)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )

    # Split left pane vertically: pane 0 (top-left) and pane 2 (bottom-left)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )

    # Split right pane vertically: pane 1 becomes top-right, new pane 3 bottom-right
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def _build_deploy_args(
    config: DataCollectionLaunchConfig, zmq_host: str, deploy_mode: str
) -> list[str]:
    """Build deploy.sh arguments shared by local and onboard deploy."""
    args = [
        "./deploy.sh",
        "--input-type",
        config.deploy_input_type,
        "--zmq-host",
        zmq_host,
    ]
    if config.deploy_checkpoint:
        args += ["--cp", config.deploy_checkpoint]
    if config.deploy_obs_config:
        args += ["--obs-config", config.deploy_obs_config]
    if config.deploy_planner:
        args += ["--planner", config.deploy_planner]
    if config.deploy_motion_data:
        args += ["--motion-data", config.deploy_motion_data]
    if config.deploy_output_type:
        args += ["--output-type", config.deploy_output_type]
    args.append(deploy_mode)
    return args


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent
    local_ip = _get_local_ip()

    _check_prerequisites(config)
    _kill_existing_session()

    onboard_host = config.deploy_onboard_host or config.camera_host
    state_zmq_host = config.state_zmq_host or (
        onboard_host if config.deploy_onboard else "localhost"
    )
    pico_feedback_host = config.pico_zmq_feedback_host or state_zmq_host
    deploy_zmq_host = (
        (config.offboard_zmq_host or local_ip)
        if config.deploy_onboard
        else config.deploy_zmq_host
    )

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Deploy location: {'G1 onboard' if config.deploy_onboard else 'Local PC'}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Deploy ZMQ host: {deploy_zmq_host}")
    if config.deploy_onboard:
        print(f"  Onboard host:    {onboard_host}")
        print(f"  Onboard repo:    {config.deploy_onboard_repo_root}")
    print(f"  Teleop input:    {config.pico_input_source}")
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Sonic ZMQ:       {config.sonic_zmq_host}:{config.sonic_zmq_port}")
    print(f"  State ZMQ:       {state_zmq_host}:{config.state_zmq_port}")
    print(f"  PICO feedback:   {pico_feedback_host}:{config.pico_zmq_feedback_port}")
    print(f"  Inspire bridge:  {'Yes' if config.inspire_hand_bridge else 'No'}")
    if config.inspire_hand_bridge:
        print(f"  Inspire config:  {config.inspire_hand_pose_config}")
    print(f"  DC frequency:    {config.data_exporter_frequency} Hz")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Profile timing:  {'Yes' if config.profile_timing else 'No'}")
    print(f"  Overwrite data:  {'Yes' if config.overwrite_existing_dataset else 'No'}")
    print(f"  Text-to-speech:  {'Yes' if config.text_to_speech else 'No'}")
    print(f"  PC IP (for PICO): {local_ip}")
    print(f"  Teleop vis:      vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
    print("=" * 60)

    _create_tmux_session()
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window: Inspire hand DDS->Modbus bridge ---
    if config.inspire_hand_bridge and not config.sim:
        if config.restart_inspire_hand_bridge:
            _kill_existing_inspire_hand_bridge()
            time.sleep(0.5)

        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "inspire_hand"],
        )
        inspire_cmd = (
            f"cd {repo_root} && "
            f"source .venv_teleop/bin/activate && "
            f"python decoupled_wbc/scripts/inspire_modbus_hand.py "
            f"--mode dds "
            f"--network {shlex.quote(config.inspire_hand_network)} "
            f"--left-ip {shlex.quote(config.inspire_left_ip)} "
            f"--right-ip {shlex.quote(config.inspire_right_ip)} "
            f"--hand-pose-config {shlex.quote(config.inspire_hand_pose_config)} "
            f"--dds-pose-mode profile"
        )
        hand_target = f"{SESSION_NAME}:inspire_hand"
        subprocess.run(
            ["tmux", "send-keys", "-t", hand_target, inspire_cmd, "C-m"],
        )
        print("Starting Inspire hand bridge (window: inspire_hand)...")
        time.sleep(1.0)
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        # Switch back to the data_collection window for the remaining panes
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_args = _build_deploy_args(config, deploy_zmq_host, deploy_mode)
    if config.deploy_onboard:
        ssh_target = f"{config.deploy_onboard_user}@{onboard_host}"
        remote_cmd = (
            f"cd {shlex.quote(config.deploy_onboard_repo_root + '/gear_sonic_deploy')} && "
            f"{_shell_join(deploy_args)}"
        )
        deploy_cmd = f"ssh -t {shlex.quote(ssh_target)} {shlex.quote(remote_cmd)}"
    else:
        deploy_cmd = (
            f"cd {shlex.quote(str(repo_root / 'gear_sonic_deploy'))} && "
            f"{_shell_join(deploy_args)}"
        )

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): Teleop Streamer ---
    pico_cmd = (
        f"cd {repo_root} && "
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source} "
        f"--zmq_feedback_host {pico_feedback_host} "
        f"--zmq_feedback_port {config.pico_zmq_feedback_port}"
    )
    if config.pico_manager:
        pico_cmd += " --manager"
    if config.pico_vis_vr3pt:
        pico_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_cmd += " --waist_tracking"

    print("Starting teleop streamer (pane 2)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Camera Viewer ---
    if config.camera_viewer:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_camera_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        if config.profile_timing:
            viewer_cmd += f" --profile-timing --profile-interval {config.profile_interval}"
        print("Starting camera viewer (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)

    # --- Pane 1 (top-right): Data Exporter ---
    exporter_cmd = (
        f"cd {repo_root} && "
        f"source .venv_data_collection/bin/activate && "
        f"python gear_sonic/scripts/run_data_exporter.py "
        f"--task-prompt '{config.task_prompt}' "
        f"--data-collection-frequency {config.data_exporter_frequency} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port} "
        f"--sonic-zmq-host {config.sonic_zmq_host} "
        f"--sonic-zmq-port {config.sonic_zmq_port} "
        f"--state-zmq-host {state_zmq_host} "
        f"--state-zmq-port {config.state_zmq_port}"
    )
    if config.profile_timing:
        exporter_cmd += f" --profile-timing --profile-interval {config.profile_interval}"
    if config.dataset_name:
        exporter_cmd += f" --dataset-name '{config.dataset_name}'"
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if config.overwrite_existing_dataset:
        exporter_cmd += " --overwrite-existing-dataset"
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 1)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Select the data exporter pane so the user lands there for interactive input
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    if config.inspire_hand_bridge and not config.sim:
        print("  Window 'inspire_hand':")
        print("    Inspire DDS -> Modbus bridge (.venv_teleop)")
        print()
    print("  Window 'data_collection':")
    print(
        "    Pane 0 (top-left):     C++ Deploy"
        + (" (SSH to G1 onboard)" if config.deploy_onboard else "")
    )
    print("    Pane 1 (bottom-left):  Teleop Streamer")
    print("    Pane 2 (top-right):    Data Exporter  <-- you are here")
    if config.camera_viewer:
        print("    Pane 3 (bottom-right): Camera Viewer")
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DataCollectionLaunchConfig)
    main(config)
