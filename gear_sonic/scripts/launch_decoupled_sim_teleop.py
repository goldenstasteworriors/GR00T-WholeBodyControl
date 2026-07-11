"""
One-command tmux launcher for decoupled WBC teleop in MuJoCo simulation.

Usage from repo root:
    python gear_sonic/scripts/launch_decoupled_sim_teleop.py

This starts:
    1. decoupled_wbc/control/main/teleop/run_sim_loop.py
    2. decoupled_wbc/control/main/teleop/run_g1_control_loop.py
    3. decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


SESSION_NAME = "decoupled_sim_teleop"
DEFAULT_CONDA_ENV = "decoupled_vla_collection"
DEFAULT_WBC_MODEL_PATH = (
    "policy/GR00T-WholeBodyControl-Balance.onnx,"
    "policy/GR00T-WholeBodyControl-Walk.onnx"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _bool_arg(name: str, enabled: bool) -> list[str]:
    return [f"--{name}" if enabled else f"--no-{name}"]


def _conda_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CONDA_NO_PLUGINS"] = "true"
    return env


def _conda_prefix(repo_root: Path, conda_env: str) -> str:
    conda_base = subprocess.check_output(
        ["conda", "info", "--base"], env=_conda_env(), text=True
    ).strip()
    env_prefix = Path(conda_base) / "envs" / conda_env
    repo = shlex.quote(str(repo_root))
    env_path = shlex.quote(str(env_prefix))
    env_bin = shlex.quote(str(env_prefix / "bin"))
    return (
        "export CONDA_NO_PLUGINS=true && "
        f"export CONDA_PREFIX={env_path} && "
        f"export CONDA_DEFAULT_ENV={shlex.quote(conda_env)} && "
        f"export PATH={env_bin}:$PATH && "
        'if [ -f "$CONDA_PREFIX/setup.bash" ]; then source "$CONDA_PREFIX/setup.bash"; fi && '
        f"cd {repo} && "
        "export PYTHONUNBUFFERED=1 && "
    )


def _check_prerequisites(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    errors: list[str] = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")
    if not shutil.which("conda"):
        errors.append("conda is not on PATH")

    required_files = [
        repo_root / "decoupled_wbc/control/main/teleop/run_sim_loop.py",
        repo_root / "decoupled_wbc/control/main/teleop/run_g1_control_loop.py",
        repo_root / "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py",
    ]
    for path in required_files:
        if not path.exists():
            errors.append(f"Required script not found: {path}")

    if shutil.which("conda"):
        conda_base = subprocess.check_output(
            ["conda", "info", "--base"], env=_conda_env(), text=True
        ).strip()
        env_prefix = Path(conda_base) / "envs" / args.conda_env
        if not env_prefix.exists():
            errors.append(
                f"Conda env '{args.conda_env}' not found at {env_prefix}. "
                "Create it with: bash install_scripts/install_decoupled_vla_collection.sh"
            )

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
    subprocess.run(["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "sim_teleop"])

    # Pane layout:
    #   0: simulator
    #   1: control loop
    #   2: teleop loop
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"])
    time.sleep(1.0)


def _send_to_pane(pane_index: int, command: str, wait: float = 1.0) -> None:
    target = f"{SESSION_NAME}:0.{pane_index}"
    subprocess.run(["tmux", "send-keys", "-t", target, command, "C-m"])
    time.sleep(wait)


def _build_sim_args(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "decoupled_wbc/control/main/teleop/run_sim_loop.py",
        "--interface",
        "sim",
        "--simulator",
        args.simulator,
        "--env-name",
        args.env_name,
        "--mp-start-method",
        args.mp_start_method,
        "--camera-port",
        str(args.camera_port),
        *_bool_arg("enable-image-publish", args.enable_image_publish),
        *_bool_arg("enable-offscreen", args.enable_offscreen),
        *_bool_arg("enable-onscreen", args.enable_onscreen),
        *_bool_arg("enable-waist", args.enable_waist),
        *_bool_arg("with-hands", args.with_hands),
        *_bool_arg("high-elbow-pose", args.high_elbow_pose),
    ]


def _build_control_args(args: argparse.Namespace) -> list[str]:
    return [
        "python",
        "decoupled_wbc/control/main/teleop/run_g1_control_loop.py",
        "--interface",
        "sim",
        "--simulator",
        "none",
        "--env-name",
        args.env_name,
        "--wbc-version",
        args.wbc_version,
        "--wbc-model-path",
        args.wbc_model_path,
        "--control-frequency",
        str(args.control_frequency),
        "--upper-body-joint-speed",
        str(args.upper_body_joint_speed),
        "--keyboard-dispatcher-type",
        args.keyboard_dispatcher_type,
        *_bool_arg("enable-waist", args.enable_waist),
        *_bool_arg("with-hands", args.with_hands),
        *_bool_arg("high-elbow-pose", args.high_elbow_pose),
    ]


def _build_teleop_args(args: argparse.Namespace) -> list[str]:
    teleop_args = [
        "python",
        "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py",
        "--interface",
        "sim",
        "--simulator",
        "none",
        "--env-name",
        args.env_name,
        "--wbc-version",
        args.wbc_version,
        "--body-control-device",
        args.body_control_device,
        "--body-streamer-ip",
        args.body_streamer_ip,
        "--body-streamer-keyword",
        args.body_streamer_keyword,
        "--teleop-frequency",
        str(args.teleop_frequency),
        *_bool_arg("enable-waist", args.enable_waist),
        *_bool_arg("with-hands", args.with_hands),
        *_bool_arg("high-elbow-pose", args.high_elbow_pose),
        *_bool_arg("enable-visualization", args.enable_visualization),
        *_bool_arg("enable-real-device", args.enable_real_device),
    ]
    if args.with_hands and args.hand_control_device:
        teleop_args += ["--hand-control-device", args.hand_control_device]
    return teleop_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch decoupled WBC MuJoCo sim + control + teleop in one tmux session."
    )
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--simulator", default="mujoco")
    parser.add_argument("--env-name", default="default")
    parser.add_argument("--wbc-version", default="gear_wbc")
    parser.add_argument("--wbc-model-path", default=DEFAULT_WBC_MODEL_PATH)
    parser.add_argument("--control-frequency", type=int, default=50)
    parser.add_argument("--teleop-frequency", type=float, default=20.0)
    parser.add_argument("--upper-body-joint-speed", type=float, default=1000.0)
    parser.add_argument("--keyboard-dispatcher-type", default="raw")
    parser.add_argument("--body-control-device", default="pico")
    parser.add_argument("--hand-control-device", default="pico")
    parser.add_argument("--body-streamer-ip", default="10.110.67.24")
    parser.add_argument("--body-streamer-keyword", default="foot")
    parser.add_argument("--mp-start-method", default="spawn")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--enable-waist", action="store_true")
    parser.add_argument("--with-hands", dest="with_hands", action="store_true")
    parser.add_argument("--no-with-hands", dest="with_hands", action="store_false")
    parser.set_defaults(with_hands=True)
    parser.add_argument("--high-elbow-pose", action="store_true")
    parser.add_argument("--enable-visualization", action="store_true")
    parser.add_argument(
        "--no-enable-real-device",
        dest="enable_real_device",
        action="store_false",
        help="Disable the real PICO/device stream in the teleop loop.",
    )
    parser.set_defaults(enable_real_device=True)
    parser.add_argument("--no-enable-onscreen", dest="enable_onscreen", action="store_false")
    parser.set_defaults(enable_onscreen=True)
    parser.add_argument("--enable-offscreen", action="store_true")
    parser.add_argument("--enable-image-publish", action="store_true")
    parser.add_argument(
        "--auto-enable-lower-body-policy",
        action="store_true",
        help="Send ']' to the control loop after launch so the standing/walking policy controls the lower body.",
    )
    parser.add_argument(
        "--auto-enable-lower-body-delay",
        type=float,
        default=8.0,
        help="Seconds to wait before auto-enabling the lower-body policy.",
    )
    parser.add_argument(
        "--auto-activate-teleop",
        action="store_true",
        help="Send 'l' after launch to activate and calibrate upper-body teleop.",
    )
    parser.add_argument(
        "--auto-activate-teleop-delay",
        type=float,
        default=10.0,
        help="Seconds to wait before auto-activating upper-body teleop.",
    )
    parser.add_argument(
        "--no-attach",
        dest="attach",
        action="store_false",
        help="Launch tmux in the background and do not attach.",
    )
    parser.set_defaults(attach=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()

    if args.enable_image_publish and not args.enable_offscreen:
        args.enable_offscreen = True

    _check_prerequisites(args)
    _kill_existing_session()

    prefix = _conda_prefix(repo_root, args.conda_env)

    print("=" * 64)
    print("  Decoupled WBC Sim Teleop Launcher")
    print("=" * 64)
    print(f"  tmux session:   {SESSION_NAME}")
    print(f"  conda env:      {args.conda_env}")
    print(f"  simulator:      {args.simulator}")
    print(f"  env name:       {args.env_name}")
    print(f"  WBC version:    {args.wbc_version}")
    print(f"  WBC model path: {args.wbc_model_path}")
    hand_device = args.hand_control_device if args.with_hands else "disabled"
    print(f"  teleop device:  body={args.body_control_device} hand={hand_device}")
    print(f"  hands model:    {'Dex3-1' if args.with_hands else 'disabled'}")
    print(f"  PICO/device:    {'enabled' if args.enable_real_device else 'disabled'}")
    print("=" * 64)

    _create_tmux_session()

    print("Starting MuJoCo simulator (pane 0)...")
    _send_to_pane(0, prefix + _shell_join(_build_sim_args(args)), wait=3.0)

    print("Starting decoupled G1 control loop (pane 1)...")
    _send_to_pane(1, prefix + _shell_join(_build_control_args(args)), wait=2.0)

    print("Starting decoupled teleop loop (pane 2)...")
    _send_to_pane(2, prefix + _shell_join(_build_teleop_args(args)), wait=1.0)

    if args.auto_enable_lower_body_policy:
        print(
            "Auto-enabling lower-body standing policy "
            f"in {args.auto_enable_lower_body_delay:.1f}s..."
        )
        time.sleep(max(0.0, args.auto_enable_lower_body_delay))
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:0.1", "]"])

    if args.auto_activate_teleop:
        print(f"Auto-activating upper-body teleop in {args.auto_activate_teleop_delay:.1f}s...")
        time.sleep(max(0.0, args.auto_activate_teleop_delay))
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:0.1", "l"])

    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.1"])

    print()
    print("=" * 64)
    print("  All components launched.")
    print()
    print("  Panes:")
    print("    Pane 0 (top-left):     MuJoCo simulator")
    print("    Pane 1 (right):        G1 control loop  <-- selected")
    print("    Pane 2 (bottom-left):  PICO/teleop loop")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch panes")
    print("    Ctrl+b, d           - Detach")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 64)

    if args.attach:
        try:
            subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
        except KeyboardInterrupt:
            pass

    result = subprocess.run(["tmux", "has-session", "-t", SESSION_NAME], capture_output=True)
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach: tmux attach -t {SESSION_NAME}")
        print(f"  Kill:     tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig: int, _frame: object) -> None:
    print("\nShutdown requested...")
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], capture_output=True)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main()
