"""
All-in-one tmux launcher for SONIC VR teleop in MuJoCo simulation.

Usage from repo root:
    python gear_sonic/scripts/launch_sim_teleop.py

This starts the same three processes that are usually launched manually:
    1. gear_sonic/scripts/run_sim_loop.py
    2. gear_sonic_deploy/deploy.sh --input-type zmq_manager ... sim
    3. gear_sonic/scripts/pico_manager_thread_server.py --manager ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


SESSION_NAME = "sonic_sim_teleop"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _check_prerequisites(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    errors: list[str] = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    teleop_activate = repo_root / args.teleop_venv / "bin" / "activate"
    if not teleop_activate.exists():
        errors.append(f"{args.teleop_venv} not found at {teleop_activate}")

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(f"deploy.sh not found at {deploy_dir / 'deploy.sh'}")
    if not (deploy_dir / "scripts" / "setup_env.sh").exists():
        errors.append(
            f"setup_env.sh not found at {deploy_dir / 'scripts' / 'setup_env.sh'}"
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for error in errors:
            print(f"  - {error}")
        print()
        sys.exit(1)


def _kill_existing_session() -> None:
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session() -> None:
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )
    subprocess.run(["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"])
    subprocess.run(["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"])
    subprocess.run(["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "sim_teleop"])

    # Pane layout:
    #   0: simulator
    #   1: deploy
    #   2: pico manager
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"])
    subprocess.run(["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"])
    time.sleep(1.0)


def _send_to_pane(pane_index: int, command: str, wait: float = 1.0) -> None:
    target = f"{SESSION_NAME}:0.{pane_index}"
    subprocess.run(["tmux", "send-keys", "-t", target, command, "C-m"])
    time.sleep(wait)


def _build_deploy_args(args: argparse.Namespace) -> list[str]:
    deploy_args = [
        "./deploy.sh",
        "--input-type",
        args.deploy_input_type,
        "--cp",
        args.deploy_checkpoint,
        "--obs-config",
        args.deploy_obs_config,
    ]
    if args.deploy_zmq_host:
        deploy_args += ["--zmq-host", args.deploy_zmq_host]
    if args.deploy_planner:
        deploy_args += ["--planner", args.deploy_planner]
    if args.deploy_motion_data:
        deploy_args += ["--motion-data", args.deploy_motion_data]
    if args.deploy_output_type:
        deploy_args += ["--output-type", args.deploy_output_type]
    deploy_args.append("sim")
    return deploy_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SONIC MuJoCo sim + deploy + PICO manager in one tmux session."
    )
    parser.add_argument(
        "--teleop-venv",
        default=".venv_teleop",
        help="Virtualenv used for run_sim_loop.py and pico_manager_thread_server.py.",
    )
    parser.add_argument(
        "--deploy-input-type",
        default="zmq_manager",
        help="Input type passed to gear_sonic_deploy/deploy.sh.",
    )
    parser.add_argument(
        "--deploy-checkpoint",
        default="policy/low_latency/model",
        help="Checkpoint/model prefix passed to deploy.sh --cp.",
    )
    parser.add_argument(
        "--deploy-obs-config",
        default="policy/low_latency/observation_config.yaml",
        help="Observation config passed to deploy.sh --obs-config.",
    )
    parser.add_argument(
        "--deploy-zmq-host",
        default="",
        help="Optional --zmq-host passed to deploy.sh.",
    )
    parser.add_argument(
        "--deploy-planner",
        default="",
        help="Optional planner model path passed to deploy.sh.",
    )
    parser.add_argument(
        "--deploy-motion-data",
        default="",
        help="Optional motion data path passed to deploy.sh.",
    )
    parser.add_argument(
        "--deploy-output-type",
        default="",
        help="Optional output type passed to deploy.sh.",
    )
    parser.add_argument(
        "--auto-confirm-delay",
        type=float,
        default=6.0,
        help=(
            "Seconds to wait before sending Enter to deploy.sh confirmation. "
            "Set to 0 to confirm manually."
        ),
    )
    parser.add_argument(
        "--pico-input-source",
        default="xrt",
        choices=("xrt", "isaac-teleop"),
        help="Input source for pico_manager_thread_server.py.",
    )
    parser.add_argument(
        "--no-pico-manager",
        action="store_true",
        help="Do not pass --manager to pico_manager_thread_server.py.",
    )
    parser.add_argument(
        "--no-pico-vis-vr3pt",
        action="store_true",
        help="Do not pass --vis_vr3pt to pico_manager_thread_server.py.",
    )
    parser.add_argument(
        "--no-pico-vis-smpl",
        action="store_true",
        help="Do not pass --vis_smpl to pico_manager_thread_server.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()

    _check_prerequisites(args)
    _kill_existing_session()

    print("=" * 60)
    print("  SONIC Sim Teleop Launcher")
    print("=" * 60)
    print(f"  tmux session:      {SESSION_NAME}")
    print(f"  teleop venv:       {args.teleop_venv}")
    print(f"  deploy input:      {args.deploy_input_type}")
    print(f"  deploy checkpoint: {args.deploy_checkpoint}")
    print(f"  deploy obs config: {args.deploy_obs_config}")
    print(f"  pico input:        {args.pico_input_source}")
    print(f"  pico manager:      {'No' if args.no_pico_manager else 'Yes'}")
    print(
        "  pico vis:          "
        f"vr3pt={not args.no_pico_vis_vr3pt} smpl={not args.no_pico_vis_smpl}"
    )
    print("=" * 60)

    _create_tmux_session()

    venv_activate = shlex.quote(str(repo_root / args.teleop_venv / "bin" / "activate"))

    sim_cmd = (
        f"cd {shlex.quote(str(repo_root))} && "
        f"source {venv_activate} && "
        "python gear_sonic/scripts/run_sim_loop.py"
    )
    print("Starting MuJoCo simulator (pane 0)...")
    _send_to_pane(0, sim_cmd, wait=2.0)

    deploy_cmd = (
        f"cd {shlex.quote(str(repo_root / 'gear_sonic_deploy'))} && "
        "source scripts/setup_env.sh && "
        f"{_shell_join(_build_deploy_args(args))}"
    )
    print("Starting C++ deploy (pane 1)...")
    _send_to_pane(1, deploy_cmd, wait=1.0)
    if args.auto_confirm_delay > 0:
        time.sleep(args.auto_confirm_delay)
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:0.1", "C-m"])

    pico_args = [
        "python",
        "gear_sonic/scripts/pico_manager_thread_server.py",
        "--input-source",
        args.pico_input_source,
    ]
    if not args.no_pico_manager:
        pico_args.append("--manager")
    if not args.no_pico_vis_vr3pt:
        pico_args.append("--vis_vr3pt")
    if not args.no_pico_vis_smpl:
        pico_args.append("--vis_smpl")

    pico_cmd = (
        f"cd {shlex.quote(str(repo_root))} && "
        f"source {venv_activate} && "
        f"{_shell_join(pico_args)}"
    )
    print("Starting PICO manager (pane 2)...")
    _send_to_pane(2, pico_cmd, wait=1.0)

    subprocess.run(["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.1"])

    print()
    print("=" * 60)
    print("  All components launched.")
    print()
    print("  Panes:")
    print("    Pane 0 (top-left):     MuJoCo Simulator")
    print("    Pane 1 (right):        C++ Deploy  <-- selected")
    print("    Pane 2 (bottom-left):  PICO Manager")
    print()
    if args.auto_confirm_delay > 0:
        print("  deploy.sh confirmation Enter was sent automatically.")
    else:
        print("  deploy.sh is waiting for confirmation in pane 1.")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig: int, _frame: object) -> None:
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    main()
