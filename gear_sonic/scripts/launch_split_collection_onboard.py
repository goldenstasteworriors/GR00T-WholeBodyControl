"""Split collection: retain onboard control, record camera/REF on ThinkPad.

Default invocation prints commands only. --start enables physical control.
Reuse the installed robot launcher so task configuration and control semantics
remain those of the original robot checkout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

SESSION = "decoupled_wbc_split_onboard"


def load_original(repo: Path):
    path = repo / "gear_sonic/scripts/launch_decoupled_vla_collection.py"
    spec = importlib.util.spec_from_file_location("_split_original_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_plan(original, repo: Path, args):
    config = original.DecoupledVLACollectionLaunchConfig(
        task_prompt=args.task_prompt, dataset_name=args.dataset_name,
        hand_task=args.hand_task, hand_task_config=args.hand_task_config,
        conda_env=args.conda_env, lower_body_controller="unitree_loco",
        with_hands=True, inspire_hand_bridge=True, ref_gripper_bridge=True,
        pico_data_streamer=False, pico_navigation_range=args.pico_navigation_range,
        unitree_loco_navigation_enabled=True,
    )
    # ref_gripper_bridge preserves mixed-hand mapping. Never launch its process.
    if original._resolve_inspire_hand_side(config) != "left":
        raise RuntimeError("Original launcher no longer resolves mixed hands to left Inspire")
    prefix = original._conda_prefix(repo, config.conda_env)
    runtime = original._runtime_env_prefix(repo, config)
    native = original._aarch64_control_runtime_prefix()
    entries = [
        ("inspire_hand", original._build_inspire_hand_args(config, mode="dds"), native),
        ("control", original._build_control_args(config, "real"), native),
        ("teleop", original._build_teleop_args(config, "real"),
         original._teleop_numerical_thread_prefix()),
    ]
    return [
        {"name": name, "argv": argv,
         "command": prefix + runtime + extra + shlex.join(argv)}
        for name, argv, extra in entries
    ]


def start_plan(plan, tmux=("tmux",), session=SESSION):
    """Create all idle panes first, then dispatch commands; never kill sessions."""
    def run(*args, **kwargs):
        return subprocess.run([*tmux, *args], check=True, **kwargs)

    for name in (session, "decoupled_vla_collection"):
        probe = subprocess.run([*tmux, "has-session", "-t", "=" + name],
                               capture_output=True)
        if probe.returncode == 0:
            raise RuntimeError(f"Existing session {name}; refusing duplicate control")
    # Use explicit shells and literal send-keys to avoid user shell startup hooks.
    for index, item in enumerate(plan):
        shell = "bash --noprofile --norc"
        if index == 0:
            run("new-session", "-d", "-s", session, "-n", item["name"], shell)
            run("set-option", "-t", session, "remain-on-exit", "on")
        else:
            run("new-window", "-t", session, "-n", item["name"], shell)
    for item in plan:
        target = f"{session}:{item['name']}.0"
        run("send-keys", "-t", target, "-l", item["command"])
        run("send-keys", "-t", target, "Enter")
        time.sleep(1)
    print("Started. Attach: " + shlex.join([*tmux, "attach", "-t", session]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-repo", type=Path,
                        default=Path("/home/unitree/data_collection/GR00T-WholeBodyControl"))
    parser.add_argument("--task-prompt", default="move the beaker from left")
    parser.add_argument("--dataset-name", default="8_25_stir_2")
    parser.add_argument("--hand-task", default="grab_beaker")
    parser.add_argument("--hand-task-config", default="")
    parser.add_argument("--conda-env", default="decoupled_vla_collection")
    parser.add_argument("--pico-navigation-range", type=float, default=0.5)
    parser.add_argument("--start", action="store_true",
                        help="Actually start physical body/left-hand control")
    args = parser.parse_args()
    if not 0 < args.pico_navigation_range < float("inf"):
        parser.error("--pico-navigation-range must be finite and positive")
    repo = args.robot_repo.expanduser().resolve(strict=True)
    original = load_original(repo)
    plan = build_plan(original, repo, args)
    print(json.dumps({"mode": "START" if args.start else "PLAN ONLY",
                      "robot_repo": str(repo), "processes": plan}, indent=2))
    if not args.start:
        return
    # Read-only process check catches controllers running outside legacy tmux.
    processes = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    markers = ("run_g1_control_loop.py", "run_teleop_policy_loop.py",
               "inspire_modbus_hand.py", "ref_gripper_dds_bridge.py")
    for line in processes.splitlines():
        if any(marker in line for marker in markers):
            raise RuntimeError(f"Existing control process; refusing to start: {line.strip()}")
    for item in plan:
        entry = repo / item["argv"][1]
        if not entry.is_file():
            raise FileNotFoundError(entry)
    socket_dir = repo / ".runtime/split_collection"
    socket_dir.mkdir(parents=True, exist_ok=True)
    start_plan(plan, tmux=("tmux", "-S", str(socket_dir / "tmux.sock"), "-f", "/dev/null"))


if __name__ == "__main__":
    main()
