"""Three-Enter JAR operator using the referenced SONICMJ standing controller."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading

from contract import BODY_JOINT_NAMES, TASKS
from gravity import add_gravity_arguments, gravity_options, build_arm_gravity
from initial_poses import PoseCatalog
from rpc import Client


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    add_gravity_arguments(parser)
    parser.add_argument("--sonic-root", type=Path, default=root/"runtime/sonicmj")
    parser.add_argument("--task", choices=TASKS, default="DISPENSE_TO_HELD_TUBE")
    parser.add_argument("--policy-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--start-policy", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--chunks", type=int, default=1000000)
    args, rest = parser.parse_known_args()
    if os.environ.get("SONIC_ONBOARD_BODY_HOST"):
        parser.error("This operator uses direct DDS; unset SONIC_ONBOARD_BODY_HOST")
    if args.chunks < 1 or not 1 <= args.camera_port <= 65535:
        parser.error("Invalid chunks/camera port")
    sys.path.insert(0, str(args.sonic_root))
    sys.path.insert(0, str(args.sonic_root/"external_dependencies/unitree_sdk2_python"))
    import unitree_sdk2py.core.channel as channels
    channels.ChannelConfigHasInterface = channels.ChannelConfigHasInterface.replace("/tmp/cdds.LOG", str(root/"runtime/operator-dds.log"))
    import tyro
    from decoupled_wbc.control.main.teleop import run_g1_control_loop as loop
    from operator_control import install_operator, validate_pose
    config = tyro.cli(loop.ControlLoopConfig,
                      default=loop.ControlLoopConfig(upper_body_joint_speed=.25, enable_gravity_compensation=True), args=rest)
    config.with_hands, config.hand_type, config.enable_waist = True, "dex1", False
    config.startup_t_pose = False
    config.diagnostic_read_only = not args.execute
    robot = loop.instantiate_g1_robot_model(waist_location="lower_body")
    names = [robot.joint_names[i] for i in robot.get_body_actuated_joint_indices()]
    if names != BODY_JOINT_NAMES:
        raise ValueError("Controller body order differs from the recorded JAR observation order")
    catalog = PoseCatalog(root/"bundle/deployment/initial_poses/catalog.json")
    selection_path = root/"runtime/selected_initial_pose.json"
    selection = catalog.load_selection(selection_path, default_task=args.task)
    validate_pose(robot, catalog.pose(selection))
    gravity = build_arm_gravity(robot, config.load_wbc_yaml(), gravity_options(args))
    gravity_status = gravity.summary() if gravity else {"enabled": False}
    ready, stopped = threading.Event(), threading.Event()
    child = None
    logs = root/"runtime/operator"
    logs.mkdir(exist_ok=True)

    def monitor():
        while not stopped.is_set():
            client = Client(args.policy_endpoint, timeout_ms=1000)
            try:
                status = client.status()
                (ready.set if status["task"] == args.task else ready.clear)()
            except Exception:
                ready.clear()
            finally:
                client.close()
            stopped.wait(1)

    def close():
        stopped.set()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)

    install_operator(loop, catalog=catalog, selection_path=selection_path, task=args.task,
                     sonic_root=args.sonic_root, policy_endpoint=args.policy_endpoint, log_dir=logs,
                     execute=args.execute, chunks=args.chunks, policy_ready=ready.is_set, on_close=close,
                     camera_host=args.camera_host, camera_port=args.camera_port, gravity_options=gravity_options(args))
    if args.check_config:
        loop.get_wbc_policy("g1", robot, config.load_wbc_yaml())
        report = dict(passed=True,robot_io_opened=False,selection=selection,model_task=args.task,
                      body_joint_names=names,gravity=gravity_status,reference_order="JAR: arms14 + Dex1 left/right",
                      workflow="Enter: stand / initialize arms+grippers / infer; d: damping; r: restore; p: recorded pose")
        (root/"reports/operator-config.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return
    if args.execute:
        if not sys.stdin.isatty():
            raise SystemExit("Physical operation requires an interactive terminal")
        control_lock = (root/"runtime/robot-control.lock").open("a")
        fcntl.flock(control_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.start_policy:
        if args.policy_endpoint != "tcp://127.0.0.1:5560":
            parser.error("--start-policy requires the default loopback endpoint")
        with socket.socket() as probe:
            occupied = probe.connect_ex(("127.0.0.1",5560)) == 0
        if not occupied:
            env = os.environ.copy()
            for key in ("PYTHONPATH","PYTHONHOME","LD_LIBRARY_PATH"):
                env.pop(key,None)
            with (logs/"policy.log").open("a") as log:
                child = subprocess.Popen([str(root/".venv/bin/python"),"-u",str(Path(__file__).with_name("serve.py")),
                                          "--bundle",str(root/"bundle"),"--task",args.task],
                                         env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    threading.Thread(target=monitor,daemon=True).start()
    print(json.dumps(dict(mode="execute" if args.execute else "read-only",task=args.task,gravity=gravity_status)),flush=True)
    try:
        loop.main(config)
    finally:
        close()


if __name__ == "__main__":
    main()
