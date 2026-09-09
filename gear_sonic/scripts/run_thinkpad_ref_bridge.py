#!/usr/bin/env python3
"""PICO ROS 2/DDS right trigger -> existing ThinkPad REF USB driver.

Only the REF USB device is actuated. No robot command publishers are created.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRIVER = Path("/home/data-collection/projects/inspect-robots/.deployment/ref-control")
DEFAULT_CONFIG = ROOT / "gear_sonic/config/data_collection/thinkpad_ref_gripper_tasks.json"
DEFAULT_TOPIC = "ControlPolicy/upper_body_pose"


class TriggerGate:
    """Require a fresh released trigger initially and after an input interruption."""

    def __init__(self, opened, pressed, max_age):
        self.opened, self.pressed, self.max_age = opened, pressed, max_age
        self.armed = False

    def target(self, value, age, publishers):
        if (publishers != 1 or age < 0 or age > self.max_age
                or not math.isfinite(value) or not 0 <= value <= 1):
            self.armed = False
            return None, "missing_stale_invalid_or_ambiguous_input"
        if not self.armed:
            if value > 0.5:
                return None, "waiting_for_trigger_release"
            self.armed = True
        return (self.pressed if value > 0.5 else self.opened), "ready"


def load_positions(path, task):
    data = json.loads(Path(path).read_text(encoding="utf-8"))[task]["right_gripper"]
    positions = float(data["open"]), float(data["pressed"])
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in positions):
        raise ValueError("REF positions must be finite and within [0, 1]")
    return positions


def load_driver(directory):
    directory = Path(directory).resolve()
    for name in ("agibot_ref.py", "ref_control.py"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    sys.path.insert(0, str(directory))
    sdk = importlib.import_module("agibot_ref")
    control = importlib.import_module("ref_control")
    if Path(sdk.__file__).resolve().parent != directory:
        raise RuntimeError("An unexpected REF driver module was already loaded")
    return sdk, control


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand-task", default="grab_beaker")
    p.add_argument("--hand-task-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--ref-control-dir", type=Path, default=DEFAULT_DRIVER)
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--max-input-age", type=float, default=0.3)
    p.add_argument("--duration", type=float, default=0,
                   help="Exit after this many seconds; 0 runs until Ctrl+C.")
    p.add_argument("--log-file", type=Path,
                   help="New JSONL file; default is project .runtime/ref_bridge/<uuid>.jsonl")
    p.add_argument("--preflight", action="store_true",
                   help="Check imports/configuration only, without ROS or USB initialization.")
    p.add_argument("--dry-run", action="store_true",
                   help="Receive real ROS messages but send USB packets to DryRunTransport.")
    args = p.parse_args()
    if not math.isfinite(args.max_input_age) or args.max_input_age <= 0:
        p.error("--max-input-age must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        p.error("--duration must be finite and nonnegative")
    return args


def run(args):
    import msgpack
    import msgpack_numpy as mnp
    import rclpy
    from rclpy.utilities import get_rmw_implementation_identifier
    from std_msgs.msg import ByteMultiArray

    opened, pressed = load_positions(args.hand_task_config, args.hand_task)
    sdk, control = load_driver(args.ref_control_dir)
    import usb.core
    import usb.backend.libusb1
    if usb.backend.libusb1.get_backend() is None:
        raise RuntimeError("libusb backend is unavailable")
    if args.preflight:
        print(json.dumps({
            "preflight": "ok", "driver": sdk.__file__,
            "middleware": get_rmw_implementation_identifier(), "topic": args.topic,
            "open": opened, "pressed": pressed, "hardware_opened": False,
        }))
        return

    session_id = str(uuid.uuid4())
    path = args.log_file or ROOT / ".runtime/ref_bridge" / (session_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = TriggerGate(opened, pressed, args.max_input_age)
    latest = None
    lock = threading.Lock()
    client = None
    node = None
    spin_thread = None
    sequence = 0
    last_position = None
    last_status = None
    source_gid = None
    started = time.monotonic()
    from rclpy.executors import SingleThreadedExecutor
    executor = None

    with path.open("x", encoding="utf-8", buffering=1) as log:
        def emit(event, **data):
            log.write(json.dumps(dict(event=event, session_id=session_id,
                                      monotonic_ns=time.monotonic_ns(), **data)) + "\n")

        def callback(message):
            nonlocal latest, sequence
            received_ns = time.monotonic_ns()
            try:
                raw = message.data
                packed = bytes(raw) if len(raw) == 0 or isinstance(raw[0], int) else b"".join(raw)
                data = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
                value = float(data["right_trigger"])
            except (ValueError, TypeError, KeyError, OverflowError, msgpack.UnpackException):
                value = float("nan")
            with lock:
                sequence += 1
                latest = (value, received_ns, sequence)

        emit("session_start", wall_time_ns=time.time_ns(), dry_run=args.dry_run,
             task=args.hand_task, open=opened, pressed=pressed,
             topic=args.topic, feedback_kind="last_ACKed_command_not_measured_position",
             driver=str(args.ref_control_dir), ros_domain=os.environ.get("ROS_DOMAIN_ID", "0"))
        print(f"REF bridge log: {path}", flush=True)
        print("Waiting for exactly one publisher and a fresh released right trigger.", flush=True)
        try:
            rclpy.init(args=[])
            node = rclpy.create_node("thinkpad_ref_gripper_" + session_id[:8])
            node.create_subscription(ByteMultiArray, args.topic, callback, 1)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            spin_errors = []
            def spin():
                try:
                    executor.spin()
                except Exception as exc:
                    spin_errors.append(exc)
            spin_thread = threading.Thread(target=spin, daemon=True)
            spin_thread.start()
            while rclpy.ok() and (not args.duration or time.monotonic() - started < args.duration):
                with lock:
                    sample = latest
                if spin_errors:
                    raise RuntimeError(f"ROS executor stopped: {spin_errors[0]}")
                publishers = node.get_publishers_info_by_topic(args.topic)
                count = len(publishers)
                if sample is None:
                    target, status = gate.target(float("nan"), 0, count)
                else:
                    value, received_ns, seq = sample
                    gid = bytes(publishers[0].endpoint_gid).hex() if count == 1 else None
                    if source_gid is not None and gid is not None and gid != source_gid:
                        raise RuntimeError("PICO publisher changed; restart bridge for the new source")
                    target, status = gate.target(
                        value, (time.monotonic_ns() - received_ns) / 1e9, count)
                if status != last_status:
                    emit("input_status", status=status, publisher_count=count)
                    print(status, flush=True)
                    last_status = status
                if target is None:
                    time.sleep(0.01)
                    continue
                if client is None:
                    source_gid = gid
                    if args.dry_run:
                        transport = sdk.DryRunTransport()
                        client = sdk.RefClient(transport, interface_crc=sdk.CAPTURED_INTERFACE_CRC)
                        client.initialize(check_schema=False)
                        emit("dry_run_initialized", schema_and_prepare_skipped=True)
                    else:
                        transport = sdk.PyUsbTransport()
                        client = sdk.RefClient(transport, interface_crc=sdk.CAPTURED_INTERFACE_CRC)
                        client.initialize(check_schema=True,
                                          expected_serial=control.DEFAULT_FIRMWARE_SERIAL)
                        if client.read_error_code():
                            raise RuntimeError("REF reported a fault before prepare")
                        if not all(client.prepare()):
                            raise RuntimeError("REF prepare failed")
                        emit("usb_initialized")
                    # Initialization may take time; reacquire a fresh release before motion.
                    gate.armed = False
                    continue
                if target != last_position:
                    if client.read_error_code():
                        raise RuntimeError("REF reported a motor fault")
                    # USB fault queries can block. Never execute an aged command.
                    if (time.monotonic_ns() - received_ns) / 1e9 > args.max_input_age:
                        gate.armed = False
                        continue
                    emit("command_requested", trigger=value, trigger_receive_monotonic_ns=received_ns,
                         trigger_sequence=seq, requested_position=target)
                    client.set_pos(target)
                    last_position = target
                    emit("command_acked", trigger=value, trigger_receive_monotonic_ns=received_ns,
                         trigger_sequence=seq, commanded_position=target, measured_position=None,
                         dry_run=args.dry_run)
                    print(f"{'DRY RUN ' if args.dry_run else ''}REF target ACK: {target}", flush=True)
                time.sleep(0.01)
        except KeyboardInterrupt:
            emit("interrupted")
        except BaseException as exc:
            emit("error", error=str(exc))
            raise
        finally:
            if executor is not None:
                executor.shutdown(timeout_sec=2)
            if spin_thread is not None:
                spin_thread.join(timeout=2)
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            if client is not None:
                client.close()
            emit("session_end", last_acked_command=last_position,
                 close_behavior="release_USB_only_no_motion")


if __name__ == "__main__":
    run(parse_args())
