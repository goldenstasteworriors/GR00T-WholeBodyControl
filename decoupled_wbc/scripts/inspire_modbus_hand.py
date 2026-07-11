import argparse
import itertools
import os
import socket
import struct
import threading
import time
from typing import Iterable

import numpy as np
from gear_sonic.utils.data_collection.inspire_hand_tasks import (
    DEFAULT_HAND_TASK,
    HAND_TASK_CONFIG_ENV,
    available_hand_tasks,
    normalized_pose_to_modbus_angles,
    resolve_hand_task_pose,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


REG_CLEAR_ERROR = 1004
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582

INSPIRE_HAND_DOF = 6
THUMB_ROTATE_INDEX = 5
DEFAULT_THUMB_ROTATE = 0.5


class ModbusTcpError(RuntimeError):
    pass


class InspireModbusHand:
    def __init__(self, side: str, ip: str, port: int = 6000, device_id: int = 1, timeout: float = 1.0):
        self.side = side
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.timeout = timeout
        self._transaction_ids = itertools.count(1)

    def _request(self, function_code: int, payload: bytes) -> bytes:
        transaction_id = next(self._transaction_ids) & 0xFFFF
        pdu = struct.pack(">B", function_code) + payload
        header = struct.pack(">HHHB", transaction_id, 0, len(pdu) + 1, self.device_id)

        with socket.create_connection((self.ip, self.port), timeout=self.timeout) as sock:
            sock.sendall(header + pdu)
            response_header = self._recv_exact(sock, 7)
            rx_transaction_id, protocol_id, length, _unit_id = struct.unpack(">HHHB", response_header)
            if rx_transaction_id != transaction_id or protocol_id != 0:
                raise ModbusTcpError(f"{self.side}: invalid Modbus response header")

            response_pdu = self._recv_exact(sock, length - 1)
            if not response_pdu:
                raise ModbusTcpError(f"{self.side}: empty Modbus response")
            if response_pdu[0] & 0x80:
                code = response_pdu[1] if len(response_pdu) > 1 else -1
                raise ModbusTcpError(f"{self.side}: Modbus exception code {code}")
            if response_pdu[0] != function_code:
                raise ModbusTcpError(f"{self.side}: unexpected function code {response_pdu[0]}")
            return response_pdu[1:]

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ModbusTcpError("connection closed while reading response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def write_register(self, address: int, value: int) -> None:
        payload = struct.pack(">HH", address, int(value) & 0xFFFF)
        self._request(0x06, payload)

    def write_registers(self, address: int, values: Iterable[int]) -> None:
        registers = [int(v) & 0xFFFF for v in values]
        payload = struct.pack(">HHB", address, len(registers), len(registers) * 2)
        payload += struct.pack(">" + "H" * len(registers), *registers)
        self._request(0x10, payload)

    def read_registers(self, address: int, count: int) -> list[int]:
        if count <= 0:
            raise ValueError(f"{self.side}: register count must be positive, got {count}")
        payload = struct.pack(">HH", address, count)
        response = self._request(0x03, payload)
        expected_bytes = count * 2
        if not response or response[0] != expected_bytes:
            byte_count = response[0] if response else 0
            raise ModbusTcpError(
                f"{self.side}: expected {expected_bytes} data bytes, got {byte_count}"
            )
        data = response[1 : 1 + expected_bytes]
        if len(data) != expected_bytes:
            raise ModbusTcpError(f"{self.side}: short register response")
        return list(struct.unpack(">" + "H" * count, data))

    def read_angle_normalized(self) -> np.ndarray:
        values = self.read_registers(REG_ANGLE_ACT, INSPIRE_HAND_DOF)
        return np.clip(np.asarray(values, dtype=np.float64) / 1000.0, 0.0, 1.0)

    def read_force_normalized(self) -> np.ndarray:
        values = self.read_registers(REG_FORCE_ACT, INSPIRE_HAND_DOF)
        return np.clip(np.asarray(values, dtype=np.float64) / 1000.0, 0.0, None)

    def set_angle(self, values: Iterable[int], speed: int = 3000, force: int = 12000) -> None:
        angle_values = [max(0, min(1000, int(v))) for v in values]
        if len(angle_values) != INSPIRE_HAND_DOF:
            raise ValueError(f"{self.side}: expected 6 angle values, got {len(angle_values)}")

        self.write_register(REG_CLEAR_ERROR, 1)
        time.sleep(0.02)
        self.write_registers(REG_SPEED_SET, [max(0, min(4000, int(speed)))] * INSPIRE_HAND_DOF)
        time.sleep(0.02)
        self.write_registers(REG_FORCE_SET, [max(0, min(12000, int(force)))] * INSPIRE_HAND_DOF)
        time.sleep(0.02)
        self.write_registers(REG_ANGLE_SET, angle_values)


def normalized_to_angle(values: Iterable[float], thumb_rotate_default: float = DEFAULT_THUMB_ROTATE) -> list[int]:
    q = np.asarray(list(values), dtype=np.float64)
    if q.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"expected 6 normalized values, got shape {q.shape}")
    q = np.clip(q, 0.0, 1.0)
    q[THUMB_ROTATE_INDEX] = np.clip(float(thumb_rotate_default), 0.0, 1.0)
    return [int(round(v * 1000.0)) for v in q]


def normalized_to_task_angle(
    values: Iterable[float],
    hand_task: str,
    thumb_rotate_default: float = DEFAULT_THUMB_ROTATE,
) -> list[int]:
    q = np.asarray(list(values), dtype=np.float64)
    if q.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"expected 6 normalized values, got shape {q.shape}")
    return normalized_pose_to_modbus_angles(q.tolist())


def task_command_angles(hand_task: str, command: str) -> list[int]:
    pose = resolve_hand_task_pose(hand_task, pressed=(command == "grasp"))
    return normalized_pose_to_modbus_angles(pose)


def send_to_target(hands: dict[str, InspireModbusHand], target: str, values: list[int], speed: int, force: int) -> None:
    sides = ["left", "right"] if target == "both" else [target]
    for side in sides:
        hands[side].set_angle(values, speed=speed, force=force)
        print(f"sent {side}: {values}")


def run_command(args, hands: dict[str, InspireModbusHand]) -> None:
    if args.command == "toggle":
        for i in range(args.count):
            command = "grasp" if i % 2 == 0 else "open"
            values = task_command_angles(args.hand_task, command)
            label = "grasp" if i % 2 == 0 else "open"
            print(f"sending {label}: {values}")
            send_to_target(hands, args.side, values, args.speed, args.force)
            time.sleep(args.period)
        return

    values = task_command_angles(args.hand_task, "grasp" if args.command == "grasp" else "open")
    print(f"sending {args.command}: {values}")
    send_to_target(hands, args.side, values, args.speed, args.force)


def _make_inspire_state_msg(
    right_q: np.ndarray,
    left_q: np.ndarray,
    right_dq: np.ndarray,
    left_dq: np.ndarray,
    right_tau_est: np.ndarray | None = None,
    left_tau_est: np.ndarray | None = None,
) -> MotorStates_:
    msg = MotorStates_([])
    right_tau = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64) if right_tau_est is None else right_tau_est
    left_tau = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64) if left_tau_est is None else left_tau_est
    for q, dq, tau_est in zip(right_q, right_dq, right_tau):
        state = unitree_go_msg_dds__MotorState_()
        state.q = float(q)
        state.dq = float(dq)
        state.tau_est = float(tau_est)
        msg.states.append(state)
    for q, dq, tau_est in zip(left_q, left_dq, left_tau):
        state = unitree_go_msg_dds__MotorState_()
        state.q = float(q)
        state.dq = float(dq)
        state.tau_est = float(tau_est)
        msg.states.append(state)
    return msg


def run_state_publisher(
    args,
    hands: dict[str, InspireModbusHand],
    stop_event: threading.Event,
) -> None:
    publisher = ChannelPublisher("rt/inspire/state", MotorStates_)
    publisher.Init()

    period = 1.0 / max(float(args.state_publish_frequency), 1e-6)
    last_right_q: np.ndarray | None = None
    last_left_q: np.ndarray | None = None
    last_time: float | None = None
    last_log_time = 0.0

    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            right_q = hands["right"].read_angle_normalized()
            left_q = hands["left"].read_angle_normalized()
            now = time.monotonic()
            if last_time is None or last_right_q is None or last_left_q is None:
                right_dq = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
                left_dq = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
            else:
                dt = max(now - last_time, 1e-6)
                right_dq = (right_q - last_right_q) / dt
                left_dq = (left_q - last_left_q) / dt

            right_tau = left_tau = None
            if args.read_force_state:
                right_tau = hands["right"].read_force_normalized()
                left_tau = hands["left"].read_force_normalized()

            publisher.Write(_make_inspire_state_msg(right_q, left_q, right_dq, left_dq, right_tau, left_tau))
            last_right_q = right_q
            last_left_q = left_q
            last_time = now
        except Exception as exc:
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                print(f"Inspire state publish failed: {exc}")
                last_log_time = now

        elapsed = time.monotonic() - loop_start
        stop_event.wait(max(0.0, period - elapsed))


def run_dds_bridge(args, hands: dict[str, InspireModbusHand]) -> None:
    last_command = None
    profile_samples = []
    last_profile_time = time.monotonic()

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_command, last_profile_time
        callback_start = time.perf_counter()
        if len(msg.cmds) < 12:
            print(f"skip short inspire command: {len(msg.cmds)}")
            return

        right_q = tuple(float(msg.cmds[i].q) for i in range(6))
        left_q = tuple(float(msg.cmds[i + 6].q) for i in range(6))
        command_key = (right_q, left_q)
        if command_key == last_command:
            return
        last_command = command_key

        try:
            right_angle = normalized_to_task_angle(
                right_q, args.hand_task, args.thumb_rotate_default
            )
            left_angle = normalized_to_task_angle(
                left_q, args.hand_task, args.thumb_rotate_default
            )
            right_start = time.perf_counter()
            hands["right"].set_angle(right_angle, speed=args.speed, force=args.force)
            right_ms = (time.perf_counter() - right_start) * 1000.0
            left_start = time.perf_counter()
            hands["left"].set_angle(left_angle, speed=args.speed, force=args.force)
            left_ms = (time.perf_counter() - left_start) * 1000.0
            total_ms = (time.perf_counter() - callback_start) * 1000.0
            if args.profile_timing:
                profile_samples.append((right_ms, left_ms, total_ms))
                now = time.monotonic()
                if now - last_profile_time >= args.profile_interval:
                    arr = np.asarray(profile_samples, dtype=np.float64)
                    print(
                        "[InspireHandProfile] "
                        f"n={len(profile_samples)} "
                        f"right_modbus={arr[:, 0].mean():.2f}ms "
                        f"left_modbus={arr[:, 1].mean():.2f}ms "
                        f"callback_total={arr[:, 2].mean():.2f}ms "
                        f"callback_max={arr[:, 2].max():.2f}ms"
                    )
                    profile_samples.clear()
                    last_profile_time = now
            print(f"DDS -> Modbus right={right_angle} left={left_angle}")
        except Exception as exc:
            print(f"DDS -> Modbus failed: {exc}")

    ChannelFactoryInitialize(args.domain_id, args.network)
    stop_event = threading.Event()
    if args.publish_state:
        state_thread = threading.Thread(
            target=run_state_publisher,
            args=(args, hands, stop_event),
            daemon=True,
        )
        state_thread.start()
        print(
            "Modbus -> DDS state publisher running on rt/inspire/state "
            f"at {args.state_publish_frequency:.1f} Hz."
        )

    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)
    print("DDS -> Modbus bridge running on rt/inspire/cmd. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    finally:
        stop_event.set()


def parse_args():
    parser = argparse.ArgumentParser(description="Control Inspire hands over Modbus TCP or bridge DDS commands.")
    parser.add_argument("--left-ip", default="192.168.123.210")
    parser.add_argument("--right-ip", default="192.168.123.211")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--speed", type=int, default=3000)
    parser.add_argument("--force", type=int, default=12000)
    parser.add_argument("--mode", choices=["command", "dds"], default="command")
    parser.add_argument("--network", default="enp7s0", help="DDS network interface for --mode dds.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--command", choices=["open", "grasp", "toggle"], default="toggle")
    parser.add_argument(
        "--hand-task",
        default=DEFAULT_HAND_TASK,
        help="Task-specific Inspire hand mapping name from inspire_hand_tasks.json.",
    )
    parser.add_argument(
        "--hand-task-config",
        default="",
        help="Optional path to inspire_hand_tasks.json. Defaults to the project config.",
    )
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--profile-timing", action="store_true", help="Print DDS to Modbus timing.")
    parser.add_argument("--profile-interval", type=float, default=1.0)
    parser.add_argument(
        "--publish-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In DDS mode, read ANGLE_ACT over Modbus and publish rt/inspire/state.",
    )
    parser.add_argument(
        "--state-publish-frequency",
        type=float,
        default=50.0,
        help="Frequency for publishing rt/inspire/state in DDS mode.",
    )
    parser.add_argument(
        "--read-force-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also read FORCE_ACT and publish it as tau_est. This adds two Modbus reads per cycle.",
    )
    parser.add_argument(
        "--thumb-rotate-default",
        type=float,
        default=DEFAULT_THUMB_ROTATE,
        help="Default normalized thumb rotation in DDS mode, 0.0 closed to 1.0 open.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hand_task_config:
        os.environ[HAND_TASK_CONFIG_ENV] = args.hand_task_config
    known_tasks = available_hand_tasks()
    if args.hand_task not in known_tasks:
        raise ValueError(f"Unknown --hand-task {args.hand_task!r}. Known tasks: {', '.join(known_tasks)}")
    hands = {
        "left": InspireModbusHand("left", args.left_ip, args.hand_port, args.device_id),
        "right": InspireModbusHand("right", args.right_ip, args.hand_port, args.device_id),
    }
    if args.mode == "dds":
        run_dds_bridge(args, hands)
    else:
        run_command(args, hands)


if __name__ == "__main__":
    main()
