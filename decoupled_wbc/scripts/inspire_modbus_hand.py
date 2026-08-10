import argparse
import itertools
import json
from pathlib import Path
import socket
import struct
import time
from typing import Iterable

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_


REG_CLEAR_ERROR = 1004
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522

INSPIRE_HAND_DOF = 6
THUMB_ROTATE_INDEX = 5
DEFAULT_OPEN_Q = [1.0, 1.0, 1.0, 1.0, 1.0, 0.2]
DEFAULT_GRASP_Q = [0.15, 0.15, 0.15, 0.15, 1.0, 0.2]
DEFAULT_THUMB_ROTATE = None
DEFAULT_HAND_POSE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic"
    / "config"
    / "data_collection"
    / "inspire_hand_pose.json"
)
FULL_OPEN = [1000, 1000, 1000, 1000, 1000, 200]
FULL_GRASP = [150, 150, 150, 150, 1000, 200]


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


def validate_normalized_pose(values: Iterable[float], name: str) -> list[float]:
    q = np.asarray(list(values), dtype=np.float64)
    if q.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"{name} must have {INSPIRE_HAND_DOF} values, got shape {q.shape}")
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError(f"{name} values must be in [0.0, 1.0], got {q.tolist()}")
    return q.tolist()


def normalized_to_angle(
    values: Iterable[float],
    thumb_rotate_default: float | None = DEFAULT_THUMB_ROTATE,
) -> list[int]:
    q = np.asarray(validate_normalized_pose(values, "normalized hand pose"), dtype=np.float64)
    q = np.clip(q, 0.0, 1.0)
    if thumb_rotate_default is not None:
        q[THUMB_ROTATE_INDEX] = np.clip(float(thumb_rotate_default), 0.0, 1.0)
    return [int(round(v * 1000.0)) for v in q]


def load_pose_config(path: str) -> dict[str, list[float]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    open_q = data.get("open")
    grasp_q = data.get("grasp", data.get("closed"))
    if open_q is None or grasp_q is None:
        raise ValueError(f"{path} must define 'open' and 'grasp' (or 'closed')")
    return {
        "open": validate_normalized_pose(open_q, "open"),
        "grasp": validate_normalized_pose(grasp_q, "grasp"),
    }


def resolve_hand_profiles(args) -> None:
    profiles = {
        "open": DEFAULT_OPEN_Q.copy(),
        "grasp": DEFAULT_GRASP_Q.copy(),
    }
    if args.hand_pose_config:
        profiles.update(load_pose_config(args.hand_pose_config))
    if args.open_q is not None:
        profiles["open"] = validate_normalized_pose(args.open_q, "--open-q")
    if args.grasp_q is not None:
        profiles["grasp"] = validate_normalized_pose(args.grasp_q, "--grasp-q")

    args.open_q = profiles["open"]
    args.grasp_q = profiles["grasp"]
    args.open_angle = normalized_to_angle(args.open_q)
    args.grasp_angle = normalized_to_angle(args.grasp_q)


def profile_pose_from_dds(q: Iterable[float], args) -> list[float]:
    q_arr = np.asarray(list(q), dtype=np.float64)
    if q_arr.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"DDS hand pose must have {INSPIRE_HAND_DOF} values, got {q_arr.shape}")
    if args.dds_pose_mode == "passthrough":
        return validate_normalized_pose(q_arr, "DDS hand pose")

    finger_open_mean = float(np.mean(q_arr[:4]))
    return args.grasp_q if finger_open_mean < args.dds_profile_threshold else args.open_q


def send_to_target(hands: dict[str, InspireModbusHand], target: str, values: list[int], speed: int, force: int) -> None:
    sides = ["left", "right"] if target == "both" else [target]
    for side in sides:
        hands[side].set_angle(values, speed=speed, force=force)
        print(f"sent {side}: {values}")


def run_command(args, hands: dict[str, InspireModbusHand]) -> None:
    if args.command == "toggle":
        for i in range(args.count):
            values = args.grasp_angle if i % 2 == 0 else args.open_angle
            label = "grasp" if i % 2 == 0 else "open"
            print(f"sending {label}: {values}")
            send_to_target(hands, args.side, values, args.speed, args.force)
            time.sleep(args.period)
        return

    values = args.grasp_angle if args.command == "grasp" else args.open_angle
    print(f"sending {args.command}: {values}")
    send_to_target(hands, args.side, values, args.speed, args.force)


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
            right_pose = profile_pose_from_dds(right_q, args)
            left_pose = profile_pose_from_dds(left_q, args)
            right_angle = normalized_to_angle(right_pose, args.thumb_rotate_default)
            left_angle = normalized_to_angle(left_pose, args.thumb_rotate_default)
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
    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)
    print("DDS -> Modbus bridge running on rt/inspire/cmd. Press Ctrl+C to stop.")
    print(f"Hand open q={args.open_q} angle={args.open_angle}")
    print(f"Hand grasp q={args.grasp_q} angle={args.grasp_angle}")
    print(f"DDS pose mode={args.dds_pose_mode}")
    while True:
        time.sleep(1.0)


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
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--profile-timing", action="store_true", help="Print DDS to Modbus timing.")
    parser.add_argument("--profile-interval", type=float, default=1.0)
    parser.add_argument(
        "--thumb-rotate-default",
        type=float,
        default=DEFAULT_THUMB_ROTATE,
        help=(
            "Deprecated override for DDS passthrough thumb rotation. "
            "Prefer setting the sixth value in --open-q/--grasp-q."
        ),
    )
    parser.add_argument(
        "--open-q",
        nargs=INSPIRE_HAND_DOF,
        type=float,
        default=None,
        metavar=("LITTLE", "RING", "MIDDLE", "INDEX", "THUMB_BEND", "THUMB_ROTATE"),
        help="Normalized open pose, 6 values in [0, 1]. Default: 1 1 1 1 1 0.2.",
    )
    parser.add_argument(
        "--grasp-q",
        "--closed-q",
        nargs=INSPIRE_HAND_DOF,
        type=float,
        default=None,
        metavar=("LITTLE", "RING", "MIDDLE", "INDEX", "THUMB_BEND", "THUMB_ROTATE"),
        help="Normalized grasp/closed pose, 6 values in [0, 1]. Default: 0.15 0.15 0.15 0.15 1 0.2.",
    )
    parser.add_argument(
        "--hand-pose-config",
        default=str(DEFAULT_HAND_POSE_CONFIG),
        help=(
            "Optional JSON with {'open': [6 values], 'grasp': [6 values]}. "
            "CLI --open-q/--grasp-q override this file."
        ),
    )
    parser.add_argument(
        "--dds-pose-mode",
        choices=["profile", "passthrough"],
        default="profile",
        help=(
            "In DDS mode, 'profile' maps upstream open/grasp commands to --open-q/--grasp-q; "
            "'passthrough' forwards the 6 DDS q values directly."
        ),
    )
    parser.add_argument(
        "--dds-profile-threshold",
        type=float,
        default=0.5,
        help="Finger mean threshold used by --dds-pose-mode profile to choose grasp vs open.",
    )
    args = parser.parse_args()
    resolve_hand_profiles(args)
    return args


def main():
    args = parse_args()
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
