import argparse
import itertools
import json
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Iterable

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
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
        self._io_lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._configured_speed_force: tuple[int, int] | None = None

    def close(self) -> None:
        with self._io_lock:
            self._disconnect_locked()

    def _disconnect_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _connect_locked(self) -> socket.socket:
        if self._sock is None:
            self._sock = socket.create_connection(
                (self.ip, self.port), timeout=self.timeout
            )
            self._sock.settimeout(self.timeout)
        return self._sock

    def _request(self, function_code: int, payload: bytes) -> bytes:
        with self._io_lock:
            last_error: Exception | None = None
            for _attempt in range(2):
                transaction_id = next(self._transaction_ids) & 0xFFFF
                pdu = struct.pack(">B", function_code) + payload
                header = struct.pack(
                    ">HHHB", transaction_id, 0, len(pdu) + 1, self.device_id
                )
                try:
                    sock = self._connect_locked()
                    sock.sendall(header + pdu)
                    response_header = self._recv_exact(sock, 7)
                    rx_transaction_id, protocol_id, length, _unit_id = struct.unpack(
                        ">HHHB", response_header
                    )
                    if rx_transaction_id != transaction_id or protocol_id != 0:
                        raise ModbusTcpError(
                            f"{self.side}: invalid Modbus response header"
                        )

                    response_pdu = self._recv_exact(sock, length - 1)
                    if not response_pdu:
                        raise ModbusTcpError(f"{self.side}: empty Modbus response")
                    if response_pdu[0] & 0x80:
                        code = response_pdu[1] if len(response_pdu) > 1 else -1
                        raise ModbusTcpError(
                            f"{self.side}: Modbus exception code {code}"
                        )
                    if response_pdu[0] != function_code:
                        raise ModbusTcpError(
                            f"{self.side}: unexpected function code {response_pdu[0]}"
                        )
                    return response_pdu[1:]
                except (OSError, ModbusTcpError) as exc:
                    last_error = exc
                    self._disconnect_locked()
            assert last_error is not None
            raise last_error

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
        speed_value = max(0, min(4000, int(speed)))
        force_value = max(0, min(12000, int(force)))

        # Serialize the one-time setup and command write against state reads.
        # The hand controller cannot reliably accept a storm of short-lived TCP
        # connections, so keep one connection and avoid rewriting unchanged
        # speed/force registers for every open/close transition.
        with self._io_lock:
            speed_force = (speed_value, force_value)
            if self._configured_speed_force != speed_force:
                self.write_register(REG_CLEAR_ERROR, 1)
                time.sleep(0.02)
                self.write_registers(
                    REG_SPEED_SET, [speed_value] * INSPIRE_HAND_DOF
                )
                time.sleep(0.02)
                self.write_registers(
                    REG_FORCE_SET, [force_value] * INSPIRE_HAND_DOF
                )
                time.sleep(0.02)
                self._configured_speed_force = speed_force
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


def _make_inspire_state_msg(
    right_q: np.ndarray,
    left_q: np.ndarray,
    right_dq: np.ndarray,
    left_dq: np.ndarray,
    right_tau_est: np.ndarray | None = None,
    left_tau_est: np.ndarray | None = None,
) -> MotorStates_:
    msg = MotorStates_([])
    right_tau = (
        np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
        if right_tau_est is None
        else right_tau_est
    )
    left_tau = (
        np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
        if left_tau_est is None
        else left_tau_est
    )
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
    active_sides: list[str],
    stop_event: threading.Event,
    publisher: ChannelPublisher,
) -> None:
    """Publish native six-motor hand state, synthesizing inactive sides as open."""
    period = 1.0 / max(float(args.state_publish_frequency), 1e-6)
    last_q: dict[str, np.ndarray] = {}
    last_time: float | None = None
    last_log_time = 0.0
    open_q = np.asarray(args.open_q, dtype=np.float64)

    while not stop_event.is_set():
        loop_start = time.monotonic()
        try:
            q = {
                side: (
                    hands[side].read_angle_normalized()
                    if side in active_sides
                    else open_q.copy()
                )
                for side in ("right", "left")
            }
            now = time.monotonic()
            dq = {}
            for side in ("right", "left"):
                if last_time is None or side not in last_q or side not in active_sides:
                    dq[side] = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
                else:
                    dq[side] = (q[side] - last_q[side]) / max(now - last_time, 1e-6)

            tau = {"right": None, "left": None}
            if args.read_force_state:
                for side in active_sides:
                    tau[side] = hands[side].read_force_normalized()

            publisher.Write(
                _make_inspire_state_msg(
                    q["right"],
                    q["left"],
                    dq["right"],
                    dq["left"],
                    tau["right"],
                    tau["left"],
                )
            )
            last_q = q
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
    last_attempt_command = None
    last_attempt_time = 0.0
    profile_samples = []
    last_profile_time = time.monotonic()
    active_sides = ["left", "right"] if args.side == "both" else [args.side]

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_command, last_attempt_command
        nonlocal last_attempt_time, last_profile_time
        callback_start = time.perf_counter()
        if len(msg.cmds) < 12:
            print(f"skip short inspire command: {len(msg.cmds)}")
            return

        right_q = tuple(float(msg.cmds[i].q) for i in range(6))
        left_q = tuple(float(msg.cmds[i + 6].q) for i in range(6))
        dds_q = {"left": left_q, "right": right_q}
        command_key = tuple(dds_q[side] for side in active_sides)
        if command_key == last_command:
            return
        now = time.monotonic()
        if command_key == last_attempt_command and now - last_attempt_time < 0.05:
            return
        last_attempt_command = command_key
        last_attempt_time = now

        try:
            angles = {
                side: normalized_to_angle(
                    profile_pose_from_dds(dds_q[side], args),
                    args.thumb_rotate_default,
                )
                for side in active_sides
            }
            side_ms = {"left": 0.0, "right": 0.0}
            successful_sides = []
            for side in active_sides:
                side_start = time.perf_counter()
                try:
                    hands[side].set_angle(
                        angles[side], speed=args.speed, force=args.force
                    )
                    successful_sides.append(side)
                except Exception as exc:
                    print(f"DDS -> Modbus {side} failed: {exc}")
                finally:
                    side_ms[side] = (time.perf_counter() - side_start) * 1000.0

            total_ms = (time.perf_counter() - callback_start) * 1000.0
            if args.profile_timing:
                profile_samples.append((side_ms["right"], side_ms["left"], total_ms))
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
            if successful_sides:
                summary = " ".join(
                    f"{side}={angles[side]}" for side in successful_sides
                )
                print(f"DDS -> Modbus {summary}")
            if len(successful_sides) == len(active_sides):
                last_command = command_key
        except Exception as exc:
            print(f"DDS -> Modbus failed: {exc}")

    ChannelFactoryInitialize(args.domain_id, args.network)
    stop_event = threading.Event()
    publisher = None
    if args.publish_state:
        publisher = ChannelPublisher("rt/inspire/state", MotorStates_)
        publisher.Init()

    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)

    if publisher is not None:
        state_thread = threading.Thread(
            target=run_state_publisher,
            args=(args, hands, active_sides, stop_event, publisher),
            daemon=True,
        )
        state_thread.start()
        print(
            "Modbus -> DDS state publisher running on rt/inspire/state "
            f"at {args.state_publish_frequency:.1f} Hz."
        )

    print(
        "DDS -> Modbus bridge running on rt/inspire/cmd "
        f"for {','.join(active_sides)}. Press Ctrl+C to stop."
    )
    print(f"Hand open q={args.open_q} angle={args.open_angle}")
    print(f"Hand grasp q={args.grasp_q} angle={args.grasp_angle}")
    print(f"DDS pose mode={args.dds_pose_mode}")
    try:
        while True:
            time.sleep(1.0)
    finally:
        stop_event.set()
        for hand in hands.values():
            hand.close()


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
        "--publish-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In DDS mode, read ANGLE_ACT over Modbus and publish rt/inspire/state.",
    )
    parser.add_argument(
        "--state-publish-frequency",
        type=float,
        default=50.0,
        help="Frequency for publishing native Inspire state in DDS mode.",
    )
    parser.add_argument(
        "--read-force-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also read FORCE_ACT and publish it as tau_est.",
    )
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
