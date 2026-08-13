import argparse
from collections import deque
import itertools
import json
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Iterable

import numpy as np
import zmq

from gear_sonic.utils.data_collection.inspire_tactile import (
    MODBUS_METRIC_NAMES,
    TACTILE_BATCHES,
    TACTILE_BATCH_COUNT_WITH_FORCE,
    TACTILE_FORCE_COUNT,
    TACTILE_PROTOCOL_VERSION,
    TACTILE_REGION_COUNT,
    TACTILE_REGION_INDEX_BY_NAME,
    TACTILE_REGIONS,
    flatten_regions,
    pack_snapshot,
    unpack_batch,
)
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


class ModbusProfiler:
    """Rolling Modbus utilization and deadline statistics."""

    METRIC_NAMES = MODBUS_METRIC_NAMES

    def __init__(self, target_full_refresh_hz: float, window_s: float = 30.0):
        self.target_full_refresh_hz = float(target_full_refresh_hz)
        self.window_s = float(window_s)
        self._lock = threading.Lock()
        self._events = deque()
        self._state_cycles = deque()
        self._full_refresh_times = deque()

    def record_request(
        self,
        kind: str,
        *,
        wait_ms: float,
        io_ms: float,
        success: bool,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, kind, wait_ms, io_ms, success))
            self._trim(now)

    def record_state_cycle(self, deadline_missed: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._state_cycles.append((now, bool(deadline_missed)))
            self._trim(now)

    def record_full_refresh(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._full_refresh_times.append(now)
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        for samples in (self._events, self._state_cycles):
            while samples and samples[0][0] < cutoff:
                samples.popleft()
        while self._full_refresh_times and self._full_refresh_times[0] < cutoff:
            self._full_refresh_times.popleft()

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _p95(values: list[float]) -> float:
        return float(np.percentile(values, 95)) if values else 0.0

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            events = list(self._events)
            state_cycles = list(self._state_cycles)
            refresh_times = list(self._full_refresh_times)

        if events:
            # Avoid misleadingly large rates in the first few frames before the
            # rolling window has accumulated a meaningful observation span.
            span_s = max(1.0, min(self.window_s, now - events[0][0]))
        else:
            span_s = self.window_s
        tactile = [event for event in events if event[1] == "tactile"]
        state = [event for event in events if event[1] == "state"]
        tactile_ok = [event for event in tactile if event[4]]
        tactile_errors = len(tactile) - len(tactile_ok)
        tactile_io = [event[3] for event in tactile_ok]
        state_io = [event[3] for event in state if event[4]]
        lock_wait = [event[2] for event in events]
        busy_ratio = min(1.0, sum(event[3] for event in events) / (span_s * 1000.0))
        non_tactile_busy = sum(event[3] for event in events if event[1] != "tactile") / (
            span_s * 1000.0
        )
        tactile_p95_ms = self._p95(tactile_io)
        safe_budget_ratio = max(0.0, 0.60 - non_tactile_busy)
        estimated_max_hz = (
            safe_budget_ratio * 1000.0 / (tactile_p95_ms * TACTILE_BATCH_COUNT_WITH_FORCE)
            if tactile_p95_ms > 0.0
            else 0.0
        )
        state_miss_ratio = (
            sum(missed for _, missed in state_cycles) / len(state_cycles)
            if state_cycles
            else 0.0
        )
        if state_miss_ratio > 0.01:
            estimated_max_hz = min(estimated_max_hz, self.target_full_refresh_hz)
        state_cycle_hz = (
            (len(state_cycles) - 1)
            / max(1e-6, state_cycles[-1][0] - state_cycles[0][0])
            if len(state_cycles) >= 2
            else 0.0
        )
        actual_full_hz = (
            (len(refresh_times) - 1) / max(1e-6, refresh_times[-1] - refresh_times[0])
            if len(refresh_times) >= 2
            else 0.0
        )
        values = np.array(
            [
                self.target_full_refresh_hz,
                actual_full_hz,
                len(tactile_ok) / span_s,
                tactile_errors / span_s,
                self._mean(tactile_io),
                tactile_p95_ms,
                state_cycle_hz,
                self._p95(state_io),
                self._p95(lock_wait),
                busy_ratio,
                state_miss_ratio,
                estimated_max_hz,
            ],
            dtype=np.float32,
        )
        return {
            "time_s": time.time(),
            "window_s": span_s,
            "metric_names": self.METRIC_NAMES,
            "values": values,
        }


class InspireModbusHand:
    def __init__(
        self,
        side: str,
        ip: str,
        port: int = 6000,
        device_id: int = 1,
        timeout: float = 1.0,
        profiler: ModbusProfiler | None = None,
    ):
        self.side = side
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.timeout = timeout
        self._transaction_ids = itertools.count(1)
        self._io_lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._configured_speed_force: tuple[int, int] | None = None
        self.profiler = profiler

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
            self._sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)
            self._sock.settimeout(self.timeout)
        return self._sock

    def _request(self, function_code: int, payload: bytes, *, kind: str) -> bytes:
        wait_started = time.perf_counter()
        with self._io_lock:
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            io_started = time.perf_counter()
            success = False
            try:
                last_error: Exception | None = None
                for _attempt in range(2):
                    transaction_id = next(self._transaction_ids) & 0xFFFF
                    pdu = struct.pack(">B", function_code) + payload
                    header = struct.pack(">HHHB", transaction_id, 0, len(pdu) + 1, self.device_id)
                    try:
                        sock = self._connect_locked()
                        sock.sendall(header + pdu)
                        response_header = self._recv_exact(sock, 7)
                        rx_transaction_id, protocol_id, length, _unit_id = struct.unpack(
                            ">HHHB", response_header
                        )
                        if rx_transaction_id != transaction_id or protocol_id != 0:
                            raise ModbusTcpError(f"{self.side}: invalid Modbus response header")
                        response_pdu = self._recv_exact(sock, length - 1)
                        if not response_pdu:
                            raise ModbusTcpError(f"{self.side}: empty Modbus response")
                        if response_pdu[0] & 0x80:
                            code = response_pdu[1] if len(response_pdu) > 1 else -1
                            raise ModbusTcpError(f"{self.side}: Modbus exception code {code}")
                        if response_pdu[0] != function_code:
                            raise ModbusTcpError(
                                f"{self.side}: unexpected function code {response_pdu[0]}"
                            )
                        success = True
                        return response_pdu[1:]
                    except (OSError, ModbusTcpError) as exc:
                        last_error = exc
                        self._disconnect_locked()
                assert last_error is not None
                raise last_error
            finally:
                if self.profiler is not None:
                    self.profiler.record_request(
                        kind,
                        wait_ms=wait_ms,
                        io_ms=(time.perf_counter() - io_started) * 1000.0,
                        success=success,
                    )

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

    def write_register(self, address: int, value: int, *, kind: str = "command") -> None:
        payload = struct.pack(">HH", address, int(value) & 0xFFFF)
        self._request(0x06, payload, kind=kind)

    def write_registers(
        self, address: int, values: Iterable[int], *, kind: str = "command"
    ) -> None:
        registers = [int(v) & 0xFFFF for v in values]
        payload = struct.pack(">HHB", address, len(registers), len(registers) * 2)
        payload += struct.pack(">" + "H" * len(registers), *registers)
        self._request(0x10, payload, kind=kind)

    def read_registers(self, address: int, count: int, *, kind: str = "state") -> list[int]:
        if count <= 0:
            raise ValueError(f"{self.side}: register count must be positive, got {count}")
        payload = struct.pack(">HH", address, count)
        response = self._request(0x03, payload, kind=kind)
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

        speed_force = (
            max(0, min(4000, int(speed))),
            max(0, min(12000, int(force))),
        )
        with self._io_lock:
            if self._configured_speed_force != speed_force:
                self.write_register(REG_CLEAR_ERROR, 1)
                time.sleep(0.02)
                self.write_registers(REG_SPEED_SET, [speed_force[0]] * INSPIRE_HAND_DOF)
                time.sleep(0.02)
                self.write_registers(REG_FORCE_SET, [speed_force[1]] * INSPIRE_HAND_DOF)
                time.sleep(0.02)
                self._configured_speed_force = speed_force
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


class ModbusPrioritySchedule:
    """Let 50 Hz state reads and hand commands take priority over tactile reads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_state_deadline = 0.0
        self.command_active = threading.Event()

    def set_state_deadline(self, deadline: float) -> None:
        with self._lock:
            self._next_state_deadline = deadline

    def tactile_has_budget(self, guard_s: float) -> bool:
        if self.command_active.is_set():
            return False
        with self._lock:
            deadline = self._next_state_deadline
        return deadline <= 0.0 or (deadline - time.monotonic()) > guard_s


def run_state_publisher(
    args,
    hands: dict[str, InspireModbusHand],
    active_sides: list[str],
    stop_event: threading.Event,
    publisher: ChannelPublisher,
    schedule: ModbusPrioritySchedule,
    profiler: ModbusProfiler,
) -> None:
    period = 1.0 / max(float(args.state_publish_frequency), 1e-6)
    last_q: dict[str, np.ndarray] = {}
    last_time: float | None = None
    last_log_time = 0.0
    open_q = np.asarray(
        resolve_hand_task_pose(args.hand_task, pressed=False),
        dtype=np.float64,
    )

    while not stop_event.is_set():
        loop_start = time.monotonic()
        schedule.set_state_deadline(loop_start + period)
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
        profiler.record_state_cycle(elapsed > period)
        stop_event.wait(max(0.0, period - elapsed))


def run_tactile_publisher(
    args,
    hand: InspireModbusHand,
    stop_event: threading.Event,
    schedule: ModbusPrioritySchedule,
    profiler: ModbusProfiler,
) -> None:
    """Read one low-priority batch at a time and publish latest-value snapshots."""
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 2)
    publisher.bind(f"tcp://{args.tactile_publish_host}:{args.tactile_publish_port}")

    region_values = {
        region.name: np.zeros(region.size, dtype=np.uint16) for region in TACTILE_REGIONS
    }
    valid = np.zeros(TACTILE_REGION_COUNT, dtype=np.bool_)
    updated_time_s = np.zeros(TACTILE_REGION_COUNT, dtype=np.float64)
    update_sequence = np.zeros(TACTILE_REGION_COUNT, dtype=np.int64)
    force_act_g = np.zeros(TACTILE_FORCE_COUNT, dtype=np.int16)
    force_valid = False
    force_updated_time_s = 0.0
    force_update_sequence = 0
    publish_sequence = 0
    schedule_items = [*TACTILE_BATCHES, None]
    item_index = 0
    successful_items: set[int] = set()
    period = 1.0 / max(
        float(args.tactile_full_refresh_hz) * TACTILE_BATCH_COUNT_WITH_FORCE,
        1e-6,
    )
    guard_s = max(0.0, float(args.tactile_state_guard_ms) / 1000.0)
    next_attempt = time.monotonic()
    last_error_log = 0.0
    last_metrics_log = time.monotonic()
    metrics_path = Path(args.tactile_metrics_log).expanduser() if args.tactile_metrics_log else None

    def make_snapshot(metrics: np.ndarray) -> dict:
        return {
            "version": TACTILE_PROTOCOL_VERSION,
            "sequence": publish_sequence,
            "publish_time_s": time.time(),
            "values": flatten_regions(region_values),
            "valid": valid.copy(),
            "updated_time_s": updated_time_s.copy(),
            "update_sequence": update_sequence.copy(),
            "force_act_g": force_act_g.copy(),
            "force_valid": force_valid,
            "force_updated_time_s": force_updated_time_s,
            "force_update_sequence": force_update_sequence,
            "metrics": metrics.copy(),
        }

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_attempt:
                stop_event.wait(min(next_attempt - now, 0.01))
                continue
            if not schedule.tactile_has_budget(guard_s):
                stop_event.wait(0.001)
                continue

            item = schedule_items[item_index]
            read_ok = False
            try:
                if item is None:
                    raw_force = hand.read_registers(
                        REG_FORCE_ACT, TACTILE_FORCE_COUNT, kind="tactile"
                    )
                    force_act_g[:] = np.asarray(raw_force, dtype=np.uint16).view(np.int16)
                    force_valid = True
                    force_updated_time_s = time.time()
                    force_update_sequence += 1
                else:
                    matrices = unpack_batch(
                        item,
                        hand.read_registers(item.address, item.count, kind="tactile"),
                    )
                    update_time = time.time()
                    for region_name, values in matrices.items():
                        region_index = TACTILE_REGION_INDEX_BY_NAME[region_name]
                        region_values[region_name][:] = values
                        valid[region_index] = True
                        updated_time_s[region_index] = update_time
                        update_sequence[region_index] += 1
                read_ok = True
                successful_items.add(item_index)
            except Exception as exc:
                wall_now = time.monotonic()
                if wall_now - last_error_log >= 1.0:
                    print(f"Inspire tactile batch read failed: {exc}")
                    last_error_log = wall_now

            item_index = (item_index + 1) % len(schedule_items)
            if item_index == 0:
                if len(successful_items) == len(schedule_items):
                    profiler.record_full_refresh()
                successful_items.clear()

            metrics_record = profiler.snapshot()
            if read_ok:
                publish_sequence += 1
                try:
                    publisher.send(
                        pack_snapshot(make_snapshot(metrics_record["values"])),
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    pass

            wall_now = time.monotonic()
            if wall_now - last_metrics_log >= args.tactile_metrics_interval:
                serializable = {
                    "time_s": metrics_record["time_s"],
                    "window_s": metrics_record["window_s"],
                    **dict(zip(metrics_record["metric_names"], metrics_record["values"].tolist())),
                }
                print("[InspireTactileModbus] " + " ".join(
                    f"{key}={value:.3f}" for key, value in serializable.items() if key != "time_s"
                ))
                if metrics_path is not None:
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(serializable, ensure_ascii=False) + "\n")
                last_metrics_log = wall_now

            next_attempt = max(next_attempt + period, time.monotonic())
    finally:
        publisher.close(linger=0)
        context.term()


def run_dds_bridge(args, hands: dict[str, InspireModbusHand]) -> None:
    last_command = None
    profile_samples = []
    last_profile_time = time.monotonic()
    active_sides = ["left", "right"] if args.side == "both" else [args.side]
    schedule = ModbusPrioritySchedule()
    profiler = next(iter(hands.values())).profiler
    assert profiler is not None

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_command, last_profile_time
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

        schedule.command_active.set()
        try:
            angles = {
                side: normalized_to_task_angle(
                    dds_q[side], args.hand_task, args.thumb_rotate_default
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
        finally:
            schedule.command_active.clear()

    ChannelFactoryInitialize(args.domain_id, args.network)
    stop_event = threading.Event()
    publisher = None
    if args.publish_state:
        # CycloneDDS 0.10.x lazily builds global IDL/XTypes metadata while a
        # Topic is created.  Creating the state publisher in a worker thread
        # at the same time as the command subscriber can race inside that
        # metadata initialization, causing either an AttributeError in
        # key_scan() or a native segmentation fault.  Initialize all topics
        # serially on this thread before starting the Modbus polling worker.
        publisher = ChannelPublisher("rt/inspire/state", MotorStates_)
        publisher.Init()

    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)

    if publisher is not None:
        state_thread = threading.Thread(
            target=run_state_publisher,
            args=(args, hands, active_sides, stop_event, publisher, schedule, profiler),
            daemon=True,
        )
        state_thread.start()
        print(
            "Modbus -> DDS state publisher running on rt/inspire/state "
            f"at {args.state_publish_frequency:.1f} Hz."
        )

    if args.collect_tactile:
        if "left" not in active_sides:
            raise ValueError("Tactile collection currently requires the left hand to be active")
        tactile_thread = threading.Thread(
            target=run_tactile_publisher,
            args=(args, hands["left"], stop_event, schedule, profiler),
            daemon=True,
        )
        tactile_thread.start()
        print(
            "Low-priority left tactile publisher running at "
            f"{args.tactile_full_refresh_hz:.2f} full refreshes/s on "
            f"tcp://{args.tactile_publish_host}:{args.tactile_publish_port}."
        )

    print(
        "DDS -> Modbus bridge running on rt/inspire/cmd "
        f"for {','.join(active_sides)}. Press Ctrl+C to stop."
    )
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
        "--collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect left RH56DFTP taxels in low-priority Modbus batches.",
    )
    parser.add_argument(
        "--tactile-full-refresh-hz",
        type=float,
        default=2.0,
        help="Target complete tactile refresh rate. Dataset snapshots remain at exporter FPS.",
    )
    parser.add_argument("--tactile-publish-host", default="127.0.0.1")
    parser.add_argument("--tactile-publish-port", type=int, default=5558)
    parser.add_argument(
        "--tactile-state-guard-ms",
        type=float,
        default=6.0,
        help="Do not begin a tactile batch this close to the next 50 Hz state deadline.",
    )
    parser.add_argument("--tactile-metrics-interval", type=float, default=5.0)
    parser.add_argument("--tactile-metrics-log", default="")
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
    if args.tactile_full_refresh_hz <= 0.0:
        raise ValueError("--tactile-full-refresh-hz must be positive")
    if args.tactile_state_guard_ms < 0.0:
        raise ValueError("--tactile-state-guard-ms must be nonnegative")
    profiler = ModbusProfiler(args.tactile_full_refresh_hz)
    hands = {
        "left": InspireModbusHand(
            "left", args.left_ip, args.hand_port, args.device_id, profiler=profiler
        ),
        "right": InspireModbusHand(
            "right", args.right_ip, args.hand_port, args.device_id, profiler=profiler
        ),
    }
    if args.mode == "dds":
        run_dds_bridge(args, hands)
    else:
        run_command(args, hands)


if __name__ == "__main__":
    main()
