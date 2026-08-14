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
REG_ERROR = 1606
REG_STATUS = 1612

INSPIRE_HAND_DOF = 6
THUMB_ROTATE_INDEX = 5
DEFAULT_THUMB_ROTATE = 0.5
STATE_TO_FORCE_REGISTER_COUNT = REG_FORCE_ACT - REG_ANGLE_ACT + INSPIRE_HAND_DOF


class ModbusTcpError(RuntimeError):
    pass


class ModbusProfiler:
    """Rolling Modbus utilization and deadline statistics."""

    METRIC_NAMES = MODBUS_METRIC_NAMES

    def __init__(
        self,
        target_full_refresh_hz: float,
        tactile_batches_per_refresh: int,
        window_s: float = 30.0,
    ):
        self.target_full_refresh_hz = float(target_full_refresh_hz)
        self.tactile_batches_per_refresh = int(tactile_batches_per_refresh)
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
            safe_budget_ratio * 1000.0
            / (tactile_p95_ms * self.tactile_batches_per_refresh)
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

    def read_angle_and_force(self) -> tuple[np.ndarray, np.ndarray]:
        """Read 50 Hz angle feedback and force feedback in one FC=03 request."""
        values = self.read_registers(
            REG_ANGLE_ACT,
            STATE_TO_FORCE_REGISTER_COUNT,
            kind="state",
        )
        raw = np.asarray(values, dtype=np.uint16)
        q = np.clip(raw[:INSPIRE_HAND_DOF].astype(np.float64) / 1000.0, 0.0, 1.0)
        force_offset = REG_FORCE_ACT - REG_ANGLE_ACT
        force_g = raw[force_offset : force_offset + INSPIRE_HAND_DOF].view(np.int16)
        return q, force_g.copy()

    def read_force_normalized(self) -> np.ndarray:
        values = self.read_registers(REG_FORCE_ACT, INSPIRE_HAND_DOF)
        return np.clip(np.asarray(values, dtype=np.float64) / 1000.0, 0.0, None)

    def read_error_and_status(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the six actuator error codes and six motion status codes."""
        register_count = (REG_STATUS - REG_ERROR + INSPIRE_HAND_DOF) // 2
        words = self.read_registers(REG_ERROR, register_count, kind="state")
        raw = np.frombuffer(
            struct.pack(">" + "H" * len(words), *words),
            dtype=np.uint8,
        )
        errors = raw[:INSPIRE_HAND_DOF].copy()
        status_offset = REG_STATUS - REG_ERROR
        statuses = raw[status_offset : status_offset + INSPIRE_HAND_DOF].copy()
        return errors, statuses

    def clear_error(self) -> None:
        self.write_register(REG_CLEAR_ERROR, 1)
        time.sleep(0.02)

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
    """Thread-safe latest-target cache consumed by the single Modbus I/O loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._targets: dict[str, list[int]] = {}
        self._generation = 0
        self._applied_generation = 0

    def set_targets(self, targets: dict[str, list[int]]) -> None:
        with self._lock:
            copied = {side: values.copy() for side, values in targets.items()}
            if copied != self._targets:
                self._targets = copied
                self._generation += 1

    def get_pending_targets(self) -> tuple[int, dict[str, list[int]]]:
        with self._lock:
            if self._applied_generation == self._generation:
                return self._generation, {}
            return self._generation, {
                side: values.copy() for side, values in self._targets.items()
            }

    def mark_applied(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._applied_generation = generation


class TactileBatchPlanner:
    """Round-robin tactile batches that only run when the 50 Hz deadline permits."""

    def __init__(
        self,
        target_full_refresh_hz: float,
        *,
        fallback_batch_ms: float,
        deadline_margin_ms: float,
    ):
        self.items = list(TACTILE_BATCHES)
        self.item_index = 0
        self.successful_items: set[int] = set()
        self.period_s = 1.0 / max(float(target_full_refresh_hz) * len(self.items), 1e-6)
        self.next_attempt_s = time.monotonic()
        self.fallback_batch_s = max(0.0, float(fallback_batch_ms) / 1000.0)
        self.deadline_margin_s = max(0.0, float(deadline_margin_ms) / 1000.0)
        self._durations_s = {item.name: deque(maxlen=128) for item in self.items}
        self._deferrals = {item.name: 0 for item in self.items}

    @property
    def item(self):
        return self.items[self.item_index]

    def predicted_duration_s(self) -> float:
        samples = self._durations_s[self.item.name]
        if len(samples) < 8 or self._deferrals[self.item.name] >= len(self.items):
            return self.fallback_batch_s
        return float(np.percentile(samples, 95))

    def can_run(self, now_s: float, deadline_s: float) -> bool:
        return now_s >= self.next_attempt_s and (
            deadline_s - now_s >= self.predicted_duration_s() + self.deadline_margin_s
        )

    def is_due(self, now_s: float) -> bool:
        return now_s >= self.next_attempt_s

    def defer_attempt(self, now_s: float) -> None:
        """Try another batch without letting one slow batch block all taxels."""
        self._deferrals[self.item.name] += 1
        self.item_index = (self.item_index + 1) % len(self.items)
        self.next_attempt_s = max(self.next_attempt_s + self.period_s, now_s)

    def finish_attempt(self, *, success: bool, elapsed_s: float, now_s: float) -> bool:
        self._durations_s[self.item.name].append(float(elapsed_s))
        self._deferrals[self.item.name] = 0
        if success:
            self.successful_items.add(self.item_index)
        self.item_index = (self.item_index + 1) % len(self.items)
        completed_full_refresh = len(self.successful_items) == len(self.items)
        if completed_full_refresh:
            self.successful_items.clear()
        self.next_attempt_s = max(self.next_attempt_s + self.period_s, now_s)
        return completed_full_refresh


def run_state_publisher(
    args,
    hands: dict[str, InspireModbusHand],
    active_sides: list[str],
    stop_event: threading.Event,
    publisher: ChannelPublisher | None,
    schedule: ModbusPrioritySchedule,
    profiler: ModbusProfiler,
) -> None:
    """Run all left-hand Modbus I/O in one 50 Hz, deadline-aware worker."""
    period = 1.0 / max(float(args.state_publish_frequency), 1e-6)
    last_q: dict[str, np.ndarray] = {}
    last_time: float | None = None
    last_log_time = 0.0
    open_q = np.asarray(
        resolve_hand_task_pose(args.hand_task, pressed=False),
        dtype=np.float64,
    )

    tactile_publisher = None
    tactile_context = None
    planner = None
    region_values = None
    valid = None
    updated_time_s = None
    update_sequence = None
    force_act_g = None
    force_valid = False
    force_updated_time_s = 0.0
    force_update_sequence = 0
    publish_sequence = 0
    last_tactile_error_log = 0.0
    last_metrics_log = time.monotonic()
    metrics_path = Path(args.tactile_metrics_log).expanduser() if args.tactile_metrics_log else None
    next_cycle_start = time.monotonic()

    if args.collect_tactile:
        if "left" not in active_sides:
            raise ValueError("Tactile collection currently requires the left hand to be active")
        tactile_context = zmq.Context()
        tactile_publisher = tactile_context.socket(zmq.PUB)
        tactile_publisher.setsockopt(zmq.SNDHWM, 2)
        tactile_publisher.bind(f"tcp://{args.tactile_publish_host}:{args.tactile_publish_port}")
        planner = TactileBatchPlanner(
            args.tactile_full_refresh_hz,
            fallback_batch_ms=args.tactile_default_batch_ms,
            deadline_margin_ms=args.tactile_state_guard_ms,
        )
        region_values = {
            region.name: np.zeros(region.size, dtype=np.uint16) for region in TACTILE_REGIONS
        }
        valid = np.zeros(TACTILE_REGION_COUNT, dtype=np.bool_)
        updated_time_s = np.zeros(TACTILE_REGION_COUNT, dtype=np.float64)
        update_sequence = np.zeros(TACTILE_REGION_COUNT, dtype=np.int64)
        force_act_g = np.zeros(TACTILE_FORCE_COUNT, dtype=np.int16)

    def make_snapshot(metrics: np.ndarray) -> dict:
        assert region_values is not None and valid is not None
        assert updated_time_s is not None and update_sequence is not None and force_act_g is not None
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
            loop_start = time.monotonic()
            deadline = next_cycle_start + period
            try:
                generation, pending_targets = schedule.get_pending_targets()
                for side, target in pending_targets.items():
                    if side in active_sides:
                        errors, statuses = hands[side].read_error_and_status()
                        protected = bool(np.any(errors)) or bool(
                            np.any(np.isin(statuses, [5, 6, 7]))
                        )
                        if protected:
                            print(
                                f"Inspire {side} protection before new target: "
                                f"errors={errors.tolist()} statuses={statuses.tolist()}; "
                                "clearing recoverable errors once."
                            )
                            hands[side].clear_error()
                        else:
                            print(
                                f"Inspire {side} status before new target: "
                                f"errors={errors.tolist()} statuses={statuses.tolist()}"
                            )
                        hands[side].set_angle(target, speed=args.speed, force=args.force)
                if pending_targets:
                    schedule.mark_applied(generation)

                q = {"right": open_q.copy(), "left": open_q.copy()}
                force_g: dict[str, np.ndarray | None] = {"right": None, "left": None}
                for side in active_sides:
                    q[side], force_g[side] = hands[side].read_angle_and_force()

                now = time.monotonic()
                dq = {}
                for side in ("right", "left"):
                    if last_time is None or side not in last_q or side not in active_sides:
                        dq[side] = np.zeros(INSPIRE_HAND_DOF, dtype=np.float64)
                    else:
                        dq[side] = (q[side] - last_q[side]) / max(now - last_time, 1e-6)

                if force_act_g is not None and force_g["left"] is not None:
                    force_act_g[:] = force_g["left"]
                    force_updated_time_s = time.time()
                    force_update_sequence += 1
                    force_valid = True

                if publisher is not None:
                    tau = {
                        side: (
                            np.clip(force_g[side].astype(np.float64) / 1000.0, 0.0, None)
                            if args.read_force_state and force_g[side] is not None
                            else None
                        )
                        for side in ("right", "left")
                    }
                    publisher.Write(
                        _make_inspire_state_msg(
                            q["right"], q["left"], dq["right"], dq["left"],
                            tau["right"], tau["left"],
                        )
                    )
                last_q = q
                last_time = now

                tactile_now = time.monotonic()
                if planner is not None and planner.can_run(tactile_now, deadline):
                    item = planner.item
                    tactile_start = tactile_now
                    read_ok = False
                    try:
                        matrices = unpack_batch(
                            item,
                            hands["left"].read_registers(item.address, item.count, kind="tactile"),
                        )
                        update_time = time.time()
                        assert region_values is not None and valid is not None
                        assert updated_time_s is not None and update_sequence is not None
                        for region_name, values in matrices.items():
                            region_index = TACTILE_REGION_INDEX_BY_NAME[region_name]
                            region_values[region_name][:] = values
                            valid[region_index] = True
                            updated_time_s[region_index] = update_time
                            update_sequence[region_index] += 1
                        read_ok = True
                    except Exception as exc:
                        wall_now = time.monotonic()
                        if wall_now - last_tactile_error_log >= 1.0:
                            print(f"Inspire tactile batch read failed: {exc}")
                            last_tactile_error_log = wall_now
                    completed = planner.finish_attempt(
                        success=read_ok,
                        elapsed_s=time.monotonic() - tactile_start,
                        now_s=time.monotonic(),
                    )
                    if completed:
                        profiler.record_full_refresh()
                elif planner is not None and planner.is_due(tactile_now):
                    planner.defer_attempt(tactile_now)
            except Exception as exc:
                now = time.monotonic()
                if now - last_log_time >= 1.0:
                    print(f"Inspire priority I/O cycle failed: {exc}")
                    last_log_time = now

            elapsed = time.monotonic() - loop_start
            profiler.record_state_cycle(elapsed > period)
            if tactile_publisher is not None:
                metrics_record = profiler.snapshot()
                publish_sequence += 1
                try:
                    tactile_publisher.send(
                        pack_snapshot(make_snapshot(metrics_record["values"])), flags=zmq.NOBLOCK
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
            next_cycle_start += period
            now = time.monotonic()
            if now - next_cycle_start > period:
                # Drop only a full stale cycle (for example the one-time hand
                # configuration); catch up sub-period jitter on the next loop.
                next_cycle_start = now
            stop_event.wait(max(0.0, next_cycle_start - now))
    finally:
        if tactile_publisher is not None:
            tactile_publisher.close(linger=0)
        if tactile_context is not None:
            tactile_context.term()


def run_dds_bridge(args, hands: dict[str, InspireModbusHand]) -> None:
    last_command = None
    active_sides = ["left", "right"] if args.side == "both" else [args.side]
    schedule = ModbusPrioritySchedule()
    profiler = next(iter(hands.values())).profiler
    assert profiler is not None

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_command
        if len(msg.cmds) < 12:
            print(f"skip short inspire command: {len(msg.cmds)}")
            return

        right_q = tuple(float(msg.cmds[i].q) for i in range(6))
        left_q = tuple(float(msg.cmds[i + 6].q) for i in range(6))
        dds_q = {"left": left_q, "right": right_q}
        command_key = tuple(dds_q[side] for side in active_sides)
        try:
            angles = {
                side: normalized_to_task_angle(
                    dds_q[side], args.hand_task, args.thumb_rotate_default
                )
                for side in active_sides
            }
            schedule.set_targets(angles)
            if command_key != last_command:
                summary = " ".join(
                    f"{side}={angles[side]}" for side in active_sides
                )
                print(f"DDS target queued for 50 Hz Modbus I/O: {summary}")
                last_command = command_key
        except Exception as exc:
            print(f"DDS target mapping failed: {exc}")

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

    if args.collect_tactile and "left" not in active_sides:
        raise ValueError("Tactile collection currently requires the left hand to be active")
    state_thread = threading.Thread(
        target=run_state_publisher,
        args=(args, hands, active_sides, stop_event, publisher, schedule, profiler),
        daemon=True,
    )
    state_thread.start()
    if publisher is not None:
        print(
            "Modbus -> DDS state publisher running on rt/inspire/state "
            f"at {args.state_publish_frequency:.1f} Hz."
        )
    if args.collect_tactile:
        print(
            "Deadline-aware left tactile publisher running at "
            f"up to {args.tactile_full_refresh_hz:.2f} full refreshes/s on "
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
        help="Publish FORCE_ACT from the combined ANGLE_ACT-to-FORCE_ACT state read as tau_est.",
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
        default=5.0,
        help="Target complete tactile refresh rate. Dataset snapshots remain at exporter FPS.",
    )
    parser.add_argument("--tactile-publish-host", default="127.0.0.1")
    parser.add_argument("--tactile-publish-port", type=int, default=5558)
    parser.add_argument(
        "--tactile-state-guard-ms",
        type=float,
        default=0.0,
        help="Extra margin after the dynamically estimated tactile batch duration.",
    )
    parser.add_argument(
        "--tactile-default-batch-ms",
        type=float,
        default=6.0,
        help="Initial duration estimate before per-batch P95 timing is available.",
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
    if args.tactile_default_batch_ms <= 0.0:
        raise ValueError("--tactile-default-batch-ms must be positive")
    profiler = ModbusProfiler(
        args.tactile_full_refresh_hz,
        tactile_batches_per_refresh=len(TACTILE_BATCHES),
    )
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
