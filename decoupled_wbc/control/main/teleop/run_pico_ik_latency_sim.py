"""Direct PICO -> IK -> MuJoCo latency diagnostic.

This tool intentionally bypasses the real-robot WBC, ROS control topics, DDS
commands, camera pipeline, and dataset exporter.  PICO input is sampled on a
dedicated thread.  Pose updates use a latest-only slot while button edges are
latched separately, so slow IK or rendering can never replay a backlog of stale
poses or silently erase a captured button edge.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np
import tyro
import yaml

from decoupled_wbc.control.envs.g1.sim.base_sim import DefaultEnv
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.teleop.teleop_streamer import TeleopStreamer


BUTTON_NAMES = (
    "A",
    "B",
    "X",
    "Y",
    "left_menu_button",
    "right_menu_button",
    "left_axis_click",
    "right_axis_click",
)


@dataclass
class PicoIKLatencySimConfig:
    frequency: float = 50.0
    """IK and MuJoCo update frequency."""

    pico_poll_frequency: float = 200.0
    """Raw PICO polling frequency; pose frames are latest-only."""

    duration: float = 0.0
    """Run duration in seconds; zero runs until the viewer closes or Ctrl+C."""

    onscreen: bool = True
    """Show the MuJoCo viewer."""

    with_hands: bool = True
    """Run the existing hand IK and map hand joints into MuJoCo."""

    high_elbow_pose: bool = False
    """Use the high-elbow G1 default pose."""

    stale_frame_ms: float = 100.0
    """Do not apply a PICO snapshot older than this threshold."""

    stats_interval: float = 1.0
    """Console statistics interval."""

    csv_path: str = ""
    """CSV output path; empty creates logs/pico_ik_latency/<timestamp>.csv."""


@dataclass(frozen=True)
class RawPicoSnapshot:
    sequence: int
    xr_timestamp_ns: int
    local_capture_ns: int
    read_ms: float
    data: dict[str, Any]


@dataclass(frozen=True)
class ButtonEdge:
    name: str
    pressed: bool
    local_capture_ns: int
    xr_timestamp_ns: int


class LatestPicoSampler:
    """Poll PICO quickly, retain only the newest pose, and latch button edges."""

    def __init__(self, streamer: PicoStreamer, poll_frequency: float):
        self.streamer = streamer
        self.period = 1.0 / poll_frequency
        self._lock = threading.Lock()
        self._latest: RawPicoSnapshot | None = None
        self._edges: deque[ButtonEdge] = deque()
        self._sequence = 0
        self._consumed_sequence = 0
        self._last_buttons = {name: False for name in BUTTON_NAMES}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pico-latest-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def take_latest(self) -> tuple[RawPicoSnapshot | None, list[ButtonEdge], int]:
        with self._lock:
            snapshot = self._latest
            edges = list(self._edges)
            self._edges.clear()
            overwritten = 0
            if snapshot is not None:
                if snapshot.sequence <= self._consumed_sequence:
                    snapshot = None
                else:
                    overwritten = max(0, snapshot.sequence - self._consumed_sequence - 1)
                    self._consumed_sequence = snapshot.sequence
            return snapshot, edges, overwritten

    def _read_snapshot(self) -> dict[str, Any]:
        client = self.streamer.xr_client
        return {
            "left_pose": np.asarray(
                client.get_pose_by_name("left_controller"), dtype=np.float64
            ).copy(),
            "right_pose": np.asarray(
                client.get_pose_by_name("right_controller"), dtype=np.float64
            ).copy(),
            "head_pose": np.asarray(client.get_pose_by_name("headset"), dtype=np.float64).copy(),
            "left_trigger": float(client.get_key_value_by_name("left_trigger")),
            "right_trigger": float(client.get_key_value_by_name("right_trigger")),
            "left_grip": float(client.get_key_value_by_name("left_grip")),
            "right_grip": float(client.get_key_value_by_name("right_grip")),
            "A": bool(client.get_button_state_by_name("A")),
            "B": bool(client.get_button_state_by_name("B")),
            "X": bool(client.get_button_state_by_name("X")),
            "Y": bool(client.get_button_state_by_name("Y")),
            "left_menu_button": bool(
                client.get_button_state_by_name("left_menu_button")
            ),
            "right_menu_button": bool(
                client.get_button_state_by_name("right_menu_button")
            ),
            "left_axis_click": bool(client.get_button_state_by_name("left_axis_click")),
            "right_axis_click": bool(client.get_button_state_by_name("right_axis_click")),
            "left_joystick": np.asarray(
                client.get_joystick_state("left"), dtype=np.float64
            ).copy(),
            "right_joystick": np.asarray(
                client.get_joystick_state("right"), dtype=np.float64
            ).copy(),
            "timestamp": int(client.get_timestamp_ns()),
            # These fields are unused when SMPL visualization is disabled, but
            # retain the PicoStreamer input contract.
            "left_hand_tracking_state": None,
            "right_hand_tracking_state": None,
            "motion_tracker_data": None,
            "body_tracking_data": None,
        }

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            start_ns = time.monotonic_ns()
            try:
                data = self._read_snapshot()
            except Exception as exc:
                print(f"[PICO-READ-ERROR] {exc}", flush=True)
                self._stop.wait(0.05)
                continue
            capture_ns = time.monotonic_ns()
            self._sequence += 1
            xr_timestamp_ns = int(data["timestamp"])
            snapshot = RawPicoSnapshot(
                sequence=self._sequence,
                xr_timestamp_ns=xr_timestamp_ns,
                local_capture_ns=capture_ns,
                read_ms=(capture_ns - start_ns) * 1e-6,
                data=data,
            )

            new_edges = []
            for name in BUTTON_NAMES:
                pressed = bool(data[name])
                if pressed != self._last_buttons[name]:
                    new_edges.append(
                        ButtonEdge(
                            name=name,
                            pressed=pressed,
                            local_capture_ns=capture_ns,
                            xr_timestamp_ns=xr_timestamp_ns,
                        )
                    )
                self._last_buttons[name] = pressed

            with self._lock:
                self._latest = snapshot
                self._edges.extend(new_edges)

            deadline += self.period
            wait = deadline - time.monotonic()
            if wait > 0:
                self._stop.wait(wait)
            else:
                deadline = time.monotonic()


class RollingStats:
    def __init__(self, maxlen: int = 500):
        self.values: dict[str, deque[float]] = {}
        self.maxlen = maxlen

    def add(self, **values: float) -> None:
        for name, value in values.items():
            self.values.setdefault(name, deque(maxlen=self.maxlen)).append(float(value))

    def summary(self, name: str) -> tuple[float, float, float, float]:
        values = np.asarray(self.values.get(name, ()), dtype=np.float64)
        if values.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.mean(values)),
            float(np.percentile(values, 95)),
            float(np.percentile(values, 99)),
            float(np.max(values)),
        )


CSV_FIELDS = (
    "wall_time",
    "sequence",
    "xr_timestamp_ns",
    "xr_delta_ms",
    "xr_relative_age_ms",
    "read_ms",
    "snapshot_age_ms",
    "preprocess_ms",
    "ik_ms",
    "sim_apply_ms",
    "total_after_capture_ms",
    "loop_ms",
    "overwritten_pose_frames",
    "stale_dropped",
    "duplicate_xr_timestamp",
    "max_joint_delta",
    "button_edges",
)


def _default_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs/pico_ik_latency") / f"pico_ik_latency_{timestamp}.csv"


def _load_sim_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent / "configs/g1_29dof_gear_wbc.yaml"
    with config_path.open() as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["ENABLE_ELASTIC_BAND"] = False
    config["ENABLE_OFFSCREEN"] = False
    return config


def _build_joint_mapping(
    model: mujoco.MjModel, robot_joint_names: list[str]
) -> list[tuple[int, int, str]]:
    mapping = []
    for robot_index, name in enumerate(robot_joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        mapping.append((robot_index, int(model.jnt_qposadr[joint_id]), name))
    if not mapping:
        raise RuntimeError("No common joints found between the IK robot and MuJoCo model")
    return mapping


def _apply_configuration(
    sim_env: DefaultEnv,
    target_q: np.ndarray,
    mapping: list[tuple[int, int, str]],
) -> None:
    for robot_index, qpos_address, _ in mapping:
        sim_env.mj_data.qpos[qpos_address] = target_q[robot_index]
    sim_env.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim_env.mj_model, sim_env.mj_data)
    sim_env.update_viewer()


def _format_stats(stats: RollingStats, applied: int, stale: int, overwritten: int) -> str:
    parts = [f"frames={applied}", f"stale={stale}", f"overwritten={overwritten}"]
    for name in ("read_ms", "snapshot_age_ms", "preprocess_ms", "ik_ms", "loop_ms"):
        mean, p95, p99, maximum = stats.summary(name)
        parts.append(
            f"{name} mean/p95/p99/max={mean:.2f}/{p95:.2f}/{p99:.2f}/{maximum:.2f}"
        )
    return " | ".join(parts)


def main(config: PicoIKLatencySimConfig) -> None:
    if config.frequency <= 0 or config.pico_poll_frequency <= 0:
        raise ValueError("frequency and pico_poll_frequency must be positive")
    if config.stale_frame_ms <= 0:
        raise ValueError("stale_frame_ms must be positive")

    robot_model = instantiate_g1_robot_model(
        high_elbow_pose=config.high_elbow_pose,
        with_hands=config.with_hands,
    )
    left_hand_solver, right_hand_solver = (
        instantiate_g1_hand_ik_solver() if config.with_hands else (None, None)
    )
    retargeting_ik = TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_solver,
        right_hand_ik_solver=right_hand_solver,
        enable_visualization=False,
        body_active_joint_groups=["upper_body"],
    )

    sim_env = DefaultEnv(
        _load_sim_config(),
        env_name="default",
        onscreen=config.onscreen,
        offscreen=False,
        enable_image_publish=False,
    )
    mapping = _build_joint_mapping(sim_env.mj_model, robot_model.joint_names)
    target_q = robot_model.default_body_pose.copy()
    _apply_configuration(sim_env, target_q, mapping)
    print(
        f"Mapped {len(mapping)}/{len(robot_model.joint_names)} IK joints into MuJoCo",
        flush=True,
    )

    pico = PicoStreamer(enable_smpl_visualization=False)
    pico.start_streaming()
    teleop_streamer = TeleopStreamer(
        robot_model=robot_model,
        body_control_device="pico",
        hand_control_device="pico" if config.with_hands else None,
        enable_real_device=False,
    )
    teleop_streamer.body_streamer = pico
    teleop_streamer.enable_real_device = True
    teleop_streamer.calibrate(reference_body_q=target_q)

    initial_stream = pico.get()
    body_data, left_hand_data, right_hand_data = teleop_streamer.pre_process(
        initial_stream.ik_data
    )
    retargeting_ik.reset(reference_full_q=target_q)
    print("Warming up IK...", flush=True)
    target_q = retargeting_ik.compute_joint_positions(
        body_data, left_hand_data, right_hand_data
    )
    _apply_configuration(sim_env, target_q, mapping)
    print("PICO direct-IK latency diagnostic started", flush=True)

    sampler = LatestPicoSampler(pico, config.pico_poll_frequency)
    sampler.start()

    csv_path = Path(config.csv_path).expanduser() if config.csv_path else _default_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats = RollingStats()
    applied = 0
    stale_count = 0
    overwritten_total = 0
    last_xr_timestamp = 0
    xr_clock_offset_ns: int | None = None
    last_report = time.monotonic()
    previous_q = target_q.copy()
    started = time.monotonic()
    loop_period = 1.0 / config.frequency
    deadline = time.monotonic()

    try:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while config.duration <= 0 or time.monotonic() - started < config.duration:
                if sim_env.viewer is not None and not sim_env.viewer.is_running():
                    break
                loop_start_ns = time.monotonic_ns()
                snapshot, edges, overwritten = sampler.take_latest()
                overwritten_total += overwritten

                for edge in edges:
                    age_ms = (loop_start_ns - edge.local_capture_ns) * 1e-6
                    state = "DOWN" if edge.pressed else "UP"
                    print(
                        f"[BUTTON] {edge.name} {state} capture_to_consumer={age_ms:.2f}ms",
                        flush=True,
                    )

                if snapshot is not None:
                    snapshot_age_ms = (loop_start_ns - snapshot.local_capture_ns) * 1e-6
                    stale = snapshot_age_ms > config.stale_frame_ms
                    duplicate_timestamp = snapshot.xr_timestamp_ns == last_xr_timestamp
                    xr_delta_ms = (
                        (snapshot.xr_timestamp_ns - last_xr_timestamp) * 1e-6
                        if last_xr_timestamp > 0
                        else 0.0
                    )
                    if xr_clock_offset_ns is None and snapshot.xr_timestamp_ns > 0:
                        xr_clock_offset_ns = (
                            snapshot.local_capture_ns - snapshot.xr_timestamp_ns
                        )
                    xr_relative_age_ms = (
                        (
                            loop_start_ns
                            - snapshot.xr_timestamp_ns
                            - xr_clock_offset_ns
                        )
                        * 1e-6
                        if xr_clock_offset_ns is not None and snapshot.xr_timestamp_ns > 0
                        else 0.0
                    )

                    preprocess_ms = 0.0
                    ik_ms = 0.0
                    sim_apply_ms = 0.0
                    max_joint_delta = 0.0
                    if stale:
                        stale_count += 1
                    else:
                        preprocess_start = time.monotonic_ns()
                        raw_stream = pico._generate_unified_raw_data(snapshot.data)
                        body_data, left_hand_data, right_hand_data = (
                            teleop_streamer.pre_process(raw_stream.ik_data)
                        )
                        preprocess_end = time.monotonic_ns()
                        target_q = retargeting_ik.compute_joint_positions(
                            body_data, left_hand_data, right_hand_data
                        )
                        ik_end = time.monotonic_ns()
                        _apply_configuration(sim_env, target_q, mapping)
                        apply_end = time.monotonic_ns()
                        preprocess_ms = (preprocess_end - preprocess_start) * 1e-6
                        ik_ms = (ik_end - preprocess_end) * 1e-6
                        sim_apply_ms = (apply_end - ik_end) * 1e-6
                        max_joint_delta = float(np.max(np.abs(target_q - previous_q)))
                        previous_q = target_q.copy()
                        applied += 1

                    loop_end_ns = time.monotonic_ns()
                    loop_ms = (loop_end_ns - loop_start_ns) * 1e-6
                    total_after_capture_ms = (
                        loop_end_ns - snapshot.local_capture_ns
                    ) * 1e-6
                    button_text = ";".join(
                        f"{edge.name}:{'down' if edge.pressed else 'up'}" for edge in edges
                    )
                    row = {
                        "wall_time": time.time(),
                        "sequence": snapshot.sequence,
                        "xr_timestamp_ns": snapshot.xr_timestamp_ns,
                        "xr_delta_ms": xr_delta_ms,
                        "xr_relative_age_ms": xr_relative_age_ms,
                        "read_ms": snapshot.read_ms,
                        "snapshot_age_ms": snapshot_age_ms,
                        "preprocess_ms": preprocess_ms,
                        "ik_ms": ik_ms,
                        "sim_apply_ms": sim_apply_ms,
                        "total_after_capture_ms": total_after_capture_ms,
                        "loop_ms": loop_ms,
                        "overwritten_pose_frames": overwritten,
                        "stale_dropped": int(stale),
                        "duplicate_xr_timestamp": int(duplicate_timestamp),
                        "max_joint_delta": max_joint_delta,
                        "button_edges": button_text,
                    }
                    writer.writerow(row)
                    stats.add(
                        read_ms=snapshot.read_ms,
                        snapshot_age_ms=snapshot_age_ms,
                        preprocess_ms=preprocess_ms,
                        ik_ms=ik_ms,
                        sim_apply_ms=sim_apply_ms,
                        total_after_capture_ms=total_after_capture_ms,
                        loop_ms=loop_ms,
                    )
                    last_xr_timestamp = snapshot.xr_timestamp_ns

                now = time.monotonic()
                if now - last_report >= config.stats_interval:
                    print(
                        _format_stats(
                            stats,
                            applied=applied,
                            stale=stale_count,
                            overwritten=overwritten_total,
                        ),
                        flush=True,
                    )
                    csv_file.flush()
                    last_report = now

                deadline += loop_period
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        pico.stop_streaming()
        if sim_env.viewer is not None:
            sim_env.viewer.close()

    print(f"Latency CSV: {csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(PicoIKLatencySimConfig))
