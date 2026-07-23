"""Run the real PICO teleop chain against a directly driven MuJoCo robot.

This diagnostic reuses the same PicoStreamer, TeleopStreamer, TeleopPolicy,
activation state machine, calibration, and TeleopRetargetingIK as collection.
Only the real robot/control-loop endpoint is replaced by MuJoCo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import csv
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import tyro
import yaml

from decoupled_wbc.control.envs.g1.sim.base_sim import DefaultEnv
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK


@dataclass
class PicoIKLatencySimConfig:
    frequency: float = 20.0
    """Match the collection teleop loop frequency."""

    duration: float = 0.0
    """Run duration in seconds; zero runs until the viewer closes or Ctrl+C."""

    onscreen: bool = True
    """Show the MuJoCo viewer."""

    with_hands: bool = True
    """Use the same PICO hand and hand-IK setup as collection."""

    enable_waist: bool = False
    """Match the collection enable_waist option."""

    high_elbow_pose: bool = False
    """Match the collection high_elbow_pose option."""

    initial_pose_duration: float = 2.0
    """Initial-pose hold used by the real teleop publisher."""

    lower_body_ack_delay: float = 0.05
    """Simulated lower-body status acknowledgement delay."""

    stats_interval: float = 1.0
    """Console latency statistics interval."""

    csv_path: str = ""
    """CSV path; empty creates logs/pico_ik_latency/<timestamp>.csv."""


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
    "iteration",
    "teleop_state",
    "lower_body_policy_active",
    "pending_policy_action",
    "get_action_ms",
    "sim_apply_ms",
    "loop_ms",
    "period_overrun_ms",
    "max_upper_joint_delta",
)


def _default_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs/pico_ik_latency") / f"real_chain_{timestamp}.csv"


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
        if joint_id >= 0:
            mapping.append((robot_index, int(model.jnt_qposadr[joint_id]), name))
    if not mapping:
        raise RuntimeError("IK robot and MuJoCo model have no common joints")
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


def _format_stats(stats: RollingStats, state: str, lower_active: bool) -> str:
    parts = [f"state={state}", f"lower_body={'on' if lower_active else 'off'}"]
    for name in ("get_action_ms", "sim_apply_ms", "loop_ms", "period_overrun_ms"):
        mean, p95, p99, maximum = stats.summary(name)
        parts.append(
            f"{name} mean/p95/p99/max="
            f"{mean:.2f}/{p95:.2f}/{p99:.2f}/{maximum:.2f}"
        )
    return " | ".join(parts)


def main(config: PicoIKLatencySimConfig) -> None:
    if config.frequency <= 0:
        raise ValueError("frequency must be positive")
    if config.initial_pose_duration < 0 or config.lower_body_ack_delay < 0:
        raise ValueError("durations must be non-negative")

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
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

    # These constructor arguments intentionally match run_teleop_policy_loop.
    # The keyboard subscriber is disabled because this standalone diagnostic
    # has no ROS node; all PICO controls and their state machine are unchanged.
    teleop_policy = TeleopPolicy(
        robot_model=robot_model,
        retargeting_ik=retargeting_ik,
        body_control_device="pico",
        hand_control_device="pico" if config.with_hands else None,
        enable_real_device=True,
        pico_vis_smpl=False,
        activate_keyboard_listener=False,
    )

    sim_env = DefaultEnv(
        _load_sim_config(),
        env_name="default",
        onscreen=config.onscreen,
        offscreen=False,
        enable_image_publish=False,
    )
    mapping = _build_joint_mapping(sim_env.mj_model, robot_model.joint_names)
    upper_indices = robot_model.get_joint_group_indices("upper_body")

    # Collection moves the robot to this pose before normal teleoperation.  Use
    # it as both MuJoCo state and the first robot-feedback sample seen by policy.
    sim_q = robot_model.initial_body_pose.copy()
    _apply_configuration(sim_env, sim_q, mapping)
    teleop_policy.set_robot_state({"q": sim_q.copy()})

    print(
        f"Mapped {len(mapping)}/{len(robot_model.joint_names)} robot joints into MuJoCo",
        flush=True,
    )
    print(
        "Initial pose ready. Press A+B+X+Y to enable the simulated lower-body "
        "policy, then press A+X to start/pause/resume upper-body teleoperation.",
        flush=True,
    )
    if config.initial_pose_duration:
        print(
            f"Holding the collection initial pose for {config.initial_pose_duration:.1f}s",
            flush=True,
        )
        time.sleep(config.initial_pose_duration)

    csv_path = Path(config.csv_path).expanduser() if config.csv_path else _default_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats = RollingStats()
    loop_period = 1.0 / config.frequency
    started = time.monotonic()
    deadline = started
    last_report = started
    iteration = 0
    pending_policy_action: bool | None = None
    pending_ack_time: float | None = None
    previous_upper_q = sim_q[upper_indices].copy()

    try:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while config.duration <= 0 or time.monotonic() - started < config.duration:
                if sim_env.viewer is not None and not sim_env.viewer.is_running():
                    break

                loop_start = time.monotonic()
                if pending_ack_time is not None and loop_start >= pending_ack_time:
                    teleop_policy.set_lower_body_policy_active(pending_policy_action)
                    print(
                        "Simulated lower-body policy request acknowledged: "
                        f"{'enabled' if pending_policy_action else 'disabled'}",
                        flush=True,
                    )
                    pending_policy_action = None
                    pending_ack_time = None

                # Use simulated feedback exactly where the real loop supplies
                # robot state before calling TeleopPolicy.get_action().
                teleop_policy.set_robot_state({"q": sim_q.copy()})
                get_action_start = time.monotonic()
                action = teleop_policy.get_action()
                get_action_end = time.monotonic()

                requested_policy_action = action.get("set_policy_action")
                if requested_policy_action is not None:
                    requested_policy_action = bool(requested_policy_action)
                    if requested_policy_action != pending_policy_action:
                        pending_policy_action = requested_policy_action
                        pending_ack_time = (
                            get_action_end + config.lower_body_ack_delay
                        )
                        print(
                            "Simulated lower-body policy request queued: "
                            f"{'enable' if pending_policy_action else 'disable'}",
                            flush=True,
                        )

                sim_apply_start = time.monotonic()
                target_upper_q = np.asarray(
                    action["target_upper_body_pose"], dtype=np.float64
                )
                sim_q[upper_indices] = target_upper_q
                _apply_configuration(sim_env, sim_q, mapping)
                sim_apply_end = time.monotonic()

                loop_ms = (sim_apply_end - loop_start) * 1000.0
                period_overrun_ms = max(0.0, loop_ms - loop_period * 1000.0)
                max_upper_joint_delta = float(
                    np.max(np.abs(target_upper_q - previous_upper_q))
                )
                previous_upper_q = target_upper_q.copy()
                get_action_ms = (get_action_end - get_action_start) * 1000.0
                sim_apply_ms = (sim_apply_end - sim_apply_start) * 1000.0
                state = teleop_policy._teleop_state

                writer.writerow(
                    {
                        "wall_time": time.time(),
                        "iteration": iteration,
                        "teleop_state": state,
                        "lower_body_policy_active": int(
                            teleop_policy._lower_body_policy_active
                        ),
                        "pending_policy_action": (
                            "" if pending_policy_action is None else int(pending_policy_action)
                        ),
                        "get_action_ms": get_action_ms,
                        "sim_apply_ms": sim_apply_ms,
                        "loop_ms": loop_ms,
                        "period_overrun_ms": period_overrun_ms,
                        "max_upper_joint_delta": max_upper_joint_delta,
                    }
                )
                stats.add(
                    get_action_ms=get_action_ms,
                    sim_apply_ms=sim_apply_ms,
                    loop_ms=loop_ms,
                    period_overrun_ms=period_overrun_ms,
                )

                now = time.monotonic()
                if now - last_report >= config.stats_interval:
                    print(
                        _format_stats(
                            stats,
                            state=state,
                            lower_active=teleop_policy._lower_body_policy_active,
                        ),
                        flush=True,
                    )
                    csv_file.flush()
                    last_report = now

                iteration += 1
                deadline += loop_period
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        teleop_policy.close()
        if sim_env.viewer is not None:
            sim_env.viewer.close()

    print(f"Latency CSV: {csv_path.resolve()}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(PicoIKLatencySimConfig))
