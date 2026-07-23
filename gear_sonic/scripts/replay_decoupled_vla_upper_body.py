#!/usr/bin/env python3
"""Safely replay both arms and Inspire hands from one LeRobot episode on a real G1.

This is a direct DDS deployment: legs and waist remain in damping mode while
only motors 15:29 are position-controlled.  The robot must be fully suspended.
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEFT_ARM = np.arange(15, 22)
RIGHT_ARM = np.arange(22, 29)
ARMS = np.arange(15, 29)
LOWER_BODY = np.arange(0, 15)
POS_STOP_F = 2146000000.0
ARM_KP = np.array([100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0])
ARM_KD = np.array([5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
LOWER_BODY_KD = np.array(
    [2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0]
)
LEFT_ARM_LOWER = np.array([-3.0892, -1.5882, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
LEFT_ARM_UPPER = np.array([2.6704, 2.2515, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
RIGHT_ARM_LOWER = np.array([-3.0892, -2.2515, -2.618, -1.0472, -1.9722, -1.6144, -1.6144])
RIGHT_ARM_UPPER = np.array([2.6704, 1.5882, 2.618, 2.0944, 1.9722, 1.6144, 1.6144])
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
GRAVITY_URDF = PROJECT_ROOT / "decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf"
RH56E2_ADDED_MASS = 0.6200
RH56E2_ADDED_COM = {
    "left_wrist_yaw_joint": np.array([0.15919150, -0.00103825, 0.00622361]),
    "right_wrist_yaw_joint": np.array([0.14969764, 0.00160315, 0.00629971]),
}


class ArmGravityCompensator:
    """Pinocchio RNEA compensation matching the reference direct-joint client."""

    def __init__(self, scale: float) -> None:
        try:
            import pinocchio as pin
        except ImportError as error:
            raise RuntimeError("缺少 pinocchio，请使用 decoupled_vla_collection 环境") from error
        if not GRAVITY_URDF.is_file():
            raise FileNotFoundError(f"重力补偿 URDF 不存在：{GRAVITY_URDF}")
        model = pin.buildModelFromUrdf(str(GRAVITY_URDF))
        missing = [name for name in ARM_JOINT_NAMES if not model.existJointName(name)]
        if missing:
            raise ValueError(f"URDF 缺少双臂关节：{missing}")
        zero_inertia = np.zeros((3, 3), dtype=np.float64)
        for joint_name, com in RH56E2_ADDED_COM.items():
            joint_id = model.getJointId(joint_name)
            model.inertias[joint_id] += pin.Inertia(RH56E2_ADDED_MASS, com, zero_inertia)
        self._pin = pin
        self._model = model
        self._data = model.createData()
        self._scale = scale

    def compute(self, target: np.ndarray, measured: np.ndarray) -> np.ndarray:
        q = measured.copy()
        q[ARMS] = target
        tau = np.asarray(
            self._pin.rnea(
                self._model,
                self._data,
                q,
                np.zeros(self._model.nv),
                np.zeros(self._model.nv),
            ),
            dtype=np.float64,
        )
        result = tau[ARMS] * self._scale
        if result.shape != (14,) or not np.isfinite(result).all():
            raise RuntimeError("Pinocchio 返回了无效的重力补偿力矩")
        return result


class EStop:
    def __init__(self) -> None:
        self.latched = False
        self.reason = ""

    def trigger(self, reason: str) -> None:
        if not self.latched:
            self.latched = True
            self.reason = reason
            print(f"\n[E-STOP] 已锁存：{reason}", flush=True)


class G1ArmDDS:
    """Direct 29-DOF LowCmd transport with arm-only position control."""

    def __init__(self, network_interface: str, gravity_scale: float) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__LowCmd_,
            unitree_go_msg_dds__MotorCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        available = {name for _, name in socket.if_nameindex()}
        if network_interface not in available:
            raise ValueError(
                f"网卡 {network_interface!r} 不存在；可选：{', '.join(sorted(available))}"
            )
        ChannelFactoryInitialize(0, network_interface)
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._q: np.ndarray | None = None
        self._stamp = 0.0
        self._mode_machine = 0
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._enabled = False
        self._damping = False
        self._left_target: np.ndarray | None = None
        self._right_target: np.ndarray | None = None
        self._publisher_stop = threading.Event()
        self._publisher_error: Exception | None = None
        self._publisher_thread: threading.Thread | None = None
        self._gravity = ArmGravityCompensator(gravity_scale)
        self._motion_switcher = MotionSwitcherClient()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_state, 10)
        self._hand_cmd = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        self._hand_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._hand_publisher.Init()
        self._hand_q: np.ndarray | None = None
        self._hand_stamp = 0.0
        self._hand_subscriber = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self._hand_subscriber.Init(self._on_hand_state, 10)
        self._wait_for_state(5.0)

    def _on_state(self, msg: object) -> None:
        with self._state_lock:
            self._q = np.array([motor.q for motor in msg.motor_state[:29]], dtype=np.float64)
            self._mode_machine = int(msg.mode_machine)
            self._stamp = time.monotonic()

    def _on_hand_state(self, msg: object) -> None:
        if len(msg.states) < 12:
            return
        with self._state_lock:
            self._hand_q = np.array([state.q for state in msg.states[:12]], dtype=np.float64)
            self._hand_stamp = time.monotonic()

    def _wait_for_state(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                if self._q is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("5 秒内未收到 rt/lowstate，请检查 G1、网卡和 DDS domain")

    def state(self, max_age: float = 0.5) -> np.ndarray:
        with self._state_lock:
            if self._q is None:
                raise RuntimeError("LowState 不可用")
            age = time.monotonic() - self._stamp
            state = self._q.copy()
        if age > max_age:
            raise RuntimeError(f"LowState 超时：{age * 1000:.1f} ms")
        if state.shape != (29,) or not np.isfinite(state).all():
            raise RuntimeError("LowState 关节角无效")
        if self._publisher_error is not None:
            raise RuntimeError(f"LowCmd 发布线程失败：{self._publisher_error}")
        return state

    def hand_state(self, max_age: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        with self._state_lock:
            if self._hand_q is None:
                raise RuntimeError("未收到 rt/inspire/state")
            age = time.monotonic() - self._hand_stamp
            state = self._hand_q.copy()
        if age > max_age:
            raise RuntimeError(f"Inspire 状态超时：{age * 1000:.1f} ms")
        return state[6:12], state[:6]  # left, right

    def enter_low_level(self, measured: np.ndarray) -> None:
        status, result = self._motion_switcher.CheckMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode 失败：{status}, {result}")
        for _ in range(10):
            if not result.get("name"):
                break
            status, result = self._motion_switcher.ReleaseMode()
            if status != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode 失败：{status}, {result}")
            time.sleep(1.0)
            status, result = self._motion_switcher.CheckMode()
        if result.get("name"):
            raise RuntimeError(f"无法释放当前运动服务：{result}")
        with self._command_lock:
            self._left_target = measured[LEFT_ARM].copy()
            self._right_target = measured[RIGHT_ARM].copy()
            self._enabled = True
        self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publisher_thread.start()
        print("[LOW LEVEL] rt/lowcmd 已启用；腿和腰为阻尼，双臂为位置控制")

    def set_arms(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError("双臂目标必须是 14 个有限数值")
        with self._command_lock:
            if self._damping:
                raise RuntimeError("急停后拒绝新的双臂目标")
            self._left_target = value[:7].copy()
            self._right_target = value[7:].copy()

    def set_hands(self, target: np.ndarray) -> None:
        value = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
        if value.shape != (12,) or not np.isfinite(value).all():
            raise ValueError("双手目标必须是 12 个有限数值")
        # Bridge order is right hand first, then left hand.
        ordered = np.concatenate((value[6:12], value[:6]))
        for index, joint in enumerate(ordered):
            self._hand_cmd.cmds[index].q = float(joint)
        self._hand_publisher.Write(self._hand_cmd)

    def latch_damping(self) -> None:
        with self._command_lock:
            self._damping = True

    def _publish_loop(self) -> None:
        while not self._publisher_stop.is_set():
            started = time.monotonic()
            try:
                self._write_lowcmd()
            except Exception as error:
                self._publisher_error = error
                self._publisher_stop.set()
                return
            self._publisher_stop.wait(max(0.0, 0.01 - (time.monotonic() - started)))

    def _write_lowcmd(self) -> None:
        with self._command_lock:
            if not self._enabled or self._left_target is None or self._right_target is None:
                return
            left = self._left_target.copy()
            right = self._right_target.copy()
            damping = self._damping
        with self._state_lock:
            if self._q is None:
                raise RuntimeError("计算重力补偿时 LowState 不可用")
            measured = self._q.copy()
        gravity_tau = (
            np.zeros(14, dtype=np.float64)
            if damping
            else self._gravity.compute(np.concatenate((left, right)), measured)
        )
        self._cmd.level_flag = 0xFF
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = self._mode_machine
        for index in range(29):
            motor = self._cmd.motor_cmd[index]
            motor.mode = 0x01
            motor.dq = 0.0
            motor.tau = 0.0
            if index in LOWER_BODY:
                motor.q = POS_STOP_F
                motor.kp = 0.0
                motor.kd = float(LOWER_BODY_KD[index])
            else:
                arm_index = index - 15
                target = left[arm_index] if arm_index < 7 else right[arm_index - 7]
                motor.q = POS_STOP_F if damping else float(target)
                motor.tau = float(gravity_tau[arm_index])
                motor.kp = 0.0 if damping else float(ARM_KP[arm_index % 7])
                motor.kd = float(ARM_KD[arm_index % 7])
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._publisher.Write(self._cmd)

    def close(self) -> None:
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join(timeout=1.0)


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def episode_path(dataset: Path, info: dict, episode: int) -> Path:
    chunk = episode // int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return dataset / pattern.format(episode_chunk=chunk, episode_index=episode)


def interpolate_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (positions - lower)[:, None]
    return values[lower] * (1.0 - weight) + values[upper] * weight


def rate_limit(start: np.ndarray, targets: np.ndarray, max_step: float) -> tuple[np.ndarray, float]:
    commands = np.empty_like(targets)
    previous = start.copy()
    max_lag = 0.0
    for index, target in enumerate(targets):
        command = previous + np.clip(target - previous, -max_step, max_step)
        commands[index] = command
        max_lag = max(max_lag, float(np.max(np.abs(target - command))))
        previous = command
    return commands, max_lag


def load_commands(
    dataset: Path,
    episode: int,
    speed: float,
    frequency: float,
    max_arm_speed: float,
    max_hand_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    with (dataset / "meta/info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    table = pd.read_parquet(episode_path(dataset, info, episode))
    if table.empty or "action.wbc" not in table or "observation.state" not in table:
        raise ValueError("episode 必须包含非空的 action.wbc 和 observation.state")
    order = "frame_index" if "frame_index" in table else "timestamp"
    table = table.sort_values(order)
    actions = np.stack(table["action.wbc"].map(lambda value: np.asarray(value, dtype=np.float64)))
    states = np.stack(
        table["observation.state"].map(lambda value: np.asarray(value, dtype=np.float64))
    )
    if actions.shape[1] != 41 or states.shape[1] != 41:
        raise ValueError(f"期望 41 维 action/state，实际 {actions.shape}/{states.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("episode 包含 NaN 或 Inf")
    source_fps = float(info.get("fps", 50.0))
    duration = (len(actions) - 1) / source_fps / speed
    count = max(2, int(round(duration * frequency)) + 1)
    source_positions = np.minimum(
        np.arange(count, dtype=np.float64) / frequency * speed * source_fps,
        len(actions) - 1,
    )
    arm_targets = interpolate_rows(actions[:, 15:29], source_positions)
    hand_targets = interpolate_rows(actions[:, 29:41], source_positions)
    initial_arms = states[0, 15:29].copy()
    initial_hands = states[0, 29:41].copy()
    lower = np.concatenate((LEFT_ARM_LOWER, RIGHT_ARM_LOWER))
    upper = np.concatenate((LEFT_ARM_UPPER, RIGHT_ARM_UPPER))
    if np.any(arm_targets < lower) or np.any(arm_targets > upper):
        raise ValueError("轨迹超过 G1 双臂 URDF 限位")
    arm_commands, arm_lag = rate_limit(initial_arms, arm_targets, max_arm_speed / frequency)
    hand_commands, hand_lag = rate_limit(
        initial_hands, np.clip(hand_targets, 0.0, 1.0), max_hand_speed / frequency
    )
    arm_peak_speed = float(np.max(np.abs(np.diff(arm_commands, axis=0))) * frequency)
    hand_peak_speed = float(np.max(np.abs(np.diff(hand_commands, axis=0))) * frequency)
    if arm_peak_speed > max_arm_speed + 1e-9:
        raise ValueError(f"双臂最终命令超过限速：{arm_peak_speed:.6f} rad/s")
    if hand_peak_speed > max_hand_speed + 1e-9:
        raise ValueError(f"双手最终命令超过限速：{hand_peak_speed:.6f}/s")
    return initial_arms, initial_hands, arm_commands, hand_commands, duration, arm_lag, hand_lag


def read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单窗口直控 G1 双臂和 Inspire 手回放")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.25, help="回放倍速，默认 0.25")
    parser.add_argument("--frequency", type=float, default=100.0)
    parser.add_argument("--hand-frequency", type=float, default=5.0)
    parser.add_argument("--max-arm-speed", type=float, default=0.15, help="rad/s")
    parser.add_argument("--max-hand-speed", type=float, default=0.3, help="normalized/s")
    parser.add_argument("--initial-duration", type=float, default=5.0)
    parser.add_argument("--initial-speed", type=float, default=0.1, help="rad/s")
    parser.add_argument("--initial-tolerance", type=float, default=0.1, help="rad")
    parser.add_argument("--initial-hand-tolerance", type=float, default=0.05)
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--left-hand-ip", default="192.168.123.210")
    parser.add_argument("--right-hand-ip", default="192.168.123.211")
    parser.add_argument("--hand-task", default="grab_red_bottle")
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--arm", action="store_true", help="真正连接 DDS 并下发；默认仅离线检查")
    return parser.parse_args()


def start_hand_bridge(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "decoupled_wbc/scripts/inspire_modbus_hand.py"),
        "--mode",
        "dds",
        "--network",
        args.network_interface,
        "--left-ip",
        args.left_hand_ip,
        "--right-ip",
        args.right_hand_ip,
        "--hand-task",
        args.hand_task,
    ]
    return subprocess.Popen(command, cwd=PROJECT_ROOT)


def main() -> int:
    args = parse_args()
    positive = (
        args.speed,
        args.frequency,
        args.hand_frequency,
        args.max_arm_speed,
        args.max_hand_speed,
        args.initial_duration,
        args.initial_speed,
        args.initial_tolerance,
        args.initial_hand_tolerance,
    )
    if args.episode < 0 or min(positive) <= 0.0:
        raise ValueError("episode 必须非负，速度和时间参数必须为正数")
    if not 0.0 <= args.gravity_scale <= 1.0:
        raise ValueError("--gravity-scale 必须在 [0, 1] 内")
    hand_stride = round(args.frequency / args.hand_frequency)
    if args.hand_frequency > args.frequency or not np.isclose(
        hand_stride * args.hand_frequency, args.frequency
    ):
        raise ValueError("--frequency 必须是 --hand-frequency 的整数倍")
    dataset = args.dataset.expanduser().resolve()
    initial_arms, initial_hands, arms, hands, duration, arm_lag, hand_lag = load_commands(
        dataset,
        args.episode,
        args.speed,
        args.frequency,
        args.max_arm_speed,
        args.max_hand_speed,
    )
    print(f"数据集：{dataset}")
    print(f"Episode：{args.episode}；{len(arms)} 个 {args.frequency:g} Hz 下发点")
    print(f"回放时长：{duration:.2f}s；倍速：{args.speed:g}x")
    print(f"双臂限速：{args.max_arm_speed:g}rad/s；最大限速滞后：{arm_lag:.3f}rad")
    print(f"双手限速：{args.max_hand_speed:g}/s；最大限速滞后：{hand_lag:.3f}")
    print(f"首帧双臂 observation.state：{np.round(initial_arms, 4)}")
    if not args.arm:
        print("[DRY RUN] 离线检查通过；未连接 DDS、未发送真机命令。加 --arm 才会实机运行。")
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError("真机模式必须在交互终端运行")

    bridge = start_hand_bridge(args)
    robot: G1ArmDDS | None = None
    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    enabled = False
    phase = "DISARMED"
    init_start_arms = np.zeros(14)
    init_start_hands = np.zeros(12)
    init_started = 0.0
    init_duration = args.initial_duration
    stable_since: float | None = None
    last_init_status = 0.0
    play_index = 0
    period = 1.0 / args.frequency
    last_hands = initial_hands.copy()
    cycle_index = 0
    try:
        robot = G1ArmDDS(args.network_interface, args.gravity_scale)
        print("机器人必须由可靠吊架完全承重。Enter=初始化/播放，Space/Q/Ctrl+C=急停。")
        while not estop.latched:
            cycle_started = time.monotonic()
            if bridge.poll() is not None:
                raise RuntimeError(f"Inspire bridge 已退出，returncode={bridge.returncode}")
            key = read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            measured = robot.state()
            measured_arms = measured[ARMS]
            if key in ("\n", "\r") and phase == "DISARMED":
                left_hand, right_hand = robot.hand_state()
                init_start_hands = np.concatenate((left_hand, right_hand))
                init_start_arms = measured_arms.copy()
                distance = float(np.max(np.abs(initial_arms - init_start_arms)))
                init_duration = max(args.initial_duration, 1.875 * distance / args.initial_speed)
                robot.enter_low_level(measured)
                enabled = True
                init_started = time.monotonic()
                phase = "INITIALIZING"
                print(f"[INITIALIZING] 从实测角 minimum-jerk 移动，预计 {init_duration:.1f}s")

            if phase == "INITIALIZING":
                progress = (time.monotonic() - init_started) / init_duration
                arm_target = minimum_jerk(init_start_arms, initial_arms, progress)
                hand_target = minimum_jerk(init_start_hands, initial_hands, progress)
                robot.set_arms(arm_target)
                if cycle_index % hand_stride == 0:
                    robot.set_hands(hand_target)
                    last_hands = hand_target
                if progress >= 1.0:
                    error = float(np.max(np.abs(initial_arms - measured_arms)))
                    left_hand, right_hand = robot.hand_state()
                    measured_hands = np.concatenate((left_hand, right_hand))
                    hand_error = float(np.max(np.abs(initial_hands - measured_hands)))
                    if (
                        error <= args.initial_tolerance
                        and hand_error <= args.initial_hand_tolerance
                    ):
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= 1.0:
                            phase = "READY"
                            print(
                                f"[READY] 双臂误差 {error:.4f}rad，"
                                f"双手误差 {hand_error:.4f}；检查后按 Enter 播放"
                            )
                    else:
                        stable_since = None
                    now = time.monotonic()
                    if now - last_init_status >= 1.0:
                        print(
                            f"[INITIALIZING] 双臂误差={error:.4f}rad，"
                            f"双手误差={hand_error:.4f}"
                        )
                        last_init_status = now
            elif phase == "READY":
                robot.set_arms(initial_arms)
                if cycle_index % hand_stride == 0:
                    robot.set_hands(initial_hands)
                    last_hands = initial_hands
                if key in ("\n", "\r"):
                    play_index = 0
                    phase = "PLAYING"
                    print("[PLAYING] 开始播放")
            elif phase == "PLAYING":
                robot.set_arms(arms[play_index])
                if cycle_index % hand_stride == 0:
                    robot.set_hands(hands[play_index])
                    last_hands = hands[play_index]
                play_index += 1
                if play_index >= len(arms):
                    phase = "HOLDING"
                    print("[DONE] 播放完成并保持末姿态；Space/Q/Ctrl+C 停止")
            elif phase == "HOLDING":
                robot.set_arms(arms[-1])
                if cycle_index % hand_stride == 0:
                    robot.set_hands(hands[-1])
                    last_hands = hands[-1]
            cycle_index += 1
            time.sleep(max(0.0, period - (time.monotonic() - cycle_started)))
    except Exception as error:
        estop.trigger(str(error))
    finally:
        try:
            if robot is not None and enabled:
                try:
                    measured = robot.state()
                    held = measured[ARMS]
                    for _ in range(20):
                        robot.set_arms(held)
                        robot.set_hands(last_hands)
                        time.sleep(0.01)
                except Exception:
                    robot.latch_damping()
                    time.sleep(1.0)
        finally:
            if robot is not None:
                robot.close()
            bridge.terminate()
            try:
                bridge.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
    return 130 if estop.latched else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(2)
