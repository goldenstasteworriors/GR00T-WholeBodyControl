import json
import os
import threading
import time
from typing import Dict

import numpy as np
from gear_sonic.utils.data_collection.inspire_hand_tasks import (
    DEFAULT_HAND_TASK,
    resolve_hand_task_pose,
)
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.utils.crc import CRC


class BodyCommandSender:
    def __init__(self, config: Dict):
        self.config = config
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif (
            self.config["ROBOT_TYPE"] == "g1_29dof"
            or self.config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(
                f"Robot type {self.config['ROBOT_TYPE']} is not supported yet"
            )
        # init kp kd
        self.kp_level = 1.0
        self.waist_kp_level = 1.0
        self.robot_kp = np.zeros(self.config["NUM_MOTORS"])
        self.robot_kd = np.zeros(self.config["NUM_MOTORS"])
        # set kp level
        for i in range(len(self.config["MOTOR_KP"])):
            self.robot_kp[i] = self.config["MOTOR_KP"][i] * self.kp_level
        for i in range(len(self.config["MOTOR_KD"])):
            self.robot_kd[i] = self.config["MOTOR_KD"][i] * 1.0
        self.weak_motor_joint_index = []
        for _, value in self.config["WeakMotorJointIndex"].items():
            self.weak_motor_joint_index.append(value)
        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

    def InitLowCmd(self):
        # h1/go2:
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(self.config["NUM_MOTORS"]):
            if self.is_weak_motor(i):
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = self.config["UNITREE_LEGGED_CONST"]["PosStopF"]
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = self.config["UNITREE_LEGGED_CONST"]["VelStopF"]
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
            if (
                self.config["ROBOT_TYPE"] == "g1_29dof"
                or self.config["ROBOT_TYPE"] == "h1-2_21dof"
                or self.config["ROBOT_TYPE"] == "h1-2_27dof"
            ):
                self.low_cmd.mode_machine = self.config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"]
                self.low_cmd.mode_pr = self.config["UNITREE_LEGGED_CONST"]["MODE_PR"]
            else:
                pass

    def is_weak_motor(self, motor_index: int) -> bool:
        return motor_index in self.weak_motor_joint_index

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        for i in range(self.config["NUM_MOTORS"]):
            motor_index = self.config["JOINT2MOTOR"][i]
            joint_index = self.config["MOTOR2JOINT"][i]
            # print(f"motor_index: {motor_index}, joint_index: {joint_index}")
            if joint_index == -1:
                # send default joint position command
                self.low_cmd.motor_cmd[motor_index].q = self.config["DEFAULT_MOTOR_ANGLES"][
                    motor_index
                ]
                self.low_cmd.motor_cmd[motor_index].dq = 0.0
                self.low_cmd.motor_cmd[motor_index].tau = 0.0
            else:
                self.low_cmd.motor_cmd[motor_index].q = cmd_q[joint_index]
                self.low_cmd.motor_cmd[motor_index].dq = cmd_dq[joint_index]
                self.low_cmd.motor_cmd[motor_index].tau = cmd_tau[joint_index]
            # kp kd
            self.low_cmd.motor_cmd[motor_index].kp = self.robot_kp[motor_index]
            self.low_cmd.motor_cmd[motor_index].kd = self.robot_kd[motor_index]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


class UnitreeLocoArmCommandSender:
    """Use Unitree loco for the lower body and ``rt/arm_sdk`` for both arms."""

    ARM_MOTOR_INDICES = tuple(range(15, 29))
    ARM_WEIGHT_INDEX = 29
    LEG_JOINT_COUNT = 12

    def __init__(self, config: Dict):
        import unitree_sdk2py.g1.loco.g1_loco_client as g1_loco_client
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

        self.config = config
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.publisher.Init()
        # ai_sport >= 8.2 renamed this RPC service from ``loco`` to ``sport``.
        # Keep it configurable so robots on older firmware can still opt in to
        # the legacy name without modifying the vendored Unitree SDK.
        g1_loco_client.LOCO_SERVICE_NAME = str(
            config.get("unitree_loco_service_name", "sport")
        )
        self.loco = g1_loco_client.LocoClient()
        self.loco.SetTimeout(5.0)
        self.loco.Init()
        self.active = False
        self._activation_requested = False
        self._activation_stage = "idle"
        self._activation_deadline = 0.0
        self._stage_started = 0.0
        self._last_fsm_query = 0.0
        self._last_fsm_id = None
        self._last_velocity_send = 0.0
        self._latest_leg_dq = None
        self._latest_torso_quat = None
        self._latest_robot_state_time = 0.0
        self._stable_since = None
        self._stand_transition_seen = False
        self._locomotion_zeroed = False
        self._last_wait_log = 0.0
        self._velocity_period = 1.0 / float(config.get("unitree_loco_command_frequency", 10.0))
        self._start_fsm_id = int(config.get("unitree_loco_start_fsm_id", 501))
        self._damp_fsm_id = int(config.get("unitree_loco_damp_fsm_id", 1))
        self._stand_fsm_id = int(config.get("unitree_loco_stand_fsm_id", 4))
        self._damp_duration = float(config.get("unitree_loco_damp_duration", 0.5))
        self._stand_duration = float(config.get("unitree_loco_stand_duration", 4.0))
        self._stability_duration = float(config.get("unitree_loco_stability_duration", 0.5))
        self._activation_timeout = float(config.get("unitree_loco_activation_timeout", 15.0))
        self._state_timeout = float(config.get("unitree_loco_state_timeout", 0.5))
        self._max_leg_velocity = float(config.get("unitree_loco_max_leg_velocity", 0.35))
        self._max_torso_tilt = float(config.get("unitree_loco_max_torso_tilt", 0.35))
        self._max_linear_velocity = float(config.get("unitree_loco_max_linear_velocity", 0.5))
        self._max_angular_velocity = float(config.get("unitree_loco_max_angular_velocity", 1.0))
        self._weight_ramp_duration = float(config.get("unitree_arm_weight_ramp_duration", 2.0))
        self._activation_time = 0.0

    @staticmethod
    def _check_rpc(name: str, code) -> None:
        if code not in (None, 0):
            raise RuntimeError(f"Unitree loco {name} failed with code {code}")

    def _set_fsm(self, fsm_id: int, name: str) -> None:
        self._check_rpc(name, self.loco.SetFsmId(int(fsm_id)))

    def _stop_move(self) -> None:
        # Call SetVelocity directly because older Unitree Python SDK versions
        # discard the return value from StopMove().
        self._check_rpc("SetVelocity(0)", self.loco.SetVelocity(0.0, 0.0, 0.0, 0.25))
        self._last_velocity_send = time.monotonic()

    def _query_fsm_id(self) -> int:
        code, data = self.loco._Call(7001, "")
        self._check_rpc("GetFsmId", code)
        response = json.loads(data) if isinstance(data, str) else data
        if not isinstance(response, dict) or "data" not in response:
            raise RuntimeError(f"Unitree loco GetFsmId returned invalid data: {data!r}")
        return int(response["data"])

    def update_robot_state(self, body_dq: np.ndarray, torso_quat: np.ndarray) -> None:
        """Cache measured motion used to confirm that standing has settled."""
        body_dq = np.asarray(body_dq, dtype=np.float64)
        torso_quat = np.asarray(torso_quat, dtype=np.float64)
        if body_dq.ndim != 1 or body_dq.size < self.LEG_JOINT_COUNT:
            return
        if torso_quat.shape != (4,):
            return
        if not np.isfinite(body_dq).all() or not np.isfinite(torso_quat).all():
            return
        self._latest_leg_dq = body_dq[: self.LEG_JOINT_COUNT].copy()
        self._latest_torso_quat = torso_quat.copy()
        self._latest_robot_state_time = time.monotonic()

    def _robot_stability(self, now: float) -> tuple[bool, str]:
        if self._latest_leg_dq is None or self._latest_torso_quat is None:
            return False, "waiting for measured robot state"
        state_age = now - self._latest_robot_state_time
        if state_age > self._state_timeout:
            return False, f"robot state is stale ({state_age:.2f}s)"

        max_leg_velocity = float(np.max(np.abs(self._latest_leg_dq)))
        if max_leg_velocity > self._max_leg_velocity:
            return False, f"legs still moving ({max_leg_velocity:.2f} rad/s)"

        quat = self._latest_torso_quat
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-6:
            return False, "torso orientation is unavailable"
        normalized_quat = quat / quat_norm
        x = normalized_quat[1]
        y = normalized_quat[2]
        upright_cosine = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
        torso_tilt = float(np.arccos(upright_cosine))
        if torso_tilt > self._max_torso_tilt:
            return False, f"torso is not upright ({torso_tilt:.2f} rad)"
        return True, "stable"

    def _stable_for_required_duration(self, now: float) -> tuple[bool, str]:
        stable, reason = self._robot_stability(now)
        if not stable:
            self._stable_since = None
            return False, reason
        if self._stable_since is None:
            self._stable_since = now
        stable_time = now - self._stable_since
        if stable_time < self._stability_duration:
            return False, f"settling ({stable_time:.2f}/{self._stability_duration:.2f}s)"
        return True, "stable"

    def _set_activation_stage(self, stage: str, now: float) -> None:
        self._activation_stage = stage
        self._stage_started = now
        self._stable_since = None

    def _release_arms(self) -> None:
        self.low_cmd.motor_cmd[self.ARM_WEIGHT_INDEX].q = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def _reset_activation(self) -> None:
        self.active = False
        self._activation_requested = False
        self._activation_stage = "idle"
        self._activation_deadline = 0.0
        self._stable_since = None
        self._stand_transition_seen = False
        self._locomotion_zeroed = False

    def _safe_stop(self, request_damp: bool) -> list[str]:
        errors = []
        try:
            self._release_arms()
        except Exception as exc:
            errors.append(f"release arms: {exc}")
        try:
            self._stop_move()
        except Exception as exc:
            errors.append(f"zero velocity: {exc}")
        if request_damp:
            try:
                self._set_fsm(self._damp_fsm_id, "Damp")
            except Exception as exc:
                errors.append(f"Damp: {exc}")
        self._reset_activation()
        return errors

    def _fail_activation(self, message: str) -> None:
        errors = self._safe_stop(request_damp=True)
        if errors:
            message += "; safe-stop errors: " + ", ".join(errors)
        raise RuntimeError(message)

    def _log_waiting(self, now: float, reason: str) -> None:
        if now - self._last_wait_log >= 1.0:
            print(
                f"Unitree official loco startup [{self._activation_stage}]: {reason}",
                flush=True,
            )
            self._last_wait_log = now

    def set_active(self, active: bool, emergency: bool = False) -> None:
        active = bool(active)
        if active and (self.active or self._activation_requested):
            return
        if not active and not (self.active or self._activation_requested or emergency):
            return
        if active:
            now = time.monotonic()
            self.active = False
            self._activation_requested = True
            self._activation_deadline = now + self._activation_timeout
            self._stand_transition_seen = False
            self._locomotion_zeroed = False
            self._last_wait_log = 0.0
            self._last_fsm_query = 0.0
            self._last_fsm_id = None
            self._release_arms()
            self._set_fsm(self._damp_fsm_id, "Damp")
            self._set_activation_stage("wait_damp", now)
            print(
                "Unitree official loco startup requested: "
                f"Damp({self._damp_fsm_id}) -> StandUp({self._stand_fsm_id}) "
                f"-> locomotion({self._start_fsm_id})",
                flush=True,
            )
            return

        errors = self._safe_stop(request_damp=emergency)
        if emergency:
            print(
                "Unitree official loco emergency stop: arms released, "
                "velocity zero, damp requested"
            )
        if errors:
            raise RuntimeError("Unitree loco safe stop failed: " + ", ".join(errors))

    def update_status(self) -> None:
        if not self._activation_requested or self.active:
            return
        now = time.monotonic()
        if now > self._activation_deadline:
            self._fail_activation(
                "Unitree loco startup timed out during "
                f"{self._activation_stage} after {self._activation_timeout:.1f} seconds"
            )

        if now - self._last_fsm_query >= 0.2:
            self._last_fsm_id = self._query_fsm_id()
            self._last_fsm_query = now

        fsm_id = self._last_fsm_id
        if self._activation_stage == "wait_damp":
            if fsm_id != self._damp_fsm_id:
                self._log_waiting(now, f"waiting for Damp, current fsm_id={fsm_id}")
                return
            damp_time = now - self._stage_started
            if damp_time < self._damp_duration:
                self._log_waiting(
                    now, f"Damp hold ({damp_time:.2f}/{self._damp_duration:.2f}s)"
                )
                return
            self._set_fsm(self._stand_fsm_id, "StandUp")
            self._set_activation_stage("wait_stand", now)
            print(f"Unitree official loco StandUp requested (fsm_id={self._stand_fsm_id})")
            return

        if self._activation_stage == "wait_stand":
            if fsm_id in {self._stand_fsm_id, 500, self._start_fsm_id}:
                self._stand_transition_seen = True
            stand_time = now - self._stage_started
            if stand_time < self._stand_duration:
                self._log_waiting(
                    now,
                    f"standing ({stand_time:.2f}/{self._stand_duration:.2f}s), "
                    f"fsm_id={fsm_id}",
                )
                return
            if not self._stand_transition_seen:
                self._log_waiting(now, f"StandUp transition not observed, current fsm_id={fsm_id}")
                return
            stable, reason = self._stable_for_required_duration(now)
            if not stable:
                self._log_waiting(now, reason)
                return
            self._set_fsm(self._start_fsm_id, "Start locomotion")
            self._set_activation_stage("wait_locomotion", now)
            print(f"Unitree official loco locomotion requested (fsm_id={self._start_fsm_id})")
            return

        if self._activation_stage == "wait_locomotion":
            if fsm_id != self._start_fsm_id:
                self._stable_since = None
                self._log_waiting(
                    now, f"waiting for locomotion fsm_id={self._start_fsm_id}, current={fsm_id}"
                )
                return
            if not self._locomotion_zeroed:
                self._stop_move()
                self._locomotion_zeroed = True
                self._stable_since = None
                return
            stable, reason = self._stable_for_required_duration(now)
            if not stable:
                self._log_waiting(now, reason)
                return
            self.active = True
            self._activation_requested = False
            self._activation_stage = "active"
            self._activation_time = now
            print(
                f"Unitree official loco confirmed standing and active (fsm_id={fsm_id})",
                flush=True,
            )

    def send_velocity(self, navigate_cmd) -> None:
        if not self.active:
            return
        now = time.monotonic()
        if now - self._last_velocity_send < self._velocity_period:
            return
        velocity = np.asarray(navigate_cmd, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("navigate_cmd must contain three finite values")
        vx = float(np.clip(velocity[0], -self._max_linear_velocity, self._max_linear_velocity))
        vy = float(np.clip(velocity[1], -self._max_linear_velocity, self._max_linear_velocity))
        wz = float(np.clip(velocity[2], -self._max_angular_velocity, self._max_angular_velocity))
        self._check_rpc("SetVelocity", self.loco.SetVelocity(vx, vy, wz, 0.25))
        self._last_velocity_send = now

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        if not self.active:
            return
        for motor_index in self.ARM_MOTOR_INDICES:
            joint_index = self.config["MOTOR2JOINT"][motor_index]
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(cmd_q[joint_index])
            motor.dq = float(cmd_dq[joint_index])
            motor.tau = float(cmd_tau[joint_index])
            motor.kp = float(self.config["MOTOR_KP"][motor_index])
            motor.kd = float(self.config["MOTOR_KD"][motor_index])
        if self._weight_ramp_duration <= 0.0:
            weight = 1.0
        else:
            weight = np.clip(
                (time.monotonic() - self._activation_time) / self._weight_ramp_duration,
                0.0,
                1.0,
            )
        self.low_cmd.motor_cmd[self.ARM_WEIGHT_INDEX].q = float(weight)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def close(self) -> None:
        self.set_active(False, emergency=False)


def make_hand_mode(motor_index: int) -> int:
    status = 0x01
    timeout = 0x01
    mode = motor_index & 0x0F
    mode |= status << 4  # bits [4..6]
    mode |= timeout << 7  # bit 7
    return mode


class HandCommandSender:
    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        if self.is_left:
            self.cmd_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        else:
            self.cmd_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)

        self.cmd_pub.Init()
        self.cmd = unitree_hg_msg_dds__HandCmd_()

        self.hand_dof = 7

        self.kp = [1.0] * self.hand_dof
        self.kd = [0.2] * self.hand_dof
        self.kp[0] = 2.0
        self.kd[0] = 0.5

    def send_command(self, cmd: np.ndarray):
        for i in range(self.hand_dof):
            # Build the bitfield mode (see your C++ example)
            mode_val = make_hand_mode(i)
            self.cmd.motor_cmd[i].mode = mode_val
            self.cmd.motor_cmd[i].q = cmd[i]
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
            self.cmd.motor_cmd[i].kp = self.kp[i]
            self.cmd.motor_cmd[i].kd = self.kd[i]

        self.cmd_pub.Write(self.cmd)


INSPIRE_HAND_DOF = 6
INSPIRE_LEGACY_HAND_DOF = 7

# Inspire DDS order:
# [little, ring, middle, index, thumb_bend, thumb_rotate], 0 = closed, 1 = open.
INSPIRE_OPEN_Q = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)
INSPIRE_GRASP_Q = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float64)


class InspireHandCommandSender:
    """Publish RH56DFTP Inspire hand commands on Unitree's shared DDS topic."""

    _shared_lock = threading.Lock()
    _shared_hand_task: str | None = None
    _shared_left_q: np.ndarray | None = None
    _shared_right_q: np.ndarray | None = None

    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        self.hand_task = os.environ.get("SONIC_HAND_TASK", DEFAULT_HAND_TASK)
        self.cmd_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_pub.Init()
        self.cmd = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        task_open_q = np.asarray(resolve_hand_task_pose(self.hand_task, pressed=False), dtype=np.float64)
        with self._shared_lock:
            if (
                self.__class__._shared_hand_task != self.hand_task
                or self.__class__._shared_left_q is None
                or self.__class__._shared_right_q is None
            ):
                self.__class__._shared_hand_task = self.hand_task
                self.__class__._shared_left_q = task_open_q.copy()
                self.__class__._shared_right_q = task_open_q.copy()

    def send_command(self, cmd: np.ndarray):
        q = np.asarray(cmd, dtype=np.float64)
        if q.shape[0] == INSPIRE_LEGACY_HAND_DOF:
            q = self.legacy_dex3_to_inspire(q, self.hand_task)
        elif q.shape[0] != INSPIRE_HAND_DOF:
            raise ValueError(f"Inspire hand command must have 6 or 7 values, got {q.shape[0]}")

        q = np.clip(q, 0.0, 1.0)
        with self._shared_lock:
            if self.is_left:
                self.__class__._shared_left_q = q.copy()
            else:
                self.__class__._shared_right_q = q.copy()

            left_q = self.__class__._shared_left_q
            right_q = self.__class__._shared_right_q
            assert left_q is not None and right_q is not None
            for i, value in enumerate(right_q):
                self.cmd.cmds[i].q = float(value)
            for i, value in enumerate(left_q):
                self.cmd.cmds[i + INSPIRE_HAND_DOF].q = float(value)

            self.cmd_pub.Write(self.cmd)

    @staticmethod
    def legacy_dex3_to_inspire(cmd: np.ndarray, hand_task: str = "pick_up_pipette") -> np.ndarray:
        """Map the existing 7-DOF Dex3 command shape to binary Inspire open/grasp."""
        grasp = np.max(np.abs(cmd)) > 0.05
        if not grasp:
            return np.asarray(resolve_hand_task_pose(hand_task, pressed=False), dtype=np.float64)
        return np.asarray(resolve_hand_task_pose(hand_task, pressed=True), dtype=np.float64)
