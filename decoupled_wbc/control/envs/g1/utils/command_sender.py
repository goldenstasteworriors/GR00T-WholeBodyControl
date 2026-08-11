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
    """Use Unitree loco for the legs/waist and ``rt/arm_sdk`` for both arms."""

    ARM_MOTOR_INDICES = tuple(range(15, 29))
    ARM_WEIGHT_INDEX = 29
    LEG_JOINT_COUNT = 12
    BODY_JOINT_COUNT = 29

    def __init__(self, config: Dict):
        import unitree_sdk2py.g1.loco.g1_loco_client as g1_loco_client
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

        self.config = config
        self._system_service_name = str(
            config.get("unitree_loco_system_service_name", "ai_sport")
        )
        self._system_service_start_timeout = float(
            config.get("unitree_loco_service_start_timeout", 15.0)
        )
        self.robot_state = RobotStateClient()
        self.robot_state.SetTimeout(5.0)
        self.robot_state.Init()
        self._ensure_system_service_enabled()
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
        initial_fsm_id = self._wait_for_loco_rpc()
        print(
            f"Unitree {self._system_service_name} service is enabled and healthy "
            f"(fsm_id={initial_fsm_id}); it will remain enabled after this process exits",
            flush=True,
        )
        self.active = False
        self._activation_requested = False
        self._activation_stage = "idle"
        self._activation_deadline = 0.0
        self._stage_started = 0.0
        self._last_fsm_query = 0.0
        self._last_fsm_id = None
        self._last_velocity_send = 0.0
        self._last_start_request = 0.0
        self._latest_body_q = None
        self._latest_leg_dq = None
        self._latest_torso_quat = None
        self._latest_robot_state_time = 0.0
        self._stable_since = None
        self._stand_transition_seen = False
        self._last_wait_log = 0.0
        self._velocity_period = 1.0 / float(config.get("unitree_loco_command_frequency", 10.0))
        start_fsm_id = int(config.get("unitree_loco_start_fsm_id", 501))
        self._start_fsm_id = start_fsm_id if start_fsm_id >= 0 else None
        self._damp_fsm_id = int(config.get("unitree_loco_damp_fsm_id", 1))
        self._stand_fsm_id = int(config.get("unitree_loco_stand_fsm_id", 4))
        self._damp_duration = float(config.get("unitree_loco_damp_duration", 1.0))
        self._stand_duration = float(config.get("unitree_loco_stand_duration", 5.0))
        self._start_retry_interval = float(
            config.get("unitree_loco_start_retry_interval", 1.0)
        )
        self._stability_duration = float(config.get("unitree_loco_stability_duration", 0.5))
        self._activation_timeout = float(config.get("unitree_loco_activation_timeout", 25.0))
        self._state_timeout = float(config.get("unitree_loco_state_timeout", 0.5))
        self._max_leg_velocity = float(config.get("unitree_loco_max_leg_velocity", 0.35))
        self._max_torso_tilt = float(config.get("unitree_loco_max_torso_tilt", 0.35))
        self._max_linear_velocity = float(config.get("unitree_loco_max_linear_velocity", 0.05))
        self._max_angular_velocity = float(config.get("unitree_loco_max_angular_velocity", 0.1))
        self._navigation_enabled = bool(config.get("unitree_loco_navigation_enabled", False))
        self._arm_control_enabled = bool(
            config.get("unitree_loco_arm_control_enabled", False)
        )
        self._weight_ramp_duration = float(config.get("unitree_arm_weight_ramp_duration", 2.0))
        self._activation_time = 0.0
        self._arm_preparing = False
        self._arm_preparation_started = 0.0
        self._arm_preparation_complete = False

    @staticmethod
    def _check_rpc(name: str, code) -> None:
        if code not in (None, 0):
            raise RuntimeError(f"Unitree loco {name} failed with code {code}")

    @staticmethod
    def _check_robot_state_rpc(name: str, code) -> None:
        if code not in (None, 0):
            raise RuntimeError(f"Unitree robot_state {name} failed with code {code}")

    def _get_system_service_status(self) -> int:
        code, services = self.robot_state.ServiceList()
        self._check_robot_state_rpc("ServiceList", code)
        for service in services or ():
            if service.name == self._system_service_name:
                return int(service.status)
        available = ", ".join(service.name for service in services or ()) or "<none>"
        raise RuntimeError(
            f"Unitree system service {self._system_service_name!r} was not found; "
            f"available services: {available}"
        )

    def _ensure_system_service_enabled(self) -> None:
        """Start ai_sport when needed and deliberately never stop it."""
        status = self._get_system_service_status()
        # This firmware reports 0 for a running service and 1 for a stopped
        # service.  The values are service-manager states, not booleans.
        if status == 0:
            print(
                f"Unitree {self._system_service_name} service is already enabled; "
                "leaving it running",
                flush=True,
            )
            return
        if status != 1:
            raise RuntimeError(
                f"Unitree system service {self._system_service_name!r} has "
                f"unsupported status {status}"
            )

        self._check_robot_state_rpc(
            f'ServiceSwitch("{self._system_service_name}", True)',
            self.robot_state.ServiceSwitch(self._system_service_name, True),
        )
        deadline = time.monotonic() + self._system_service_start_timeout
        while True:
            status = self._get_system_service_status()
            if status == 0:
                print(
                    f"Unitree {self._system_service_name} service enabled; "
                    "it will remain enabled after this process exits",
                    flush=True,
                )
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out after {self._system_service_start_timeout:.1f}s waiting "
                    f"for Unitree {self._system_service_name} service to become enabled "
                    f"(last status={status}); the service was not stopped"
                )
            time.sleep(0.2)

    def _wait_for_loco_rpc(self) -> int:
        deadline = time.monotonic() + self._system_service_start_timeout
        last_error = None
        while True:
            try:
                return self._query_fsm_id()
            except Exception as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Unitree {self._system_service_name} reports enabled, but the "
                    f"{self.config.get('unitree_loco_service_name', 'sport')!r} loco RPC "
                    f"did not become healthy within {self._system_service_start_timeout:.1f}s: "
                    f"{last_error}. The service was left enabled; no FSM command was sent."
                ) from last_error
            time.sleep(0.2)

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

    def update_robot_state(
        self,
        body_q: np.ndarray,
        body_dq: np.ndarray,
        torso_quat: np.ndarray,
    ) -> None:
        """Cache measured motion used to confirm that standing has settled."""
        body_q = np.asarray(body_q, dtype=np.float64)
        body_dq = np.asarray(body_dq, dtype=np.float64)
        torso_quat = np.asarray(torso_quat, dtype=np.float64)
        if body_q.ndim != 1 or body_q.size < self.BODY_JOINT_COUNT:
            return
        if body_dq.ndim != 1 or body_dq.size < self.LEG_JOINT_COUNT:
            return
        if torso_quat.shape != (4,):
            return
        if (
            not np.isfinite(body_q).all()
            or not np.isfinite(body_dq).all()
            or not np.isfinite(torso_quat).all()
        ):
            return
        self._latest_body_q = body_q[: self.BODY_JOINT_COUNT].copy()
        self._latest_leg_dq = body_dq[: self.LEG_JOINT_COUNT].copy()
        self._latest_torso_quat = torso_quat.copy()
        self._latest_robot_state_time = time.monotonic()

    def _robot_stability(
        self, now: float, *, require_settled_legs: bool = True
    ) -> tuple[bool, str]:
        if self._latest_leg_dq is None or self._latest_torso_quat is None:
            return False, "waiting for measured robot state"
        state_age = now - self._latest_robot_state_time
        if state_age > self._state_timeout:
            return False, f"robot state is stale ({state_age:.2f}s)"

        if require_settled_legs:
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

    def _stable_for_required_duration(
        self, now: float, *, require_settled_legs: bool = True
    ) -> tuple[bool, str]:
        stable, reason = self._robot_stability(
            now, require_settled_legs=require_settled_legs
        )
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
        self._last_start_request = 0.0
        self._arm_preparing = False
        self._arm_preparation_started = 0.0
        self._arm_preparation_complete = False

    def operator_ready(self) -> bool:
        """Return true after locomotion and the optional arm preparation are ready."""
        if not self.active:
            return False
        return not self._arm_control_enabled or self._arm_preparation_complete

    def _arm_output_active(self) -> bool:
        return self._arm_control_enabled and (
            self._arm_preparing or self._arm_preparation_complete or self.active
        )

    def _start_arm_preparation(self, now: float) -> None:
        if self._latest_body_q is None:
            raise RuntimeError(
                "cannot initialize arm_sdk before measured 29-DOF state is available"
            )
        # Match xr_teleoperate's motion-mode initialization: seed the complete
        # arm_sdk packet from the current 29-DOF pose once, then update only the
        # dual-arm slots (15-28) in send_command(). This avoids zero/default
        # commands in the non-arm slots while leaving their live targets to the
        # official motion controller's topic arbitration.
        self.low_cmd.mode_pr = int(
            self.config["UNITREE_LEGGED_CONST"].get("MODE_PR", 0)
        )
        self.low_cmd.mode_machine = int(
            self.config["UNITREE_LEGGED_CONST"].get("MODE_MACHINE", 0)
        )
        for motor_index in range(self.BODY_JOINT_COUNT):
            joint_index = self.config["MOTOR2JOINT"][motor_index]
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.mode = 1
            motor.q = float(self._latest_body_q[joint_index])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(self.config["MOTOR_KP"][motor_index])
            motor.kd = float(self.config["MOTOR_KD"][motor_index])
        self._arm_preparing = True
        self._arm_preparation_started = now
        self._arm_preparation_complete = False
        self._set_activation_stage("prepare_arms", now)
        source = (
            f"FSM {self._stand_fsm_id} locked standing"
            if self._start_fsm_id is None
            else f"FSM {self._start_fsm_id} locomotion"
        )
        print(
            f"Unitree {source} confirmed; blending arm_sdk for motors 15-28 "
            f"over {self._weight_ramp_duration:.1f}s; the 29-DOF packet was seeded "
            "from measured pose and only dual-arm targets will change",
            flush=True,
        )

    def _mark_active(self, now: float, fsm_id: int) -> None:
        self.active = True
        self._activation_requested = False
        self._activation_stage = "active"
        self._activation_time = now
        print(
            f"Unitree official loco confirmed standing and active (fsm_id={fsm_id})",
            flush=True,
        )

    def _finish_standing_setup(self, now: float) -> None:
        if self._start_fsm_id is None:
            self._set_activation_stage("wait_ready", now)
            print(
                "Unitree official loco locked standing confirmed; no velocity RPC sent",
                flush=True,
            )
            return
        self._set_fsm(self._start_fsm_id, "Start locomotion")
        self._last_start_request = now
        self._set_activation_stage("wait_locomotion", now)
        print(
            "Unitree official loco Start requested "
            f"(fsm_id={self._start_fsm_id})"
        )

    def _safe_stop(self, request_damp: bool) -> list[str]:
        errors = []
        try:
            self._release_arms()
        except Exception as exc:
            errors.append(f"release arms: {exc}")
        if request_damp:
            try:
                self._set_fsm(self._damp_fsm_id, "Damp")
            except Exception as exc:
                errors.append(f"Damp: {exc}")
                try:
                    self._stop_move()
                except Exception as stop_exc:
                    errors.append(f"fallback zero velocity: {stop_exc}")
        else:
            try:
                self._stop_move()
            except Exception as exc:
                errors.append(f"zero velocity: {exc}")
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
            self._last_wait_log = 0.0
            self._last_fsm_query = 0.0
            self._last_fsm_id = None
            self._last_start_request = 0.0
            self._arm_preparing = False
            self._arm_preparation_started = 0.0
            self._arm_preparation_complete = False
            try:
                self._release_arms()
                self._set_fsm(self._damp_fsm_id, "Damp")
            except Exception as exc:
                errors = self._safe_stop(request_damp=True)
                if errors:
                    raise RuntimeError(
                        f"Unitree loco activation failed: {exc}; "
                        "safe-stop errors: " + ", ".join(errors)
                    ) from exc
                raise
            self._set_activation_stage("wait_damp", now)
            startup_path = f"Damp({self._damp_fsm_id}) -> StandUp({self._stand_fsm_id})"
            if self._start_fsm_id is None:
                startup_path += " -> locked standing"
            else:
                startup_path += f" -> Start({self._start_fsm_id})"
            if self._arm_control_enabled:
                startup_path += " -> prepare arms"
            print(
                f"Unitree official loco startup requested: {startup_path}",
                flush=True,
            )
            return

        errors = self._safe_stop(request_damp=emergency)
        if emergency:
            print(
                "Unitree official loco emergency stop: arms released, "
                "Damp requested, no Move sent"
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
            # A transition away from passive/zero-torque states confirms that
            # the official firmware accepted the StandUp request.
            if fsm_id not in {self._damp_fsm_id, 0}:
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
            if fsm_id != self._stand_fsm_id:
                self._log_waiting(
                    now,
                    f"waiting for locked standing fsm_id={self._stand_fsm_id}, current={fsm_id}",
                )
                return
            stable, reason = self._stable_for_required_duration(now)
            if not stable:
                self._log_waiting(now, reason)
                return
            if self._start_fsm_id is None and self._arm_control_enabled:
                self._start_arm_preparation(now)
            else:
                self._finish_standing_setup(now)
            return

        if self._activation_stage == "prepare_arms":
            expected_fsm_id = (
                self._stand_fsm_id
                if self._start_fsm_id is None
                else self._start_fsm_id
            )
            if fsm_id != expected_fsm_id:
                self._stable_since = None
                self._log_waiting(
                    now,
                    f"holding for arm preparation in fsm_id={expected_fsm_id}, current={fsm_id}",
                )
                return
            if self._start_fsm_id is not None and (
                now - self._last_velocity_send >= self._velocity_period
            ):
                self._stop_move()
            preparation_time = now - self._arm_preparation_started
            if preparation_time < self._weight_ramp_duration:
                self._stable_since = None
                self._log_waiting(
                    now,
                    "preparing arms "
                    f"({preparation_time:.2f}/{self._weight_ramp_duration:.2f}s)",
                )
                return
            stable, reason = self._stable_for_required_duration(
                now,
                require_settled_legs=self._start_fsm_id is None,
            )
            if not stable:
                self._log_waiting(now, f"arms prepared; waiting for lower body: {reason}")
                return
            self._arm_preparing = False
            self._arm_preparation_complete = True
            print(
                "Unitree arm_sdk dual-arm preparation pose ready; waist is not updated by IK",
                flush=True,
            )
            self._mark_active(now, fsm_id)
            return

        if self._activation_stage == "wait_ready":
            if fsm_id in {self._damp_fsm_id, 0}:
                self._stable_since = None
                self._log_waiting(now, f"waiting for standing mode, current fsm_id={fsm_id}")
                return
            stable, reason = self._stable_for_required_duration(now)
            if not stable:
                self._log_waiting(now, reason)
                return
            self._mark_active(now, fsm_id)
            return

        if self._activation_stage == "wait_locomotion":
            if fsm_id != self._start_fsm_id:
                self._stable_since = None
                if now - self._last_start_request >= self._start_retry_interval:
                    self._set_fsm(self._start_fsm_id, "Retry start locomotion")
                    self._last_start_request = now
                self._log_waiting(
                    now, f"waiting for locomotion fsm_id={self._start_fsm_id}, current={fsm_id}"
                )
                return
            if now - self._last_velocity_send >= self._velocity_period:
                self._stop_move()
            stable, reason = self._stable_for_required_duration(
                now, require_settled_legs=False
            )
            if not stable:
                self._log_waiting(now, reason)
                return
            if self._arm_control_enabled:
                self._start_arm_preparation(now)
            else:
                self._mark_active(now, fsm_id)

    def send_velocity(self, navigate_cmd) -> None:
        if not self.active or self._start_fsm_id is None:
            return
        now = time.monotonic()
        if now - self._last_velocity_send < self._velocity_period:
            return
        velocity = np.asarray(navigate_cmd, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("navigate_cmd must contain three finite values")
        if not self._navigation_enabled:
            velocity = np.zeros(3, dtype=np.float64)
        vx = float(np.clip(velocity[0], -self._max_linear_velocity, self._max_linear_velocity))
        vy = float(np.clip(velocity[1], -self._max_linear_velocity, self._max_linear_velocity))
        wz = float(np.clip(velocity[2], -self._max_angular_velocity, self._max_angular_velocity))
        self._check_rpc("SetVelocity", self.loco.SetVelocity(vx, vy, wz, 0.25))
        self._last_velocity_send = now

    def send_command(self, cmd_q: np.ndarray, cmd_dq: np.ndarray, cmd_tau: np.ndarray):
        if not self._arm_output_active():
            return
        for motor_index in self.ARM_MOTOR_INDICES:
            joint_index = self.config["MOTOR2JOINT"][motor_index]
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(cmd_q[joint_index])
            motor.dq = float(cmd_dq[joint_index])
            motor.tau = float(cmd_tau[joint_index])
            motor.kp = float(self.config["MOTOR_KP"][motor_index])
            motor.kd = float(self.config["MOTOR_KD"][motor_index])
        if self._arm_preparation_complete or self._weight_ramp_duration <= 0.0:
            weight = 1.0
        else:
            weight = np.clip(
                (time.monotonic() - self._arm_preparation_started)
                / self._weight_ramp_duration,
                0.0,
                1.0,
            )
        self.low_cmd.motor_cmd[self.ARM_WEIGHT_INDEX].q = float(weight)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def close(self) -> None:
        self.set_active(False, emergency=True)


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
