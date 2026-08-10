from typing import Dict

import numpy as np
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
INSPIRE_OPEN_Q = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.2], dtype=np.float64)
INSPIRE_GRASP_Q = np.array([0.15, 0.15, 0.15, 0.15, 1.0, 0.2], dtype=np.float64)


class InspireHandCommandSender:
    """Publish RH56DFTP Inspire hand commands on Unitree's shared DDS topic."""

    def __init__(self, is_left: bool = True):
        self.is_left = is_left
        self.cmd_pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self.cmd_pub.Init()
        self.cmd = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        self._last_left_q = INSPIRE_OPEN_Q.copy()
        self._last_right_q = INSPIRE_OPEN_Q.copy()

    def send_command(self, cmd: np.ndarray):
        q = np.asarray(cmd, dtype=np.float64)
        if q.shape[0] == INSPIRE_LEGACY_HAND_DOF:
            q = self.legacy_dex3_to_inspire(q)
        elif q.shape[0] != INSPIRE_HAND_DOF:
            raise ValueError(f"Inspire hand command must have 6 or 7 values, got {q.shape[0]}")

        q = np.clip(q, 0.0, 1.0)
        if self.is_left:
            self._last_left_q = q.copy()
        else:
            self._last_right_q = q.copy()

        left_q = self._last_left_q
        right_q = self._last_right_q
        for i, value in enumerate(right_q):
            self.cmd.cmds[i].q = float(value)
        for i, value in enumerate(left_q):
            self.cmd.cmds[i + INSPIRE_HAND_DOF].q = float(value)

        self.cmd_pub.Write(self.cmd)

    @staticmethod
    def legacy_dex3_to_inspire(cmd: np.ndarray) -> np.ndarray:
        """Map the existing 7-DOF Dex3 command shape to binary Inspire open/grasp."""
        grasp = np.max(np.abs(cmd)) > 0.05
        return INSPIRE_GRASP_Q.copy() if grasp else INSPIRE_OPEN_Q.copy()
