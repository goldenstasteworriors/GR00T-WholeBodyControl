"""Dex1 hardware I/O; seven-joint placeholders are only for existing wrist FK.

Native state/command angles are logged separately in radians, one per hand.
DDS topics, gains and the 0.18 rad tracking limit follow Unitree xr_teleoperate.
"""

import fcntl
from pathlib import Path
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


class G1Dex1Hand:
    def __init__(self, is_left=True, released_q=5.4, pressed_q=0.0):
        self.side = "left" if is_left else "right"
        self.released_q = self.validate_target(released_q)
        self.pressed_q = self.validate_target(pressed_q)
        self._lock = threading.Lock()
        self._sample = None
        self._stamp = 0.0
        self.last_command = None
        self.state_sub = ChannelSubscriber(f"rt/dex1/{self.side}/state", MotorStates_)
        self.state_sub.Init(self._receive, 10)
        self.cmd_pub = None
        self._command_lock = None

    @staticmethod
    def validate_target(value):
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 5.4:
            raise ValueError("Dex1 target must be finite and in [0, 5.4] rad")
        return value

    def _receive(self, msg):
        if len(msg.states) != 1:
            return
        motor = msg.states[0]
        values = np.array([motor.q, motor.dq, motor.tau_est, motor.ddq])
        if not np.all(np.isfinite(values)):
            return
        with self._lock:
            self._sample = values
            self._stamp = time.monotonic()

    def _state(self):
        with self._lock:
            if self._sample is None or time.monotonic() - self._stamp > 0.25:
                raise RuntimeError(f"Missing/stale Dex1 {self.side} state; check gripper service")
            return self._sample.copy()

    def observe(self):
        state = self._state()
        # Dex1 angles must never be treated as Dex3 finger joint angles.
        return {
            "hand_q": np.zeros(7), "hand_dq": np.zeros(7),
            "hand_ddq": np.zeros(7), "hand_tau_est": np.zeros(7),
            "dex1_hand_q": state[:1],
        }

    def queue_action(self, action):
        legacy = np.asarray(action["hand_q"], dtype=np.float64)
        if legacy.shape != (7,) or not np.all(np.isfinite(legacy)):
            raise ValueError("Expected seven finite legacy hand commands")
        # PICO trigger > 0.5 produces a nonzero grasp through the existing solver.
        target = self.pressed_q if np.max(np.abs(legacy)) > 0.05 else self.released_q
        self.command_position(target)

    def command_position(self, target):
        """Command one native angle, retaining the hardware tracking limit."""
        target = self.validate_target(target)
        measured = float(self._state()[0])
        if not -0.2 <= measured <= 5.6:
            raise RuntimeError(f"Dex1 {self.side} position {measured} outside calibrated range")
        target = float(np.clip(np.clip(target, measured - 0.18, measured + 0.18), 0.0, 5.4))
        motor = unitree_go_msg_dds__MotorCmd_()
        motor.q, motor.dq, motor.tau = target, 0.0, 0.0
        motor.kp, motor.kd = 5.0, 0.05
        if self.cmd_pub is None:
            lock_path = Path(__file__).resolve().parents[4] / ".runtime" / f"dex1-{self.side}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = lock_path.open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f"Another process is commanding Dex1 {self.side}") from None
            self._command_lock = lock
            self.cmd_pub = ChannelPublisher(f"rt/dex1/{self.side}/cmd", MotorCmds_)
            self.cmd_pub.Init()
        if not self.cmd_pub.Write(MotorCmds_([motor])):
            raise RuntimeError(f"Failed to publish Dex1 {self.side} command")
        self.last_command = np.array([target], dtype=np.float64)

    def calibrate_hand(self):
        # Calibration writes motor zero offsets and must be done separately.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                self._state()
                return
            except RuntimeError:
                time.sleep(0.02)
        self._state()

    def release_control(self):
        """Drop gains and command ownership while continuing to observe DDS."""
        try:
            if self.cmd_pub is not None:
                motor = unitree_go_msg_dds__MotorCmd_()
                motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.0
                try:
                    self.cmd_pub.Write(MotorCmds_([motor]))
                finally:
                    self.cmd_pub.Close()
        finally:
            self.cmd_pub = None
            self.last_command = None
            if self._command_lock is not None:
                self._command_lock.close()
                self._command_lock = None

    def close(self):
        try:
            self.release_control()
        finally:
            self.state_sub.Close()
