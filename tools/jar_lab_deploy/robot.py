"""G1 arm_sdk and Dex1 DDS endpoints, with read-only construction by default."""
import threading
import time
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


class UnitreeRobot:
    def __init__(self, interface, profile, live=False, domain=0):
        import unitree_sdk2py.core.channel as channels
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.utils.crc import CRC
        log_path = Path(__file__).resolve().parents[2] / "runtime/dds.log"
        log_path.parent.mkdir(exist_ok=True)
        # The SDK defaults to a system temporary directory; keep its trace in this project.
        config = channels.ChannelConfigHasInterface.replace("/tmp/cdds.LOG", escape(str(log_path)))
        if interface == "lo":
            config = config.replace("</Domain>", "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
                                    "<Peers><Peer address=\"127.0.0.1\"/></Peers></Discovery></Domain>")
        channels.ChannelConfigHasInterface = config
        ChannelFactoryInitialize(domain, interface)
        self.live, self.profile = live, profile
        self.lock = threading.Lock()
        self.body, self.hands, self.received = None, [None, None], [0., 0., 0.]
        self.waist = None
        self.mode_machine = 0
        self.subscribers = []
        def body_callback(message):
            values = np.asarray([m.q for m in message.motor_state[:29]], dtype=np.float64)
            if values.shape != (29,) or not np.isfinite(values[12:29]).all():
                return
            with self.lock:
                self.body, self.received[0] = values, time.monotonic()
                self.mode_machine = message.mode_machine
                if self.waist is None:
                    self.waist = values[12:15].copy()
        subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        subscriber.Init(body_callback, 10)
        self.subscribers.append(subscriber)
        for side, name in enumerate(("left", "right")):
            def hand_callback(message, side=side):
                if len(message.states) != 1:
                    return
                with self.lock:
                    self.hands[side], self.received[side + 1] = float(message.states[0].q), time.monotonic()
            subscriber = ChannelSubscriber(f"rt/dex1/{name}/state", MotorStates_)
            subscriber.Init(hand_callback, 10)
            self.subscribers.append(subscriber)
        self.publishers = []
        self.arm_message = unitree_hg_msg_dds__LowCmd_()
        self.hand_messages = []
        self.crc = CRC()
        self.weight = 0.
        if live:
            publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            publisher.Init()
            self.publishers.append(publisher)
            for name in ("left", "right"):
                publisher = ChannelPublisher(f"rt/dex1/{name}/cmd", MotorCmds_)
                publisher.Init()
                self.publishers.append(publisher)
                message = MotorCmds_()
                message.cmds = [unitree_go_msg_dds__MotorCmd_()]
                self.hand_messages.append(message)

    def observe(self):
        with self.lock:
            if self.body is None or any(h is None for h in self.hands):
                raise TimeoutError("Waiting for G1 body and both Dex1 state topics")
            values = np.concatenate((self.body[15:29], self.hands))
            age = time.monotonic() - min(self.received)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite robot observations")
        return values, age

    def send(self, reference, dt=.02, release=False):
        if not self.live:
            raise RuntimeError("Read-only robot has no command publishers")
        reference = np.asarray(reference)
        if reference.shape != (16,) or not np.isfinite(reference).all():
            raise ValueError("Invalid upper-body reference")
        self.weight = float(np.clip(self.weight + (-dt if release else dt) / 2, 0, 1))
        self.arm_message.mode_machine = self.mode_machine
        self.arm_message.motor_cmd[29].q = self.weight
        # Native arm_sdk owns arms+waist. Hold the measured startup waist;
        # the learned policy only supplies the fourteen arm values.
        for index, value in zip(list(range(15, 29)) + [12, 13, 14], list(reference[:14]) + list(self.waist)):
            motor = self.arm_message.motor_cmd[index]
            motor.q, motor.dq, motor.tau = float(value), 0., 0.
            motor.kp, motor.kd = self.profile["arm_kp"], self.profile["arm_kd"]
        self.arm_message.crc = self.crc.Crc(self.arm_message)
        self.publishers[0].Write(self.arm_message)
        if not release:
            for side in range(2):
                motor = self.hand_messages[side].cmds[0]
                motor.q, motor.dq, motor.tau = float(reference[14 + side]), 0., 0.
                motor.kp, motor.kd = self.profile["gripper_kp"], self.profile["gripper_kd"]
                self.publishers[side + 1].Write(self.hand_messages[side])

    def close(self, last_reference=None):
        if self.live and last_reference is not None:
            for _ in range(110):
                self.send(last_reference, release=True)
                time.sleep(.02)
        for channel in self.publishers + self.subscribers:
            channel.Close()
