"""Unitree Revo2 DDS contract; positions are normalized closure, not radians."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

JOINT_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
HAND_TYPE = "brainco_revo2"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/data_collection/brainco_hand_pose.json"


def six(values, *, normalized=True):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError("Expected exactly six BrainCo motors in official order")
    if any(isinstance(v, bool) for v in values):
        raise ValueError("Boolean is not a motor value")
    result = [float(v) for v in values]
    if any(not math.isfinite(v) or (normalized and not 0 <= v <= 1) for v in result):
        raise ValueError("Expected finite motor values" + (" in [0,1]" if normalized else ""))
    return result


def load_profiles(path=DEFAULT_CONFIG):
    with Path(path).expanduser().open() as f:
        cfg = json.load(f)
    if cfg.get("hand_type") != HAND_TYPE or cfg.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("Expected brainco_revo2 and official joint_names")
    speed = float(cfg["speed"])
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("BrainCo speed must be in (0,1]")
    return {
        "hand_type": HAND_TYPE, "joint_names": list(JOINT_NAMES), "speed": speed,
        **{side: {pose: six(cfg[side][pose]) for pose in ("open", "grasp")}
           for side in ("left", "right")},
    }


def native_frame(proprio, max_age_s):
    """Validate actual C++ DDS snapshots; never reconstruct a command from a gesture."""
    meta = [float(v) for v in proprio.get("brainco_state", [])]
    if (len(meta) != 16 or any(not math.isfinite(v) for v in meta)
            or any(v < 0 or v > max_age_s for v in meta[:2]) or meta[2:4] != [1.0, 1.0]):
        raise ValueError("Waiting for fresh BrainCo states and published commands on both sides")
    return {
        "left_q": six(list(proprio["left_hand_q"])),
        "right_q": six(list(proprio["right_hand_q"])),
        "left_action": six(list(proprio["last_left_hand_action"])),
        "right_action": six(list(proprio["last_right_hand_action"])),
        "speed": six(list(proprio["left_hand_dq"]), normalized=False)
                 + six(list(proprio["right_hand_dq"]), normalized=False),
        "current_a": meta[4:], "state_age_s": meta[:2],
    }


class BraincoDdsClient:
    """Direct official MotorCmds_/MotorStates_ topics, no Modbus or proxy topics."""
    def __init__(self, network, domain_id=0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        ChannelFactoryInitialize(domain_id, network)
        self._message_type, self._motor_type = MotorCmds_, unitree_go_msg_dds__MotorCmd_
        self._lock = threading.Lock()
        self._state, self._publishers, self._subscribers = {}, {}, {}
        for side in ("left", "right"):
            pub = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
            pub.Init()
            sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
            sub.Init(lambda msg, side=side: self._receive(side, msg), 1)
            self._publishers[side], self._subscribers[side] = pub, sub

    def _receive(self, side, msg):
        try:
            values = {
                "q": six([v.q for v in msg.states]),
                "speed": six([v.dq for v in msg.states], normalized=False),
                "current_a": six([v.tau_est for v in msg.states], normalized=False),
                "received_monotonic": time.monotonic(),
            }
        except (TypeError, ValueError):
            return  # Invalid messages do not refresh freshness.
        with self._lock:
            self._state[side] = values

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            return {side: (None if side not in self._state else {
                "q": list(self._state[side]["q"]),
                "speed": list(self._state[side]["speed"]),
                "current_a": list(self._state[side]["current_a"]),
                "age_s": now - self._state[side]["received_monotonic"],
            }) for side in ("left", "right")}

    def command(self, side, values, speed=1.0):
        if side not in self._publishers:
            raise ValueError("side must be left or right")
        q = six(values)
        speed = float(speed)
        if not math.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError("speed must be in (0,1]")
        msg = self._message_type()
        msg.cmds = [self._motor_type() for _ in range(6)]
        for motor, value in zip(msg.cmds, q):
            motor.q, motor.dq = value, speed
        if not self._publishers[side].Write(msg):
            raise RuntimeError("DDS publish failed")
        return q

    def close(self):
        for channel in (*self._subscribers.values(), *self._publishers.values()):
            channel.Close()
