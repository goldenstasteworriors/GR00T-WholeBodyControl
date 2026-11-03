"""PICO locomotion/recording only: no arm IK, clutch capture or gripper commands."""

import json
import math
from pathlib import Path

import numpy as np


ARM_PROFILE = Path(__file__).resolve().parents[3] / "gear_sonic/config/data_collection/navila_arm_pose.json"


def navigation_arm_pose(robot_model, pose):
    profile = json.loads(ARM_PROFILE.read_text())
    expected = {f"{side}_{joint}_joint" for side in ("left", "right")
                for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                              "wrist_roll", "wrist_pitch", "wrist_yaw")}
    if profile["units"] != "degrees" or set(profile["joints"]) != expected:
        raise ValueError("Navigation arm profile must contain the 14 named arm joints in degrees")
    result = pose.copy()
    for name, degrees in profile["joints"].items():
        value = math.radians(degrees)
        if not math.isfinite(value):
            raise ValueError(f"Invalid navigation arm angle: {name}")
        result[robot_model.dof_index(name)] = value
    return result


class NavigationCollectionPolicy:
    def __init__(self):
        from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer

        self.streamer = PicoStreamer()
        self.streamer.start_streaming()
        self.running = False

    def set_lower_body_policy_active(self, active):
        self.running = bool(active)
        self.streamer.set_lower_body_policy_active(active)

    def get_action(self):
        data = self.streamer.get()
        # A+X, tracked hand transforms and trigger/grip values never reach IK
        # or the grippers. Four-button and recording events retain their IDs.
        return {
            "navigate_cmd": data.control_data.get("navigate_cmd", [0., 0., 0.])
            if self.running else [0., 0., 0.],
            "base_height_command": 0.74,
            "start_stop_event_id": data.control_data.get("start_stop_event_id", 0),
            "latency_trace": data.control_data.get("latency_trace", {}).copy(),
            **data.data_collection_data,
        }

    def close(self):
        self.streamer.stop_streaming()


def navigation_features():
    return {
        "navigation.requested_velocity": {"dtype": "float64", "shape": (3,), "names": ["vx_m_s", "vy_m_s", "wz_rad_s"]},
        "navigation.policy_velocity": {"dtype": "float64", "shape": (3,), "names": ["vx_m_s", "vy_m_s", "wz_rad_s"]},
        "navigation.state_time": {"dtype": "float64", "shape": (1,), "names": ["thinkpad_unix_seconds"]},
        "navigation.camera_time": {"dtype": "float64", "shape": (1,), "names": ["robot_camera_clock_seconds"]},
        "navigation.control_mode": {"dtype": "int64", "shape": (1,), "names": ["damped_0_initializing_1_ready_2_running_3"]},
    }


def navigation_frame(proprio, camera_time):
    mode = proprio["control_mode"]
    return {
        "navigation.requested_velocity": np.asarray(proprio["navigate_command"], dtype=np.float64).reshape(3),
        "navigation.policy_velocity": np.asarray(proprio["navigation_policy_velocity"], dtype=np.float64).reshape(3),
        "navigation.state_time": np.array([proprio["timestamps"]["main_loop"]], dtype=np.float64),
        "navigation.camera_time": np.array([camera_time], dtype=np.float64),
        "navigation.control_mode": np.array([{"damped": 0, "initializing": 1, "ready": 2, "running": 3}[mode]], dtype=np.int64),
    }
