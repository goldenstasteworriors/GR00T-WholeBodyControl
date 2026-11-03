"""Apply the navigation-specific arm posture without changing robot defaults."""
import json
import math
from pathlib import Path


def apply_arm_pose(robot_model, pose, path):
    profile = json.loads(Path(path).read_text())
    expected = {f"{side}_{joint}_joint" for side in ("left", "right")
                for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                              "wrist_roll", "wrist_pitch", "wrist_yaw")}
    values = profile["joints"]
    if profile.get("units") != "rad" or set(values) != expected:
        raise ValueError("Startup arm profile must specify all 14 named arm joints in radians")
    result = pose.copy()
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid startup joint value: {name}")
        result[robot_model.dof_index(name)] = value
    print(f"[NaVILA] Raised symmetric startup arms loaded from {path}", flush=True)
    return result
