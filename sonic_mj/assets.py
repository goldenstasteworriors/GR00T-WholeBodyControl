from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

SONIC_G1_XML = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "g1_29dof_rev_1_0.xml"
)
SONIC_H2_XML = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "h2.xml"
)
SONIC_H2_MESH_DIR = (
    Path(__file__).resolve().parents[1]
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "urdf"
    / "h2"
    / "meshes"
)


@dataclass(frozen=True)
class SonicRobotProfile:
    robot_type: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    action_scale: dict[str, float]
    default_joint_pos: dict[str, float]
    init_pos: tuple[float, float, float]
    isaaclab_joints: tuple[str, ...]
    motion_dof_to_mujoco: tuple[int, ...]
    robot_cfg_fn: Callable[[], EntityCfg]

SONIC_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SONIC_G1_MOTION_DOF_TO_MUJOCO = tuple(range(len(SONIC_G1_JOINT_NAMES)))

SONIC_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

G1_ISAACLAB_JOINTS = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

SONIC_H2_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SONIC_H2_MOTION_DOF_TO_MUJOCO = tuple(range(len(SONIC_H2_JOINT_NAMES)))

SONIC_H2_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_pitch_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

H2_ISAACLAB_JOINTS = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_pitch_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "head_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

G1_ISAACLAB_TO_MUJOCO_DOF = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

G1_MUJOCO_TO_ISAACLAB_DOF = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

G1_ISAACLAB_TO_MUJOCO_BODY = (
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
)

G1_MUJOCO_TO_ISAACLAB_BODY = (
    0,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    3,
    6,
    9,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    13,
    17,
    21,
    23,
    25,
    27,
    29,
)

SONIC_G1_ACTION_SCALE = {
    ".*_hip_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,
    ".*_hip_roll_joint": 0.25 * 139.0 / STIFFNESS_7520_22,
    ".*_hip_pitch_joint": 0.25 * 139.0 / STIFFNESS_7520_22,
    ".*_knee_joint": 0.25 * 139.0 / STIFFNESS_7520_22,
    "waist_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    "waist_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    "waist_yaw_joint": 0.25 * 88.0 / STIFFNESS_7520_14,
    ".*_ankle_pitch_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    ".*_ankle_roll_joint": 0.25 * 50.0 / (2.0 * STIFFNESS_5020),
    ".*_shoulder_pitch_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_shoulder_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_shoulder_yaw_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_elbow_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_wrist_roll_joint": 0.25 * 25.0 / STIFFNESS_5020,
    ".*_wrist_pitch_joint": 0.25 * 5.0 / STIFFNESS_4010,
    ".*_wrist_yaw_joint": 0.25 * 5.0 / STIFFNESS_4010,
}

SONIC_G1_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

SONIC_H2_ACTION_SCALE = {
    ".*_hip_yaw_joint": 0.25 * 264.0 / STIFFNESS_7520_14,
    ".*_hip_roll_joint": 0.25 * 417.0 / STIFFNESS_7520_22,
    ".*_hip_pitch_joint": 0.25 * 417.0 / STIFFNESS_7520_22,
    ".*_knee_joint": 0.25 * 417.0 / STIFFNESS_7520_22,
    ".*_ankle_pitch_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    ".*_ankle_roll_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    "waist_roll_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    "waist_pitch_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    "waist_yaw_joint": 0.25 * 264.0 / STIFFNESS_7520_14,
    "head_pitch_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    "head_yaw_joint": 0.25 * 150.0 / (2.0 * STIFFNESS_5020),
    ".*_shoulder_pitch_joint": 0.25 * 75.0 / STIFFNESS_5020,
    ".*_shoulder_roll_joint": 0.25 * 75.0 / STIFFNESS_5020,
    ".*_shoulder_yaw_joint": 0.25 * 75.0 / STIFFNESS_5020,
    ".*_elbow_joint": 0.25 * 75.0 / STIFFNESS_5020,
    ".*_wrist_roll_joint": 0.25 * 75.0 / STIFFNESS_5020,
    ".*_wrist_pitch_joint": 0.25 * 15.0 / STIFFNESS_4010,
    ".*_wrist_yaw_joint": 0.25 * 15.0 / STIFFNESS_4010,
}

SONIC_H2_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def get_sonic_g1_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SONIC_G1_XML))
    while spec.actuators:
        spec.delete(spec.actuators[0])
    return spec


def get_sonic_h2_spec() -> mujoco.MjSpec:
    root = ET.parse(SONIC_H2_XML).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(SONIC_H2_MESH_DIR))
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    while spec.actuators:
        spec.delete(spec.actuators[0])
    return spec


def get_sonic_g1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_sonic_g1_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            joint_pos=SONIC_G1_DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        ".*_hip_roll_joint",
                        ".*_hip_pitch_joint",
                        ".*_knee_joint",
                    ),
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    armature=ARMATURE_7520_22,
                    effort_limit=139.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_yaw_joint",),
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    effort_limit=88.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    effort_limit=50.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    effort_limit=50.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("waist_yaw_joint",),
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    effort_limit=88.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_roll_joint",
                    ),
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    armature=ARMATURE_5020,
                    effort_limit=25.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    armature=ARMATURE_4010,
                    effort_limit=5.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def get_sonic_h2_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_sonic_h2_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.04),
            joint_pos=SONIC_H2_DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        ".*_hip_roll_joint",
                        ".*_hip_pitch_joint",
                        ".*_knee_joint",
                    ),
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    armature=ARMATURE_7520_22,
                    effort_limit=417.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_yaw_joint",),
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    effort_limit=264.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    effort_limit=150.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    effort_limit=150.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("waist_yaw_joint",),
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    armature=ARMATURE_7520_14,
                    effort_limit=264.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=("head_pitch_joint", "head_yaw_joint"),
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    armature=2.0 * ARMATURE_5020,
                    effort_limit=150.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_roll_joint",
                    ),
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    armature=ARMATURE_5020,
                    effort_limit=75.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    armature=ARMATURE_4010,
                    effort_limit=15.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


SONIC_ROBOT_PROFILES = {
    "g1_model_12_dex": SonicRobotProfile(
        robot_type="g1_model_12_dex",
        joint_names=SONIC_G1_JOINT_NAMES,
        body_names=SONIC_G1_BODY_NAMES,
        action_scale=SONIC_G1_ACTION_SCALE,
        default_joint_pos=SONIC_G1_DEFAULT_JOINT_POS,
        init_pos=(0.0, 0.0, 0.76),
        isaaclab_joints=G1_ISAACLAB_JOINTS,
        motion_dof_to_mujoco=SONIC_G1_MOTION_DOF_TO_MUJOCO,
        robot_cfg_fn=get_sonic_g1_robot_cfg,
    ),
    "g1": SonicRobotProfile(
        robot_type="g1",
        joint_names=SONIC_G1_JOINT_NAMES,
        body_names=SONIC_G1_BODY_NAMES,
        action_scale=SONIC_G1_ACTION_SCALE,
        default_joint_pos=SONIC_G1_DEFAULT_JOINT_POS,
        init_pos=(0.0, 0.0, 0.76),
        isaaclab_joints=G1_ISAACLAB_JOINTS,
        motion_dof_to_mujoco=SONIC_G1_MOTION_DOF_TO_MUJOCO,
        robot_cfg_fn=get_sonic_g1_robot_cfg,
    ),
    "h2": SonicRobotProfile(
        robot_type="h2",
        joint_names=SONIC_H2_JOINT_NAMES,
        body_names=SONIC_H2_BODY_NAMES,
        action_scale=SONIC_H2_ACTION_SCALE,
        default_joint_pos=SONIC_H2_DEFAULT_JOINT_POS,
        init_pos=(0.0, 0.0, 1.04),
        isaaclab_joints=H2_ISAACLAB_JOINTS,
        motion_dof_to_mujoco=SONIC_H2_MOTION_DOF_TO_MUJOCO,
        robot_cfg_fn=get_sonic_h2_robot_cfg,
    ),
}


def get_sonic_robot_profile(robot_type: str | None) -> SonicRobotProfile:
    key = robot_type or "g1_model_12_dex"
    if key not in SONIC_ROBOT_PROFILES:
        supported = ", ".join(sorted(SONIC_ROBOT_PROFILES))
        raise ValueError(f"Unsupported SonicMJ robot type '{key}'. Supported: {supported}")
    return SONIC_ROBOT_PROFILES[key]
