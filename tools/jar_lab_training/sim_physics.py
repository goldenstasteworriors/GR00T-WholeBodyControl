"""Standalone MuJoCo arm dynamics for dataset-image/JAR state-feedback previews.

No robot, SDK, ROS or DDS imports. The original SONIC XML stays unchanged.
Pelvis/legs/waist are fixed; only the fourteen named arm joints are integrated.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ARMS = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in
        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINTS = ARMS + ["left_hand_dex1_motor", "right_hand_dex1_motor"]


class ArmPhysics:
    def __init__(self, xml_path, initial, *, speed=.25, kp=60., kd=1.5):
        xml_path = Path(xml_path).resolve()
        root = ET.parse(xml_path).getroot()
        if root.attrib.get("model") != "g1_29dof_rev_1_0":
            raise ValueError("Use the canonical SONIC G1 rev1 29-DOF model")
        root.find("compiler").set("meshdir", str((xml_path.parent/root.find("compiler").get("meshdir")).resolve()))
        effort = {}
        for body in root.iter("body"):
            for geom in list(body.findall("geom")):
                if geom.get("mesh", "").endswith("rubber_hand"):
                    body.remove(geom)  # do not present the canonical dummy hand as Dex1
            for joint in list(body.findall("joint")):
                name = joint.get("name")
                if name not in ARMS:
                    body.remove(joint)
                else:
                    effort[name] = list(map(float, joint.get("actuatorfrcrange").split()))
        actuator = root.find("actuator")
        for motor in list(actuator):
            if motor.get("joint") not in ARMS:
                actuator.remove(motor)
            else:
                # Express PD as an affine actuator so implicitfast can integrate
                # its velocity derivative; explicit wrist damping is unstable
                # at useful simulation timesteps for the small wrist inertias.
                motor.tag = "general"
                motor.set("gaintype", "fixed")
                motor.set("gainprm", "1")
                motor.set("biastype", "affine")
                motor.set("biasprm", f"0 {-kp} {-kd}")
                motor.set("forcelimited", "true")
                motor.set("forcerange", " ".join(map(str, effort[motor.get("joint")])))
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("timestep", str(1/600))
        option.set("integrator", "implicitfast")
        option.set("gravity", "0 0 -9.81")
        visual = root.find("visual/global")
        visual.set("offwidth", "560")
        visual.set("offheight", "800")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.reference = mujoco.MjData(self.model)
        if self.model.nq != 14 or self.model.nv != 14 or self.model.nu != 14:
            raise ValueError("Preview dynamics must have exactly fourteen arm DOFs")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(14)]
        if names != ARMS:
            raise ValueError("Arm joint order differs from the observation contract")
        motors = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(14)]
        if motors != ARMS:
            raise ValueError("Motor order differs from the observation contract")
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (14,) or not np.isfinite(initial).all():
            raise ValueError("Need a finite measured initial arm pose")
        self.low, self.high = self.model.jnt_range.T.copy()
        if np.any(initial < self.low-1e-5) or np.any(initial > self.high+1e-5):
            raise ValueError("Recorded initial pose lies outside canonical G1 joint limits")
        self.data.qpos[:] = self.reference.qpos[:] = initial
        self.command = initial.copy()
        self.gravity = np.zeros(14)
        self.effort = np.array([effort[n] for n in ARMS])
        self.gravity_limit = np.minimum(10., self.effort[:, 1])
        self.kp, self.kd, self.speed = kp, kd, speed
        self.target_clips = 0
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return self.data.qpos.copy()

    def advance(self, target, dt=1/30):
        target = np.asarray(target, dtype=float)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError("Invalid predicted arm observation")
        clipped = np.clip(target, self.low, self.high)
        self.target_clips += int(np.any(clipped != target))
        steps = round(dt/self.model.opt.timestep)
        if abs(steps*self.model.opt.timestep-dt) > 1e-9:
            raise ValueError("Frame interval must be an integer number of physics steps")
        for _ in range(steps):
            self.command += np.clip(clipped-self.command, -self.speed*self.model.opt.timestep,
                                    self.speed*self.model.opt.timestep)
            # Fixed-base generalized bias compensates gravity/Coriolis. Match
            # deployment's arm-only 10Nm/URDF cap and 10Nm/s startup ramp.
            desired = np.clip(self.data.qfrc_bias, -self.gravity_limit, self.gravity_limit)
            self.gravity += np.clip(desired-self.gravity, -10*self.model.opt.timestep, 10*self.model.opt.timestep)
            self.data.ctrl[:] = self.kp*self.command+self.gravity
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulated state")
        return self.state()

    def set_reference(self, arm_pose):
        self.reference.qpos[:] = np.asarray(arm_pose)
        self.reference.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.reference)

    def summary(self):
        return dict(engine="MuJoCo",version=mujoco.__version__,physics_hz=600,fps=30,
                    fixed="pelvis, legs and waist at canonical XML zero pose; no locomotion controller",
                    simulated_joints=ARMS,kp=self.kp,kd=self.kd,arm_reference_speed_rad_s=self.speed,
                    feedforward="arm generalized bias, capped at min(10Nm, URDF effort), ramp 10Nm/s",
                    tool_inertia="canonical XML inertias; no additional Dex1/adaptor/tool inertia; dummy hand visuals hidden")
