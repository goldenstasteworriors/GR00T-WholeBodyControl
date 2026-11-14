"""Standalone MuJoCo arm dynamics for dataset-image/JAR state-feedback previews.

No robot, SDK, ROS or DDS imports. The original SONIC XML stays unchanged.
Pelvis/legs/waist are fixed. Integrate fourteen named arm joints and, when
selected, four Dex1 sliders; policy state remains fourteen arms + two motors.
"""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ARMS = [f"{side}_{joint}_joint" for side in ("left", "right") for joint in
        ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINTS = ARMS + ["left_hand_dex1_motor", "right_hand_dex1_motor"]
FINGERS = [f"{side}_dex1_finger_joint_{i}" for side in ("left", "right") for i in (1, 2)]


def slider(angle):
    return -.020 + .0445*np.clip(angle, 0., 5.4)/5.4


def motor(position):
    return np.clip((np.asarray(position)+.020)*5.4/.0445, 0., 5.4)


def add_dex1(root, urdf_path):
    """Use the same official Dex1-1 links/mapping as LabWeaver's preview.

    Keep SONIC's body, arm transforms, order and joint limits. Replace the fused
    rubber-hand wrist inertia with the bare wrist from the Dex1 URDF, then add
    the two 0.55kg nominal grippers. Never modify either source asset.
    """
    urdf_path = Path(urdf_path).resolve()
    digest = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if digest != "1a0a4f493d6686cd93952d8e67d6144d4c337929151c745e0440b8a28f08ed1a":
        raise ValueError("Dex1 URDF differs from the verified LabWeaver asset")
    source = ET.parse(urdf_path).getroot()
    asset = root.find("asset")
    def inertia(parent, original, scale=1.):
        origin = original.find("origin")
        if origin.get("rpy") != "0 0 0":raise ValueError("Unexpected inertial rotation")
        values = original.find("inertia")
        ET.SubElement(parent, "inertial", pos=origin.get("xyz"),
                      mass=str(float(original.find("mass").get("value"))*scale),
                      fullinertia=" ".join(str(float(values.get(k))*scale) for k in
                                           ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
    def geometry(parent, link, role):
        item = link.find(role)
        origin = item.find("origin")
        if origin.get("rpy") != "0 0 0":raise ValueError("Unexpected mesh rotation")
        path = (urdf_path.parent/item.find("geometry/mesh").get("filename")).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        name = "dexasset_"+path.stem
        if asset.find(f"mesh[@name='{name}']") is None:
            ET.SubElement(asset,"mesh",name=name,file=str(path))
        visual = role == "visual"
        ET.SubElement(parent,"geom",type="mesh",mesh=name,pos=origin.get("xyz"),
                      group="1" if visual else "3",density="0",
                      contype="0" if visual else "1",conaffinity="0" if visual else "1",
                      rgba=".72 .75 .82 1" if "base" in link.get("name") else ".28 .29 .31 1")
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        old_mass = float(wrist.find("inertial").get("mass"))
        bare = source.find(f"link[@name='{side}_wrist_yaw_link']")
        # Canonical MJCF mass includes the 0.17kg rubber hand.
        if abs(old_mass-float(bare.find("inertial/mass").get("value"))-.17)>2e-6:
            raise ValueError("Unexpected original wrist/rubber-hand mass")
        wrist.remove(wrist.find("inertial"))
        inertia(wrist,bare.find("inertial"))
        for geom in list(wrist.findall("geom")):wrist.remove(geom)
        geometry(wrist,bare,"visual");geometry(wrist,bare,"collision")
        nodes={side+"_wrist_yaw_link":wrist}
        for joint_name in [side+"_base_joint"]+[f"{side}_dex1_finger_joint_{i}" for i in (1,2)]:
            joint=source.find(f"joint[@name='{joint_name}']")
            origin=joint.find("origin")
            if origin.get("rpy")!="0 0 0":raise ValueError("Unexpected Dex1 mount rotation")
            child=joint.find("child").get("link")
            body=ET.SubElement(nodes[joint.find("parent").get("link")],"body",name=child,pos=origin.get("xyz"))
            nodes[child]=body
            link=source.find(f"link[@name='{child}']")
            inertia(body,link.find("inertial"),.55/.364946)
            geometry(body,link,"visual");geometry(body,link,"collision")
            if joint.get("type")=="prismatic":
                limit=joint.find("limit")
                ET.SubElement(body,"joint",name=joint_name,type="slide",axis=joint.find("axis").get("xyz"),
                              range=f"{limit.get('lower')} {limit.get('upper')}",armature=".001",damping="1")
                ET.SubElement(root.find("actuator"),"position",name=joint_name,joint=joint_name,
                              kp="1000",kv="10",ctrllimited="true",ctrlrange="-.020 .0245",
                              forcelimited="true",forcerange="-20 20")
    return digest


class ArmPhysics:
    def __init__(self, xml_path, initial, *, speed=.25, kp=60., kd=1.5, dex1_urdf=None):
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
        self.dex1 = dex1_urdf is not None
        self.asset_sha256 = add_dex1(root,dex1_urdf) if self.dex1 else None
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
        dofs=18 if self.dex1 else 14
        if self.model.nq != dofs or self.model.nv != dofs or self.model.nu != dofs:
            raise ValueError("Unexpected number of arm/finger dynamics DOFs")
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(dofs)]
        if [n for n in names if n in ARMS] != ARMS or set(names)!=set(ARMS+(FINGERS if self.dex1 else [])):
            raise ValueError("Arm joint order differs from the observation contract")
        motors = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(14)]
        if motors != ARMS:
            raise ValueError("Motor order differs from the observation contract")
        self.arm_q=np.array([self.model.joint(n).qposadr[0] for n in ARMS])
        self.arm_v=np.array([self.model.joint(n).dofadr[0] for n in ARMS])
        self.finger_q=np.array([self.model.joint(n).qposadr[0] for n in FINGERS]) if self.dex1 else np.array([],dtype=int)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != ((16,) if self.dex1 else (14,)) or not np.isfinite(initial).all():
            raise ValueError("Need a finite measured initial arm pose")
        self.low, self.high = self.model.jnt_range[self.arm_q].T.copy()
        if np.any(initial[:14] < self.low-1e-5) or np.any(initial[:14] > self.high+1e-5):
            raise ValueError("Recorded initial pose lies outside canonical G1 joint limits")
        self.data.qpos[self.arm_q] = initial[:14]
        if self.dex1:
            if np.any(initial[14:]<0) or np.any(initial[14:]>5.4):raise ValueError("Invalid initial Dex1 motor angles")
            self.data.qpos[self.finger_q]=np.repeat(slider(initial[14:]),2)
            self.data.ctrl[14:]=self.data.qpos[self.finger_q]
        self.set_reference(initial)
        self.command = initial[:14].copy()
        self.gravity = np.zeros(14)
        self.effort = np.array([effort[n] for n in ARMS])
        self.gravity_limit = np.minimum(10., self.effort[:, 1])
        self.kp, self.kd, self.speed = kp, kd, speed
        self.target_clips = 0
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        arms=self.data.qpos[self.arm_q].copy()
        return np.r_[arms,motor(self.data.qpos[self.finger_q].reshape(2,2).mean(axis=1))] if self.dex1 else arms

    def advance(self, target, dt=1/30):
        target = np.asarray(target, dtype=float)
        if target.shape != ((16,) if self.dex1 else (14,)) or not np.isfinite(target).all():
            raise ValueError("Invalid predicted arm observation")
        clipped = np.clip(target[:14], self.low, self.high)
        self.target_clips += int(np.any(clipped != target[:14]))
        if self.dex1:self.data.ctrl[14:]=np.repeat(slider(target[14:]),2)
        steps = round(dt/self.model.opt.timestep)
        if abs(steps*self.model.opt.timestep-dt) > 1e-9:
            raise ValueError("Frame interval must be an integer number of physics steps")
        for _ in range(steps):
            self.command += np.clip(clipped-self.command, -self.speed*self.model.opt.timestep,
                                    self.speed*self.model.opt.timestep)
            # Fixed-base generalized bias compensates gravity/Coriolis. Match
            # deployment's arm-only 10Nm/URDF cap and 10Nm/s startup ramp.
            desired = np.clip(self.data.qfrc_bias[self.arm_v], -self.gravity_limit, self.gravity_limit)
            self.gravity += np.clip(desired-self.gravity, -10*self.model.opt.timestep, 10*self.model.opt.timestep)
            self.data.ctrl[:14] = self.kp*self.command+self.gravity
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulated state")
        return self.state()

    def set_reference(self, arm_pose):
        pose=np.asarray(arm_pose)
        self.reference.qpos[self.arm_q] = pose[:14]
        if self.dex1:self.reference.qpos[self.finger_q]=np.repeat(slider(pose[14:]),2)
        self.reference.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.reference)

    def summary(self):
        return dict(engine="MuJoCo",version=mujoco.__version__,physics_hz=600,fps=30,
                    fixed="pelvis, legs and waist at canonical XML zero pose; no locomotion controller",
                    simulated_joints=ARMS,kp=self.kp,kd=self.kd,arm_reference_speed_rad_s=self.speed,
                    feedforward="arm generalized bias, capped at min(10Nm, URDF effort), ramp 10Nm/s",
                    tool_inertia=("bare wrist + Dex1 URDF inertia scaled to 0.55kg per hand; unmeasured adapters/objects excluded"
                                  if self.dex1 else "canonical XML inertias; no additional Dex1/adaptor/tool inertia; dummy hand visuals hidden"),
                    dex1_urdf_sha256=self.asset_sha256,
                    gripper=("4 dynamic sliders, position PD 1000/10, 20N force limit; linear motor 0..5.4rad to -0.020..0.0245m is approximate"
                             if self.dex1 else "external ideal servo"))
