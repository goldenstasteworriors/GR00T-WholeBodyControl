"""Measured-state G1 arm gravity feedforward; no actuator or ROS/DDS I/O.

Use the 29-DOF arm-only inertial model, not the 43-DOF Dex3 hand placeholders.
Optional tool payloads include Dex1/adapters/objects and are expressed in each
wrist-yaw joint frame. Unknown payload mass is never replaced by Dex3 mass.
"""
import json
from pathlib import Path
import time

import numpy as np
from contract import BODY_JOINT_NAMES


class ArmGravityCompensator:
    def __init__(self, robot, *, scale=1., max_torque=10., slew_rate=10., payload=None):
        import pinocchio as pin
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        if not np.isfinite([scale, max_torque, slew_rate]).all() or not 0 <= scale <= 1 or not 0 < max_torque <= 20 or not 0 < slew_rate <= 50:
            raise ValueError("Gravity limits: scale 0..1, max torque (0,20] Nm, slew (0,50] Nm/s")
        bare = instantiate_g1_robot_model(waist_location="lower_body", with_hands=False)
        self.pin, self.model = pin, bare.pinocchio_wrapper.model.copy()
        self.body = np.asarray(robot.get_body_actuated_joint_indices(), dtype=int)
        self.input_dofs = robot.num_dofs
        if len(self.body) != 29 or self.model.nq != 29 or self.model.nv != 29:
            raise ValueError("Gravity feedforward requires fixed-base G1 29-DOF model")
        names = [robot.joint_names[i] for i in self.body]
        if names != BODY_JOINT_NAMES:
            raise ValueError("Gravity input body order differs from the verified G1 motor order")
        if names[15:22] != ["left_"+s+"_joint" for s in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]:
            raise ValueError("Unexpected G1 arm actuator order")
        if names[22:29] != [n.replace("left_", "right_") for n in names[15:22]]:
            raise ValueError("Unexpected right arm actuator order")
        joints = [self.model.joints[self.model.getJointId(n)] for n in names]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError("Expected single-axis named G1 joints")
        self.q_indices = np.array([j.idx_q for j in joints])
        self.v_indices = np.array([j.idx_v for j in joints])
        self.payload = {} if payload is None else json.loads(Path(payload).read_text())
        if not isinstance(self.payload, dict) or set(self.payload)-{"left", "right"}:
            raise ValueError("Payload must map left/right to mass_kg and com_m")
        for side, item in self.payload.items():
            if not isinstance(item, dict) or set(item) != {"mass_kg", "com_m"}:
                raise ValueError("Payload requires mass_kg and com_m in wrist-yaw joint coordinates")
            mass, com = float(item["mass_kg"]), np.asarray(item["com_m"], dtype=float)
            if com.shape != (3,) or not np.isfinite(com).all() or not np.isfinite(mass) or not 0 <= mass <= 3 or np.linalg.norm(com) > .5:
                raise ValueError("Invalid tool payload mass/COM")
            jid = self.model.getJointId(f"{side}_wrist_yaw_joint")
            self.model.inertias[jid] += pin.Inertia(mass, com, np.zeros((3, 3)))
        self.data = self.model.createData()
        self.scale, self.slew_rate = float(scale), float(slew_rate)
        efforts = np.asarray(self.model.effortLimit)[self.v_indices[15:29]]
        if not np.isfinite(efforts).all() or np.any(efforts <= 0):
            raise ValueError("Missing arm effort limits in URDF")
        self.limits = np.minimum(efforts, max_torque)
        self.reset()

    def reset(self):
        self.previous = np.zeros(14)
        self.last_time = None
        self.last_raw = np.zeros(14)
        self.clipped = False

    def raw(self, observation):
        q_obs = np.asarray(observation["q"], dtype=float)
        base = np.asarray(observation["floating_base_pose"], dtype=float)
        if q_obs.shape != (self.input_dofs,) or not np.isfinite(q_obs).all() or base.shape != (7,) or not np.isfinite(base).all():
            raise ValueError("Gravity compensation requires finite measured joints and base pose")
        return self.raw_state(q_obs[self.body], base[3:7])

    def raw_state(self, measured_body, quat):
        measured_body, quat = np.asarray(measured_body, dtype=float), np.asarray(quat, dtype=float)
        if measured_body.shape != (29,) or quat.shape != (4,) or not np.isfinite(measured_body).all() or not np.isfinite(quat).all():
            raise ValueError("Gravity requires measured body29 and pelvis IMU quaternion wxyz")
        norm = np.linalg.norm(quat)
        if not .9 <= norm <= 1.1:
            raise ValueError("Invalid measured base quaternion for gravity compensation")
        w, x, y, z = quat/norm
        rotation = self.pin.Quaternion(w, x, y, z).matrix()
        self.model.gravity.linear = rotation.T @ np.array([0., 0., -9.81])
        q = np.zeros(self.model.nq)
        q[self.q_indices] = measured_body
        tau = np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q))
        result = np.zeros(29)
        result[15:29] = tau[self.v_indices[15:29]] * self.scale
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite gravity torque")
        return result

    def sample(self, observation, *, enabled=True, now=None):
        if not enabled:
            self.reset()
            return np.zeros(29)
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("Invalid gravity timestamp")
        raw = self.raw(observation)[15:29]
        self.last_raw = raw.copy()
        self.clipped = bool(np.any(np.abs(raw) > self.limits))
        target = np.clip(raw, -self.limits, self.limits)
        if self.last_time is None or now < self.last_time or now-self.last_time > .25:
            self.previous[:] = 0.
            dt = 0.
        else:
            dt = min(.05, now-self.last_time)
        self.last_time = now
        self.previous = np.clip(target, self.previous-self.slew_rate*dt, self.previous+self.slew_rate*dt)
        result = np.zeros(29)
        result[15:29] = self.previous
        return result

    def summary(self):
        return dict(enabled=True, model="g1_29dof.urdf; no Dex3 hand inertia",
                    scale=self.scale, limits_nm=self.limits.tolist(), slew_nm_s=self.slew_rate,
                    payload=self.payload, uncompensated_tools=[s for s in ("left", "right") if s not in self.payload])


def build_arm_gravity(robot, config, options=None):
    if not config.get("enable_gravity_compensation", False):
        return None
    if config.get("gravity_compensation_joints") not in (None, ["arms"]):
        raise ValueError("LabWAM gravity feedforward supports arms only; legs/waist remain with WBC")
    return ArmGravityCompensator(robot, **(options or {}))


def add_gravity_arguments(parser):
    parser.add_argument("--arm-gravity-scale", type=float, default=1.)
    parser.add_argument("--arm-gravity-max-torque", type=float, default=10.)
    parser.add_argument("--arm-gravity-slew-rate", type=float, default=10.)
    parser.add_argument("--arm-gravity-payload", type=Path, help="Optional left/right tool mass and COM JSON")


def gravity_options(args):
    return dict(scale=args.arm_gravity_scale, max_torque=args.arm_gravity_max_torque,
                slew_rate=args.arm_gravity_slew_rate, payload=args.arm_gravity_payload)
