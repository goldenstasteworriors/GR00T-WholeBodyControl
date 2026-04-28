from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import easydict
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

from sonic_mj.assets import (
    G1_ISAACLAB_TO_MUJOCO_BODY,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_ISAACLAB_JOINTS,
    G1_MUJOCO_TO_ISAACLAB_BODY,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)
from gear_sonic.utils.motion_lib import motion_lib_robot


def _as_tensor(value, *, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


class SonicMotionCommand(CommandTerm):
    cfg: SonicMotionCommandCfg

    def __init__(self, cfg: SonicMotionCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.reward_point_body_indices = [
            self.robot.body_names.index(name) for name in cfg.reward_point_body
        ]
        self.reward_point_body_indices_motion = [
            cfg.body_names.index(name) for name in cfg.reward_point_body
        ]
        self.reward_point_body_offsets = _as_tensor(
            cfg.reward_point_body_offset, device=self.device
        ).view(1, -1, 3)
        self.vr_3point_body_indices_motion = [
            cfg.body_names.index(name) for name in cfg.vr_3point_body
        ]
        self.vr_3point_body_offsets = _as_tensor(
            cfg.vr_3point_body_offset, device=self.device
        ).view(1, -1, 3)

        motion_cfg = easydict.EasyDict(cfg.motion_lib_cfg or {})
        motion_cfg.setdefault(
            "asset",
            {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": "g1_29dof_rev_1_0.xml",
                "urdfFileName": "",
            },
        )
        motion_cfg.setdefault("extend_config", [])
        motion_cfg.setdefault("target_fps", int(round(1.0 / env.step_dt)))
        motion_cfg.setdefault("multi_thread", True)
        motion_cfg.setdefault("motion_file", cfg.motion_file)
        motion_cfg.setdefault("smpl_motion_file", cfg.smpl_motion_file)
        motion_cfg.update(
            {
                "isaaclab_joints": G1_ISAACLAB_JOINTS,
                "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
                "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
                "isaaclab_to_mujoco_body": G1_ISAACLAB_TO_MUJOCO_BODY,
                "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
                "body_indexes": self.body_indexes,
                "body_indexes_data": list(range(len(cfg.body_names))),
                "lower_joint_indices_mujoco": list(range(12)),
                "cat_upper_body_poses": cfg.cat_upper_body_poses,
                "cat_upper_body_poses_prob": cfg.cat_upper_body_poses_prob,
                "randomize_heading": cfg.randomize_heading,
                "freeze_frame_aug": cfg.freeze_frame_aug,
                "freeze_frame_aug_prob": cfg.freeze_frame_aug_prob,
            }
        )
        self.motion_lib = motion_lib_robot.MotionLibRobot(
            motion_cfg, self.num_envs, self.device
        )
        self.max_num_load_motions = min(self.num_envs, cfg.max_num_load_motions)
        self.motion_lib.load_motions_for_training(max_num_seqs=self.max_num_load_motions)
        self._motion_lib = self.motion_lib

        self.motion_ids = self.motion_lib.sample_motions(self.num_envs)
        self.motion_start_time_steps = self.motion_lib.sample_time_steps(self.motion_ids)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self.encoder_index = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self._body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self._body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self._body_quat_relative_w[..., 0] = 1.0
        self._refresh_motion_state()
        self._print_order_summary()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_state["dof_pos"]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_state["dof_vel"]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._motion_state["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._motion_state["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._motion_state["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._motion_state["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        return self._body_pos_relative_w

    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        return self._body_quat_relative_w

    def future_state(self, num_frames: int | None = None, dt: float | None = None) -> dict:
        num_frames = num_frames or self.cfg.num_future_frames
        dt = dt or self.cfg.dt_future_ref_frames
        offsets = torch.arange(num_frames, device=self.device, dtype=torch.float32) * dt
        motion_times = (
            (self.motion_start_time_steps + self.time_steps).float() * self._env.step_dt
        )[:, None] + offsets[None, :]
        flat_ids = self.motion_ids[:, None].expand(-1, num_frames).reshape(-1)
        flat_times = motion_times.reshape(-1)
        state = self.motion_lib.get_motion_state(flat_ids, flat_times)
        return {k: v.reshape(self.num_envs, num_frames, *v.shape[1:]) for k, v in state.items()}

    def _refresh_motion_state(self) -> None:
        motion_times = (self.motion_start_time_steps + self.time_steps).float() * self._env.step_dt
        self._motion_state = self.motion_lib.get_motion_state(self.motion_ids, motion_times)
        anchor_pos = self.anchor_pos_w[:, None, :]
        anchor_quat = self.anchor_quat_w[:, None, :]
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :]
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :]
        delta_pos_w = robot_anchor_pos.repeat(1, len(self.cfg.body_names), 1)
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        delta_ori_body_w = delta_ori_w.repeat(1, len(self.cfg.body_names), 1)
        self._body_quat_relative_w = quat_mul(delta_ori_body_w, self.body_quat_w)
        self._body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_body_w, self.body_pos_w - anchor_pos
        )

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
        self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
            self.motion_ids[env_ids]
        )
        self.time_steps[env_ids] = 0
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(self.motion_ids[env_ids])
        if self.cfg.encoder_sampling_mode == "cycle":
            self.encoder_index[env_ids, 0] = self.command_counter[env_ids].long() % 3
        else:
            self.encoder_index[env_ids, 0] = torch.randint(
                0, 3, (len(env_ids),), device=self.device
            )
        self._refresh_motion_state()
        root_state = torch.cat(
            [
                self._motion_state["root_pos"][env_ids] + self._env.scene.env_origins[env_ids],
                self._motion_state["root_rot"][env_ids],
                self._motion_state["root_vel"][env_ids],
                self._motion_state["root_ang_vel"][env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.joint_pos[env_ids], self.joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.reset(env_ids=env_ids)

    def _update_command(self) -> None:
        self.time_steps += 1
        done = torch.where(self.time_steps >= self.motion_num_steps)[0]
        if done.numel() > 0:
            self._resample_command(done)
        self._refresh_motion_state()

    def _print_order_summary(self) -> None:
        print("[SonicMJ] robot joint names/order:", self.robot.joint_names)
        print("[SonicMJ] robot body names/order:", self.robot.body_names)
        print("[SonicMJ] motion body names/order:", self.cfg.body_names)
        print("[SonicMJ] action term joint order should follow SONIC MuJoCo actuator order.")


@dataclass(kw_only=True)
class SonicMotionCommandCfg(CommandTermCfg):
    motion_file: str = ""
    smpl_motion_file: str | None = "dummy"
    motion_lib_cfg: dict | None = None
    entity_name: str = "robot"
    anchor_body: str = "pelvis"
    body_names: tuple[str, ...] = ()
    reward_point_body: tuple[str, ...] = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    reward_point_body_offset: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    vr_3point_body: tuple[str, ...] = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    vr_3point_body_offset: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    num_future_frames: int = 10
    dt_future_ref_frames: float = 0.1
    smpl_num_future_frames: int = 10
    smpl_dt_future_ref_frames: float = 0.02
    max_num_load_motions: int = 1024
    cat_upper_body_poses: bool = True
    cat_upper_body_poses_prob: float = 0.5
    randomize_heading: bool = False
    freeze_frame_aug: bool = False
    freeze_frame_aug_prob: float = 0.0
    encoder_sampling_mode: Literal["random", "cycle"] = "random"

    def build(self, env) -> SonicMotionCommand:
        return SonicMotionCommand(self, env)
