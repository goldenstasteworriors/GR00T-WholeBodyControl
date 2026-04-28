from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import easydict
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from sonic_mj.assets import G1_ISAACLAB_JOINTS, SONIC_G1_MOTION_DOF_TO_MUJOCO
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
        self.use_adaptive_sampling = bool(getattr(self.motion_lib, "use_adaptive_sampling", False))
        self.is_evaluating = False

        self.encoder_sample_probs_dict = cfg.encoder_sample_probs or {
            "g1": 1.0,
            "teleop": 1.0,
            "smpl": 1.0,
        }
        self.encoder_names = tuple(self.encoder_sample_probs_dict.keys())
        self.encoder_sample_probs = torch.tensor(
            list(self.encoder_sample_probs_dict.values()), dtype=torch.float32, device=self.device
        )
        self.encoder_sample_probs = self.encoder_sample_probs / self.encoder_sample_probs.sum()
        self.g1_encoder_index = self.encoder_names.index("g1") if "g1" in self.encoder_names else None
        self.smpl_encoder_index = (
            self.encoder_names.index("smpl") if "smpl" in self.encoder_names else None
        )
        self.teleop_encoder_index = (
            self.encoder_names.index("teleop") if "teleop" in self.encoder_names else None
        )
        no_smpl_probs = self.encoder_sample_probs.clone()
        if self.smpl_encoder_index is not None:
            no_smpl_probs[self.smpl_encoder_index] = 0.0
        if no_smpl_probs.sum() <= 0.0:
            no_smpl_probs[:] = 0.0
            if self.g1_encoder_index is not None:
                no_smpl_probs[self.g1_encoder_index] = 1.0
        self.encoder_sample_probs_no_smpl = no_smpl_probs / no_smpl_probs.sum()

        self.motion_ids = self._sample_initial_motion_ids()
        self.motion_start_time_steps = self.motion_lib.sample_time_steps(
            self.motion_ids, truncate_time=None
        )
        self._apply_start_time_overrides(torch.arange(self.num_envs, device=self.device))
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self.encoder_index = torch.zeros(
            self.num_envs, len(self.encoder_names), dtype=torch.long, device=self.device
        )
        self._body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self._body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self._motion_to_robot_dof = torch.tensor(
            SONIC_G1_MOTION_DOF_TO_MUJOCO, dtype=torch.long, device=self.device
        )
        self._body_quat_relative_w[..., 0] = 1.0
        self._sample_encoder_index(torch.arange(self.num_envs, device=self.device))
        self._refresh_motion_state()
        self.running_ref_root_height = self.anchor_pos_w[:, 2].clone()
        self._print_order_summary()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_state["dof_pos"][:, self._motion_to_robot_dof]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_state["dof_vel"][:, self._motion_to_robot_dof]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._motion_state["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._xyzw_to_wxyz(self._motion_state["body_quat_w"])

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
        state["dof_pos"] = state["dof_pos"][:, self._motion_to_robot_dof]
        state["dof_vel"] = state["dof_vel"][:, self._motion_to_robot_dof]
        state["body_quat_w"] = self._xyzw_to_wxyz(state["body_quat_w"])
        state["root_rot"] = self._xyzw_to_wxyz(state["root_rot"])
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

    @staticmethod
    def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
        return quat[..., [3, 0, 1, 2]]

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )

    def _sample_initial_motion_ids(self) -> torch.Tensor:
        if self.cfg.use_paired_motions:
            return torch.arange(self.num_envs, device=self.device) % self.motion_lib._num_motions
        if self.cfg.sample_unique_motions:
            num_available = len(self.motion_lib._curr_motion_ids)
            if self.num_envs > num_available:
                raise ValueError(
                    "sample_unique_motions=True requires num_envs "
                    f"({self.num_envs}) <= num_available_motions ({num_available})"
                )
            return torch.randperm(num_available, device=self.device)[: self.num_envs]
        return self.motion_lib.sample_motions(self.num_envs)

    def _apply_start_time_overrides(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sample_from_n_initial_frames is not None:
            self.motion_start_time_steps[env_ids] = torch.randint(
                0,
                int(self.cfg.sample_from_n_initial_frames),
                (len(env_ids),),
                dtype=self.motion_start_time_steps.dtype,
                device=self.device,
            )
        elif self.cfg.start_from_first_frame:
            self.motion_start_time_steps[env_ids] = 0

    def _sample_encoder_index(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.encoder_index[env_ids] = 0
        has_smpl = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if self.smpl_encoder_index is not None and hasattr(self.motion_lib, "motion_has_smpl"):
            has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]].bool()

        sampling_cases = (
            (env_ids[has_smpl], self.encoder_sample_probs),
            (env_ids[~has_smpl], self.encoder_sample_probs_no_smpl),
        )
        for subset_ids, probs in sampling_cases:
            if len(subset_ids) == 0:
                continue
            sampled = torch.multinomial(probs, len(subset_ids), replacement=True).to(self.device)
            self.encoder_index[subset_ids, sampled] = 1

        if self.g1_encoder_index is not None and self.smpl_encoder_index is not None:
            use_smpl = self.encoder_index[env_ids, self.smpl_encoder_index].bool()
            self.encoder_index[env_ids[use_smpl], self.g1_encoder_index] = 1
            if (
                self.teleop_encoder_index is not None
                and self.cfg.teleop_sample_prob_when_smpl > 0.0
                and use_smpl.any()
            ):
                smpl_env_ids = env_ids[use_smpl]
                use_teleop = (
                    torch.rand(len(smpl_env_ids), device=self.device)
                    < self.cfg.teleop_sample_prob_when_smpl
                )
                self.encoder_index[smpl_env_ids[use_teleop], self.teleop_encoder_index] = 1

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.is_evaluating:
            self.motion_ids[env_ids] = (
                torch.arange(self.num_envs, device=self.device) % self.motion_lib._num_motions
            )[env_ids]
            self.motion_start_time_steps[env_ids] = 0
        elif self.cfg.use_paired_motions:
            self.motion_ids[env_ids] = (
                torch.arange(self.num_envs, device=self.device) % self.motion_lib._num_motions
            )[env_ids]
            self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
                self.motion_ids[env_ids], truncate_time=None
            )
        elif self.use_adaptive_sampling:
            sampled_ids, sampled_times = self.motion_lib.sample_motion_ids_and_time_steps(
                len(env_ids)
            )
            self.motion_ids[env_ids] = sampled_ids.to(self.motion_ids.dtype)
            self.motion_start_time_steps[env_ids] = sampled_times.to(
                self.motion_start_time_steps.dtype
            )
        else:
            self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
            self.motion_start_time_steps[env_ids] = self.motion_lib.sample_time_steps(
                self.motion_ids[env_ids], truncate_time=None
            )
        if not self.is_evaluating:
            self._apply_start_time_overrides(env_ids)
        self.time_steps[env_ids] = 0
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(self.motion_ids[env_ids])
        if self.cfg.encoder_sampling_mode == "cycle":
            self.encoder_index[env_ids] = 0
            sampled = self.command_counter[env_ids].long() % len(self.encoder_names)
            self.encoder_index[env_ids, sampled] = 1
        else:
            self._sample_encoder_index(env_ids)
        self._refresh_motion_state()
        self.running_ref_root_height[env_ids] = self.anchor_pos_w[env_ids, 2]
        root_pos = self._motion_state["root_pos"].clone() + self._env.scene.env_origins
        root_quat = self._xyzw_to_wxyz(self._motion_state["root_rot"]).clone()
        root_lin_vel = self._motion_state["root_vel"].clone()
        root_ang_vel = self._motion_state["root_ang_vel"].clone()
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        if not self.is_evaluating:
            pose_ranges = torch.tensor(
                [
                    self.cfg.pose_range.get(key, (0.0, 0.0))
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                dtype=torch.float32,
                device=self.device,
            )
            pose_rand = sample_uniform(
                pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
            )
            root_pos[env_ids] += pose_rand[:, :3]
            root_quat[env_ids] = quat_mul(
                quat_from_euler_xyz(pose_rand[:, 3], pose_rand[:, 4], pose_rand[:, 5]),
                root_quat[env_ids],
            )
            velocity_ranges = torch.tensor(
                [
                    self.cfg.velocity_range.get(key, (0.0, 0.0))
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                dtype=torch.float32,
                device=self.device,
            )
            velocity_rand = sample_uniform(
                velocity_ranges[:, 0],
                velocity_ranges[:, 1],
                (len(env_ids), 6),
                device=self.device,
            )
            root_lin_vel[env_ids] += velocity_rand[:, :3]
            root_ang_vel[env_ids] += velocity_rand[:, 3:]
            joint_pos[env_ids] += sample_uniform(
                *self.cfg.joint_position_range,
                (len(env_ids), joint_pos.shape[1]),
                device=self.device,
            )
            joint_vel[env_ids] += sample_uniform(
                *self.cfg.joint_velocity_range,
                (len(env_ids), joint_vel.shape[1]),
                device=self.device,
            )
        root_state = torch.cat(
            [
                root_pos[env_ids],
                root_quat[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.reset(env_ids=env_ids)

    def sync_after_sim_forward(self) -> None:
        self._refresh_motion_state()

    def _update_command(self) -> None:
        if self.use_adaptive_sampling:
            cur_time_steps = self.motion_start_time_steps + self.time_steps
            self.motion_lib.update_adaptive_sampling(
                self._env.reset_terminated, self.motion_ids, cur_time_steps
            )
        self.time_steps += 1
        done = torch.where(
            self.time_steps + self.motion_start_time_steps
            >= self.motion_lib.get_time_step_total(self.motion_ids)
        )[0]
        if done.numel() > 0:
            self._resample_command(done)
        self._refresh_motion_state()
        ema_alpha = 0.1
        self.running_ref_root_height = (
            ema_alpha * self.anchor_pos_w[:, 2]
            + (1.0 - ema_alpha) * self.running_ref_root_height
        )

    def _print_order_summary(self) -> None:
        print("[SonicMJ] robot joint names/order:", self.robot.joint_names)
        print("[SonicMJ] robot body names/order:", self.robot.body_names)
        print("[SonicMJ] motion body names/order:", self.cfg.body_names)
        print("[SonicMJ] action term joint order should follow SONIC MuJoCo actuator order.")

    def set_is_evaluating(self, is_evaluating: bool = True) -> None:
        self.is_evaluating = is_evaluating


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
    encoder_sample_probs: dict[str, float] | None = None
    teleop_sample_prob_when_smpl: float = 0.0
    start_from_first_frame: bool = False
    sample_unique_motions: bool = False
    use_paired_motions: bool = False
    sample_from_n_initial_frames: int | None = None
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    joint_velocity_range: tuple[float, float] = (0.0, 0.0)

    def build(self, env) -> SonicMotionCommand:
        return SonicMotionCommand(self, env)
