from __future__ import annotations

from dataclasses import dataclass, field
import glob
import math
import os
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

from sonic_mj.assets import (
    G1_ISAACLAB_JOINTS,
    SONIC_G1_BODY_NAMES,
    SONIC_G1_JOINT_NAMES,
    SONIC_G1_MOTION_DOF_TO_MUJOCO,
)
from gear_sonic.utils.motion_lib import motion_lib_robot


def _as_tensor(value, *, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _apply_offset(quat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat, offset.expand(*quat.shape[:-1], 3))


def _init_variable_frames(
    enabled: bool,
    min_frames: int,
    num_future_frames: int,
    step: int,
    num_envs: int,
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not enabled:
        return None, None
    if step <= 0:
        raise ValueError(f"variable_frames_step must be positive, got {step}.")
    frame_choices = torch.arange(min_frames, num_future_frames + 1, step, device=device)
    if len(frame_choices) == 0:
        raise ValueError(
            "No valid variable frame choices: "
            f"variable_frames_min={min_frames}, num_future_frames={num_future_frames}."
        )
    per_env_num_frames = torch.full(
        (num_envs,), num_future_frames, device=device, dtype=torch.long
    )
    return per_env_num_frames, frame_choices


def _contact_key_aliases(key: str) -> tuple[str, ...]:
    key = str(key)
    normalized = key.replace("\\", "/")
    basename = os.path.basename(normalized)
    stem = os.path.splitext(basename)[0]
    aliases = (key, normalized, basename, stem)
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


class SonicMotionCommand(CommandTerm):
    cfg: SonicMotionCommandCfg

    def __init__(self, cfg: SonicMotionCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.cmd_body_names = cfg.body_names
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
        self.vr_3point_body_indices = [
            self.robot.body_names.index(name) for name in cfg.vr_3point_body
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
        self.motion_target_fps = int(motion_cfg.target_fps)
        if cfg.use_paired_motions:
            self.max_num_load_motions = self.motion_lib._num_unique_motions
        else:
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
        self.soma_encoder_index = (
            self.encoder_names.index("soma") if "soma" in self.encoder_names else None
        )
        no_smpl_probs = self.encoder_sample_probs.clone()
        if self.smpl_encoder_index is not None:
            no_smpl_probs[self.smpl_encoder_index] = 0.0
        if no_smpl_probs.sum() <= 0.0:
            no_smpl_probs[:] = 0.0
            if self.g1_encoder_index is not None:
                no_smpl_probs[self.g1_encoder_index] = 1.0
        self.encoder_sample_probs_no_smpl = no_smpl_probs / no_smpl_probs.sum()
        no_soma_probs = self.encoder_sample_probs.clone()
        if self.soma_encoder_index is not None:
            no_soma_probs[self.soma_encoder_index] = 0.0
        if no_soma_probs.sum() <= 0.0:
            no_soma_probs[:] = 0.0
            if self.g1_encoder_index is not None:
                no_soma_probs[self.g1_encoder_index] = 1.0
        self.encoder_sample_probs_no_soma = no_soma_probs / no_soma_probs.sum()
        no_smpl_no_soma_probs = no_smpl_probs.clone()
        if self.soma_encoder_index is not None:
            no_smpl_no_soma_probs[self.soma_encoder_index] = 0.0
        if no_smpl_no_soma_probs.sum() <= 0.0:
            no_smpl_no_soma_probs[:] = 0.0
            if self.g1_encoder_index is not None:
                no_smpl_no_soma_probs[self.g1_encoder_index] = 1.0
        self.encoder_sample_probs_no_smpl_no_soma = (
            no_smpl_no_soma_probs / no_smpl_no_soma_probs.sum()
        )

        self.motion_ids = self._sample_initial_motion_ids()
        self.motion_start_time_steps = self.motion_lib.sample_time_steps(
            self.motion_ids, truncate_time=None
        )
        self._load_contact_data()
        self._apply_start_time_overrides(torch.arange(self.num_envs, device=self.device))
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self.per_env_num_frames, self._frame_choices = _init_variable_frames(
            bool(cfg.variable_frames_enabled),
            int(cfg.variable_frames_min),
            int(cfg.num_future_frames),
            int(cfg.variable_frames_step),
            self.num_envs,
            self.device,
        )
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

    def _current_motion_steps(self) -> torch.Tensor:
        return self.motion_start_time_steps + self.time_steps

    def _require_motion_lib_field(self, field_name: str) -> None:
        if not hasattr(self.motion_lib, field_name):
            raise AttributeError(f"motion_lib does not provide {field_name}")

    @property
    def object_root_pos(self) -> torch.Tensor:
        self._require_motion_lib_field("_motion_object_root_pos")
        object_root_pos = self.motion_lib.get_object_root_pos(
            self.motion_ids, self._current_motion_steps()
        )
        if self.cfg.object_z_offset != 0.0:
            object_root_pos = object_root_pos.clone()
            object_root_pos[..., 2] += self.cfg.object_z_offset
        return object_root_pos + self._env.scene.env_origins[:, None, :]

    @property
    def object_root_quat(self) -> torch.Tensor:
        self._require_motion_lib_field("_motion_object_root_quat")
        return self.motion_lib.get_object_root_quat(self.motion_ids, self._current_motion_steps())

    @property
    def object_root_pos_multi_future(self) -> torch.Tensor:
        self._require_motion_lib_field("_motion_object_root_pos")
        num_frames = self.cfg.num_future_frames
        frame_offsets = torch.round(
            torch.arange(num_frames, device=self.device, dtype=torch.float32)
            * self.cfg.dt_future_ref_frames
            * self.motion_target_fps
        ).long()
        motion_steps = self._current_motion_steps()[:, None] + frame_offsets[None, :]
        flat_ids = self.motion_ids[:, None].expand(-1, num_frames).reshape(-1)
        flat_steps = motion_steps.reshape(-1)
        object_root_pos = self.motion_lib.get_object_root_pos(flat_ids, flat_steps)
        object_root_pos = object_root_pos.reshape(self.num_envs, num_frames, -1, 3)
        if self.cfg.object_z_offset != 0.0:
            object_root_pos = object_root_pos.clone()
            object_root_pos[..., 2] += self.cfg.object_z_offset
        return object_root_pos + self._env.scene.env_origins[:, None, None, :]

    @property
    def object_root_quat_multi_future(self) -> torch.Tensor:
        self._require_motion_lib_field("_motion_object_root_quat")
        num_frames = self.cfg.num_future_frames
        frame_offsets = torch.round(
            torch.arange(num_frames, device=self.device, dtype=torch.float32)
            * self.cfg.dt_future_ref_frames
            * self.motion_target_fps
        ).long()
        motion_steps = self._current_motion_steps()[:, None] + frame_offsets[None, :]
        flat_ids = self.motion_ids[:, None].expand(-1, num_frames).reshape(-1)
        flat_steps = motion_steps.reshape(-1)
        object_root_quat = self.motion_lib.get_object_root_quat(flat_ids, flat_steps)
        return object_root_quat.reshape(self.num_envs, num_frames, -1, 4)

    def _get_contact_center_world(self, hand: str) -> torch.Tensor | None:
        contact_center = self.motion_lib.get_object_contact_center(
            self.motion_ids, self._current_motion_steps(), hand=hand
        )
        if contact_center is None:
            return None
        object_root_pos = self.object_root_pos[:, 0]
        object_root_quat = self.object_root_quat[:, 0]
        return quat_apply(object_root_quat, contact_center) + object_root_pos

    @property
    def object_contact_center_left(self) -> torch.Tensor | None:
        return self._get_contact_center_world("left_hand")

    @property
    def object_contact_center_right(self) -> torch.Tensor | None:
        return self._get_contact_center_world("right_hand")

    def get_in_contact(self, hand: str = "right_hand") -> torch.Tensor | None:
        return self.motion_lib.get_object_in_contact(
            self.motion_ids, self._current_motion_steps(), hand=hand
        )

    @property
    def reward_point_body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.reward_point_body_indices_motion]

    @property
    def reward_point_body_pos_w(self) -> torch.Tensor:
        return (
            self.body_pos_w[:, self.reward_point_body_indices_motion]
            + _apply_offset(self.reward_point_body_quat_w, self.reward_point_body_offsets)
        )

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.vr_3point_body_indices_motion]

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        return (
            self.body_pos_w[:, self.vr_3point_body_indices_motion]
            + _apply_offset(self.vr_3point_body_quat_w, self.vr_3point_body_offsets)
        )

    @property
    def vr_3point_body_quat_w_multi_future(self) -> torch.Tensor:
        future = self.future_state()
        return future["body_quat_w"][:, :, self.vr_3point_body_indices_motion]

    @property
    def vr_3point_body_pos_w_multi_future(self) -> torch.Tensor:
        future = self.future_state()
        body_pos = (
            future["body_pos_w"][:, :, self.vr_3point_body_indices_motion]
            + self._env.scene.env_origins[:, None, None, :]
        )
        offsets = self.vr_3point_body_offsets[:, None].expand(
            self.num_envs,
            self.cfg.num_future_frames,
            len(self.cfg.vr_3point_body),
            3,
        )
        return body_pos + quat_apply(self.vr_3point_body_quat_w_multi_future, offsets)

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
    def robot_reward_point_body_pos_w(self) -> torch.Tensor:
        return (
            self.robot.data.body_link_pos_w[:, self.reward_point_body_indices]
            + _apply_offset(
                self.robot.data.body_link_quat_w[:, self.reward_point_body_indices],
                self.reward_point_body_offsets,
            )
        )

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return (
            self.robot.data.body_link_pos_w[:, self.vr_3point_body_indices]
            + _apply_offset(
                self.robot.data.body_link_quat_w[:, self.vr_3point_body_indices],
                self.vr_3point_body_offsets,
            )
        )

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

    @property
    def command_num_frames(self) -> torch.Tensor:
        if self.per_env_num_frames is None:
            return torch.full(
                (self.num_envs, 1),
                float(self.cfg.num_future_frames),
                device=self.device,
            )
        return self.per_env_num_frames.float().reshape(-1, 1)

    def soma_future_state(self) -> dict[str, torch.Tensor]:
        num_frames = self.cfg.smpl_num_future_frames
        frame_offsets = torch.round(
            torch.arange(num_frames, device=self.device, dtype=torch.float32)
            * self.cfg.smpl_dt_future_ref_frames
            * self.motion_target_fps
        ).long()
        motion_steps = (
            self.motion_start_time_steps[:, None] + self.time_steps[:, None] + frame_offsets[None, :]
        )
        flat_ids = self.motion_ids[:, None].expand(-1, num_frames).reshape(-1)
        flat_steps = motion_steps.reshape(-1)

        if hasattr(self.motion_lib, "get_soma_joints") and hasattr(self.motion_lib, "_motion_soma_joints"):
            joints = self.motion_lib.get_soma_joints(flat_ids, flat_steps).reshape(
                self.num_envs, num_frames, -1, 3
            )
        else:
            joints = torch.zeros(
                self.num_envs,
                num_frames,
                self.cfg.num_soma_joints,
                3,
                device=self.device,
            )

        if hasattr(self.motion_lib, "get_soma_root_quat") and hasattr(
            self.motion_lib, "_motion_soma_root_quat"
        ):
            root_quat = self.motion_lib.get_soma_root_quat(flat_ids, flat_steps)
            if root_quat.ndim > 2:
                root_quat = root_quat.reshape(root_quat.shape[0], -1, root_quat.shape[-1])[:, 0]
            root_quat = root_quat.reshape(self.num_envs, num_frames, 4)
        else:
            root_quat = torch.zeros(
                self.num_envs, num_frames, 4, device=self.device
            )
            root_quat[..., 0] = 1.0
        root_quat = self._soma_root_quat_to_zup_wxyz(root_quat.reshape(-1, 4)).reshape(
            self.num_envs, num_frames, 4
        )
        return {"soma_joints": joints, "soma_root_quat_w": root_quat}

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

    def _soma_root_quat_to_zup_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.motion_lib, "soma_y_up", True)):
            base_rot = torch.tensor(
                [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
                dtype=quat.dtype,
                device=quat.device,
            ).expand_as(quat)
            quat = quat_mul(base_rot, quat)
        bvh_base_rot = torch.tensor(
            [0.5, 0.5, 0.5, 0.5], dtype=quat.dtype, device=quat.device
        ).expand_as(quat)
        return quat_mul(quat, bvh_base_rot)

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
        if self.cfg.sample_before_contact and self._first_contact_lookup is not None:
            self.motion_start_time_steps[env_ids] = self._sample_before_contact(
                env_ids, self.motion_start_time_steps[env_ids]
            )

    def _load_contact_data(self) -> None:
        self._first_contact_frame = None
        self._contact_data = None
        self._contact_key_to_motion_key = {}
        self._contact_diagnostics = {
            "source": self.cfg.contact_file,
            "loaded": 0,
            "matched": 0,
            "unmatched_contact_keys": (),
            "missing_motion_keys": (),
        }
        self._first_contact_lookup = None
        self._per_env_first_contact = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._motion_contact_flags = None
        if self.cfg.contact_file is None or not os.path.exists(self.cfg.contact_file):
            self._derive_first_contact_from_in_contact_labels()
            return

        import joblib

        if os.path.isfile(self.cfg.contact_file):
            contact_data = joblib.load(self.cfg.contact_file)
            if isinstance(contact_data, dict) and (
                "object" in contact_data or "body" in contact_data
            ):
                key = os.path.splitext(os.path.basename(self.cfg.contact_file))[0]
                contact_data = {key: contact_data}
        elif os.path.isdir(self.cfg.contact_file):
            contact_data = {}
            for pkl_file in sorted(glob.glob(os.path.join(self.cfg.contact_file, "*.pkl"))):
                key = os.path.splitext(os.path.basename(pkl_file))[0]
                data = joblib.load(pkl_file)
                if isinstance(data, dict) and key in data:
                    contact_data[key] = data[key]
                elif isinstance(data, dict) and ("object" in data or "body" in data):
                    contact_data[key] = data
        else:
            return

        motion_keys = tuple(getattr(self.motion_lib, "curr_motion_keys", []))
        contact_data, key_map, unmatched = self._match_contact_data_to_motion_keys(
            contact_data, motion_keys
        )
        missing = tuple(motion_key for motion_key in motion_keys if motion_key not in contact_data)
        self._contact_key_to_motion_key = key_map
        self._contact_diagnostics = {
            "source": self.cfg.contact_file,
            "loaded": len(key_map) + len(unmatched),
            "matched": len(contact_data),
            "unmatched_contact_keys": unmatched,
            "missing_motion_keys": missing,
        }
        self._contact_data = contact_data
        first_contact_frame = {}
        for motion_name, motion_data in contact_data.items():
            object_contact = motion_data.get("object") if isinstance(motion_data, dict) else None
            if object_contact is None:
                first_contact_frame[motion_name] = 0
                continue
            contact_tensor = torch.as_tensor(object_contact)
            contact_per_frame = (contact_tensor != 0).reshape(contact_tensor.shape[0], -1).any(dim=1)
            contact_frames = torch.nonzero(contact_per_frame, as_tuple=False).flatten()
            first_contact_frame[motion_name] = (
                int(contact_frames[0].item()) if len(contact_frames) > 0 else contact_tensor.shape[0]
            )
        self._first_contact_frame = first_contact_frame or None
        self._validate_contact_frame_counts(contact_data)
        self._build_first_contact_lookup()
        self._build_motion_contact_flags()

    def _match_contact_data_to_motion_keys(
        self, contact_data: dict, motion_keys: tuple[str, ...]
    ) -> tuple[dict, dict, tuple[str, ...]]:
        if not motion_keys:
            return contact_data, {key: key for key in contact_data}, ()

        alias_to_motion_keys = {}
        for motion_key in motion_keys:
            for alias in _contact_key_aliases(motion_key):
                alias_to_motion_keys.setdefault(alias, []).append(motion_key)

        matched = {}
        key_map = {}
        unmatched = []
        for contact_key, motion_data in contact_data.items():
            candidates = []
            for alias in _contact_key_aliases(contact_key):
                candidates.extend(alias_to_motion_keys.get(alias, []))
            candidates = list(dict.fromkeys(candidates))

            if not candidates:
                contact_aliases = _contact_key_aliases(contact_key)
                for motion_key in motion_keys:
                    motion_aliases = _contact_key_aliases(motion_key)
                    if any(
                        contact_alias in motion_alias or motion_alias in contact_alias
                        for contact_alias in contact_aliases
                        for motion_alias in motion_aliases
                    ):
                        candidates.append(motion_key)
                candidates = list(dict.fromkeys(candidates))

            if len(candidates) == 1:
                motion_key = candidates[0]
                matched[motion_key] = motion_data
                key_map[contact_key] = motion_key
            else:
                unmatched.append(str(contact_key))

        return matched, key_map, tuple(unmatched)

    def _derive_first_contact_from_in_contact_labels(self) -> None:
        hand = self.cfg.sample_before_contact_hand
        side = "left" if hand == "left_hand" else "right"
        attr = f"_motion_object_in_contact_{side}"
        if not hasattr(self.motion_lib, attr):
            return

        in_contact = getattr(self.motion_lib, attr)
        length_starts = getattr(self.motion_lib, "length_starts", None)
        motion_num_frames = getattr(self.motion_lib, "_motion_num_frames", None)
        motion_keys = getattr(self.motion_lib, "curr_motion_keys", None)
        if length_starts is None or motion_num_frames is None or motion_keys is None:
            return

        first_contact_frame = {}
        for motion_idx, motion_key in enumerate(motion_keys):
            start = int(length_starts[motion_idx].item())
            num_frames = int(motion_num_frames[motion_idx].item())
            motion_contact = in_contact[start : start + num_frames]
            contact_frames = torch.nonzero(motion_contact > 0.5, as_tuple=False).flatten()
            first_contact_frame[motion_key] = (
                int(contact_frames[0].item()) if len(contact_frames) > 0 else num_frames
            )
        self._contact_data = None
        self._contact_key_to_motion_key = {key: key for key in first_contact_frame}
        self._contact_diagnostics = {
            "source": f"motion_lib.{attr}",
            "loaded": len(first_contact_frame),
            "matched": len(first_contact_frame),
            "unmatched_contact_keys": (),
            "missing_motion_keys": (),
        }
        self._first_contact_frame = first_contact_frame or None
        self._build_first_contact_lookup()

    def _build_first_contact_lookup(self) -> None:
        if self._first_contact_frame is None:
            self._first_contact_lookup = None
            return
        motion_keys = getattr(self.motion_lib, "curr_motion_keys", [])
        self._first_contact_lookup = torch.zeros(
            len(motion_keys), device=self.device, dtype=torch.long
        )
        for motion_idx, motion_key in enumerate(motion_keys):
            self._first_contact_lookup[motion_idx] = int(
                self._first_contact_frame.get(motion_key, 0)
            )
        self._update_per_env_first_contact(torch.arange(self.num_envs, device=self.device))

    def _build_motion_contact_flags(self) -> None:
        if not self._contact_data:
            self._motion_contact_flags = None
            return
        self._motion_contact_flags = {}
        for motion_idx, motion_key in enumerate(getattr(self.motion_lib, "curr_motion_keys", [])):
            motion_data = self._contact_data.get(motion_key)
            object_contact = motion_data.get("object") if isinstance(motion_data, dict) else None
            if object_contact is None:
                continue
            contact_tensor = torch.as_tensor(object_contact, device=self.device)
            contact_per_frame = (contact_tensor != 0).reshape(contact_tensor.shape[0], -1).any(dim=1)
            self._motion_contact_flags[motion_idx] = contact_per_frame.bool()

    def refresh_after_motion_lib_reload(self) -> None:
        self.use_adaptive_sampling = bool(getattr(self.motion_lib, "use_adaptive_sampling", False))
        num_loaded_motions = int(getattr(self.motion_lib, "_num_motions", 0))
        if num_loaded_motions <= 0:
            raise RuntimeError("[SonicMJ] Motion lib reload left no motions loaded.")
        self.motion_ids.remainder_(num_loaded_motions)
        self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
        self._load_contact_data()
        self._update_per_env_first_contact(torch.arange(self.num_envs, device=self.device))
        self._refresh_motion_state()

    def _update_per_env_first_contact(self, env_ids: torch.Tensor) -> None:
        if self._first_contact_lookup is None or len(env_ids) == 0:
            return
        self._per_env_first_contact[env_ids] = self._first_contact_lookup[self.motion_ids[env_ids]]

    def _validate_contact_frame_counts(self, contact_data: dict) -> None:
        frame_tolerance = int(self.cfg.contact_frame_tolerance)
        motion_keys = list(getattr(self.motion_lib, "curr_motion_keys", []))
        motion_num_frames = getattr(self.motion_lib, "_motion_num_frames", None)
        if motion_num_frames is None:
            return
        for motion_name, motion_data in contact_data.items():
            if not isinstance(motion_data, dict):
                continue
            contact_source = motion_data.get("object", None)
            if contact_source is None:
                contact_source = motion_data.get("body", None)
            if contact_source is None or motion_name not in motion_keys:
                continue
            contact_frames = int(torch.as_tensor(contact_source).shape[0])
            motion_idx = motion_keys.index(motion_name)
            expected_frames = int(motion_num_frames[motion_idx].item())
            if abs(contact_frames - expected_frames) > frame_tolerance:
                raise AssertionError(
                    "[SonicMJ] Contact frame count mismatch for "
                    f"{motion_name}: contact={contact_frames}, motion={expected_frames}, "
                    f"tolerance={frame_tolerance}."
                )

    def _sample_before_contact(
        self, env_ids: torch.Tensor, sampled_times: torch.Tensor
    ) -> torch.Tensor:
        if self._first_contact_lookup is None:
            return sampled_times
        first_contact = self._first_contact_lookup[self.motion_ids[env_ids]]
        upper = first_contact - int(self.cfg.sample_before_contact_margin)
        valid = upper > 0
        if valid.any():
            rand = torch.rand(len(env_ids), device=self.device)
            sampled_times = sampled_times.clone()
            sampled_times[valid] = torch.floor(rand[valid] * upper[valid].float()).to(
                sampled_times.dtype
            )
        sampled_times[~valid] = 0
        return sampled_times

    def _sample_encoder_index(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.encoder_index[env_ids] = 0
        has_smpl = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if self.smpl_encoder_index is not None and hasattr(self.motion_lib, "motion_has_smpl"):
            has_smpl = self.motion_lib.motion_has_smpl[self.motion_ids[env_ids]].bool()
        has_soma = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if self.soma_encoder_index is not None and hasattr(self.motion_lib, "motion_has_soma"):
            has_soma = self.motion_lib.motion_has_soma[self.motion_ids[env_ids]].bool()

        if self.soma_encoder_index is not None:
            sampling_cases = (
                (env_ids[has_smpl & has_soma], self.encoder_sample_probs),
                (env_ids[has_smpl & ~has_soma], self.encoder_sample_probs_no_soma),
                (env_ids[~has_smpl & has_soma], self.encoder_sample_probs_no_smpl),
                (env_ids[~has_smpl & ~has_soma], self.encoder_sample_probs_no_smpl_no_soma),
            )
        else:
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
        if self.g1_encoder_index is not None and self.soma_encoder_index is not None:
            use_soma = self.encoder_index[env_ids, self.soma_encoder_index].bool()
            self.encoder_index[env_ids[use_soma], self.g1_encoder_index] = 1

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
        if self.per_env_num_frames is not None and len(env_ids) > 0:
            idx = torch.randint(0, len(self._frame_choices), (len(env_ids),), device=self.device)
            self.per_env_num_frames[env_ids] = self._frame_choices[idx]
        self.time_steps[env_ids] = 0
        self.motion_num_steps[env_ids] = self.motion_lib.get_motion_num_steps(self.motion_ids[env_ids])
        self._update_per_env_first_contact(env_ids)
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
        robot_joint_names = tuple(self.robot.joint_names)
        robot_body_names = tuple(self.robot.body_names)
        motion_body_names = tuple(self.cfg.body_names)
        action_joint_names = ()
        if hasattr(self._env, "action_manager"):
            try:
                action_joint_names = tuple(self._env.action_manager.get_term("joint_pos").target_names)
            except Exception:
                action_joint_names = ()

        print("[SonicMJ] robot joint names/order:", robot_joint_names)
        print("[SonicMJ] robot body names/order:", robot_body_names)
        print("[SonicMJ] motion body names/order:", motion_body_names)
        print("[SonicMJ] action term joint names/order:", action_joint_names)
        print("[SonicMJ] motion DOF -> MuJoCo mapping:", tuple(SONIC_G1_MOTION_DOF_TO_MUJOCO))
        order_checks = {
            "robot_joints_match_sonic_mujoco": robot_joint_names == SONIC_G1_JOINT_NAMES,
            "robot_bodies_match_sonic_mujoco": robot_body_names == SONIC_G1_BODY_NAMES,
            "motion_bodies_match_sonic_mujoco": motion_body_names == SONIC_G1_BODY_NAMES,
            "motion_dof_mapping_identity": tuple(SONIC_G1_MOTION_DOF_TO_MUJOCO)
            == tuple(range(len(SONIC_G1_JOINT_NAMES))),
        }
        if action_joint_names:
            order_checks["action_joints_match_sonic_mujoco"] = (
                action_joint_names == SONIC_G1_JOINT_NAMES
            )
        else:
            order_checks["action_joints_match_sonic_mujoco"] = "pending_action_manager_init"
        print(
            "[SonicMJ] order checks:",
            order_checks,
        )

    def set_is_evaluating(self, is_evaluating: bool = True) -> None:
        self.is_evaluating = is_evaluating

    def get_contact_diagnostics(self) -> dict:
        return dict(getattr(self, "_contact_diagnostics", {}))


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
    variable_frames_enabled: bool = False
    variable_frames_min: int = 16
    variable_frames_step: int = 4
    num_soma_joints: int = 26
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
    contact_file: str | None = None
    sample_before_contact: bool = False
    sample_before_contact_margin: int = 10
    sample_before_contact_hand: str = "right_hand"
    contact_frame_tolerance: int = 3
    object_position_randomize: bool = False
    object_position_randomization: dict[str, float] | None = None
    object_z_offset: float = 0.0
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.1, 0.1)
    joint_velocity_range: tuple[float, float] = (0.0, 0.0)

    def build(self, env) -> SonicMotionCommand:
        return SonicMotionCommand(self, env)
