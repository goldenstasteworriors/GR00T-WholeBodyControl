from __future__ import annotations

from typing import cast

import torch

from mjlab.envs import mdp as mj_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude, quat_inv

from sonic_mj.mdp.commands import SonicMotionCommand


def _motion(env, command_name: str) -> SonicMotionCommand:
    return cast(SonicMotionCommand, env.command_manager.get_term(command_name))


def _body_indexes(cmd: SonicMotionCommand, body_names: list[str] | tuple[str, ...] | None):
    return [
        idx
        for idx, body_name in enumerate(cmd.cfg.body_names)
        if body_names is None or body_name in body_names
    ]


def tracking_anchor_pos(env, command_name: str = "motion", std: float = 0.3) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = torch.sum(torch.square(cmd.anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1)
    return torch.exp(-err / std**2)


def tracking_anchor_ori(env, command_name: str = "motion", std: float = 0.4) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-err / std**2)


def tracking_relative_body_pos(
    env, command_name: str = "motion", std: float = 0.3
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = torch.sum(torch.square(cmd.body_pos_relative_w - cmd.robot_body_pos_w), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def tracking_relative_body_ori(
    env, command_name: str = "motion", std: float = 0.4
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = quat_error_magnitude(cmd.body_quat_relative_w, cmd.robot_body_quat_w) ** 2
    return torch.exp(-err.mean(-1) / std**2)


def tracking_body_linvel(env, command_name: str = "motion", std: float = 1.0) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = torch.sum(torch.square(cmd.body_lin_vel_w - cmd.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def tracking_body_angvel(env, command_name: str = "motion", std: float = 3.14) -> torch.Tensor:
    cmd = _motion(env, command_name)
    err = torch.sum(torch.square(cmd.body_ang_vel_w - cmd.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def tracking_vr_5point_local(env, command_name: str = "motion", std: float = 0.3) -> torch.Tensor:
    cmd = _motion(env, command_name)
    ref_diff = cmd.reward_point_body_pos_w - cmd.anchor_pos_w[:, None, :]
    ref_root_quat = cmd.anchor_quat_w[:, None, :].repeat(1, len(cmd.cfg.reward_point_body), 1)
    ref_pos = quat_apply(quat_inv(ref_root_quat), ref_diff)
    robot_diff = cmd.robot_reward_point_body_pos_w - cmd.robot_anchor_pos_w[:, None, :]
    robot_root_quat = cmd.robot_anchor_quat_w[:, None, :].repeat(
        1, len(cmd.cfg.reward_point_body), 1
    )
    robot_pos = quat_apply(quat_inv(robot_root_quat), robot_diff)
    err = torch.sum(torch.square(robot_pos - ref_pos), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def action_rate_l2(env) -> torch.Tensor:
    return mj_mdp.action_rate_l2(env)


def joint_limit(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> torch.Tensor:
    return mj_mdp.joint_pos_limits(env, asset_cfg=asset_cfg)


def feet_acc(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*ankle.*",)),
) -> torch.Tensor:
    return mj_mdp.joint_acc_l2(env, asset_cfg=asset_cfg)


def anti_shake_ang_vel(
    env,
    command_name: str = "motion",
    threshold: float = 1.5,
    body_names: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    ids = _body_indexes(cmd, body_names)
    if not ids:
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.linalg.norm(cmd.robot_body_ang_vel_w[:, ids], dim=-1)
    excess = torch.relu(speed - threshold)
    return torch.square(excess).mean(dim=-1)


def undesired_contacts(
    env,
    sensor_name: str = "undesired_contacts",
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_data = env.scene[sensor_name].data
    if contact_data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_norm = torch.linalg.norm(contact_data.force, dim=-1)
    return torch.sum(force_norm > threshold, dim=1).to(dtype=force_norm.dtype)
