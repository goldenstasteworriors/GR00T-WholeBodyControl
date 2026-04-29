from __future__ import annotations

from typing import cast

import torch

from mjlab.envs import mdp as mj_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from sonic_mj.mdp.commands import SonicMotionCommand


def _motion(env, command_name: str) -> SonicMotionCommand:
    return cast(SonicMotionCommand, env.command_manager.get_term(command_name))


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
    target = cmd.body_pos_relative_w[:, cmd.reward_point_body_indices_motion]
    current = cmd.robot_body_pos_w[:, cmd.reward_point_body_indices]
    err = torch.sum(torch.square(target - current), dim=-1)
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
