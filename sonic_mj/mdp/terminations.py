from __future__ import annotations

from typing import cast

import torch

from mjlab.envs.mdp.terminations import time_out
from mjlab.utils.lab_api.math import quat_error_magnitude

from sonic_mj.mdp.commands import SonicMotionCommand


def _motion(env, command_name: str) -> SonicMotionCommand:
    return cast(SonicMotionCommand, env.command_manager.get_term(command_name))


def _body_indexes(cmd: SonicMotionCommand, body_names: list[str] | tuple[str, ...] | None):
    return [
        idx
        for idx, body_name in enumerate(cmd.cfg.body_names)
        if body_names is None or body_name in body_names
    ]


def anchor_pos(
    env,
    command_name: str = "motion",
    threshold: float = 0.5,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.75,
    root_height_threshold: float = 0.5,
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    height_diff = torch.abs(cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2])
    if threshold_adaptive:
        threshold_tensor = torch.full_like(height_diff, threshold)
        threshold_tensor[cmd.running_ref_root_height < root_height_threshold] = down_threshold
        return height_diff > threshold_tensor
    return height_diff > threshold


def anchor_ori_full(env, command_name: str = "motion", threshold: float = 1.0) -> torch.Tensor:
    cmd = _motion(env, command_name)
    return quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w).square() > threshold


def ee_body_pos(
    env,
    command_name: str = "motion",
    threshold: float = 0.5,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.75,
    root_height_threshold: float = 0.5,
    body_names: list[str] | tuple[str, ...] | None = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    ids = _body_indexes(cmd, body_names)
    height_err = torch.abs(cmd.body_pos_relative_w[:, ids, 2] - cmd.robot_body_pos_w[:, ids, 2])
    if threshold_adaptive:
        threshold_tensor = torch.full_like(height_err, threshold)
        threshold_tensor[cmd.running_ref_root_height < root_height_threshold] = down_threshold
        return torch.any(height_err > threshold_tensor, dim=-1)
    return torch.any(height_err > threshold, dim=-1)


def foot_pos_xyz(
    env,
    command_name: str = "motion",
    threshold: float = 0.5,
    body_names: list[str] | tuple[str, ...] | None = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
) -> torch.Tensor:
    cmd = _motion(env, command_name)
    ids = _body_indexes(cmd, body_names)
    err = torch.norm(cmd.body_pos_relative_w[:, ids] - cmd.robot_body_pos_w[:, ids], dim=-1)
    return torch.any(err > threshold, dim=-1)
