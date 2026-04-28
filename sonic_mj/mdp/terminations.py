from __future__ import annotations

from typing import cast

import torch

from mjlab.envs.mdp.terminations import time_out
from mjlab.utils.lab_api.math import quat_error_magnitude

from sonic_mj.mdp.commands import SonicMotionCommand


def _motion(env, command_name: str) -> SonicMotionCommand:
    return cast(SonicMotionCommand, env.command_manager.get_term(command_name))


def anchor_pos(env, command_name: str = "motion", threshold: float = 1.0) -> torch.Tensor:
    cmd = _motion(env, command_name)
    return torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1) > threshold


def anchor_ori_full(env, command_name: str = "motion", threshold: float = 1.57) -> torch.Tensor:
    cmd = _motion(env, command_name)
    return quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) > threshold


def ee_body_pos(env, command_name: str = "motion", threshold: float = 1.0) -> torch.Tensor:
    cmd = _motion(env, command_name)
    names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ids = [cmd.cfg.body_names.index(name) for name in names if name in cmd.cfg.body_names]
    err = torch.norm(cmd.body_pos_relative_w[:, ids] - cmd.robot_body_pos_w[:, ids], dim=-1)
    return torch.any(err > threshold, dim=-1)


def foot_pos_xyz(env, command_name: str = "motion", threshold: float = 1.0) -> torch.Tensor:
    cmd = _motion(env, command_name)
    names = ("left_ankle_roll_link", "right_ankle_roll_link")
    ids = [cmd.cfg.body_names.index(name) for name in names if name in cmd.cfg.body_names]
    err = torch.norm(cmd.body_pos_relative_w[:, ids] - cmd.robot_body_pos_w[:, ids], dim=-1)
    return torch.any(err > threshold, dim=-1)

