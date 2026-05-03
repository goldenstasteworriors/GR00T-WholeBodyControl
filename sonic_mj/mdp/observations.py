from __future__ import annotations

from typing import cast

import torch

from mjlab.envs import mdp as mj_mdp
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from sonic_mj.mdp.commands import SonicMotionCommand


def _motion(env, command_name: str) -> SonicMotionCommand:
    return cast(SonicMotionCommand, env.command_manager.get_term(command_name))


def gravity_dir(env) -> torch.Tensor:
    return mj_mdp.projected_gravity(env)


def base_lin_vel(env) -> torch.Tensor:
    return mj_mdp.base_lin_vel(env)


def base_ang_vel(env) -> torch.Tensor:
    return mj_mdp.base_ang_vel(env)


def joint_pos(env) -> torch.Tensor:
    return mj_mdp.joint_pos_rel(env)


def joint_vel(env) -> torch.Tensor:
    return mj_mdp.joint_vel_rel(env)


def actions(env) -> torch.Tensor:
    return mj_mdp.last_action(env)


def command(env, command_name: str = "motion") -> torch.Tensor:
    return _motion(env, command_name).command


def encoder_index(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    if cmd.encoder_index.ndim == 2 and cmd.encoder_index.shape[1] > 1:
        return cmd.encoder_index.float()
    out = torch.zeros(env.num_envs, 3, device=env.device)
    out.scatter_(1, cmd.encoder_index.clamp(0, 2), 1.0)
    return out


def motion_anchor_pos_b(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    pos, _ = subtract_frame_transforms(
        cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w, cmd.anchor_pos_w, cmd.anchor_quat_w
    )
    return pos


def motion_anchor_ori_b(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    _, ori = subtract_frame_transforms(
        cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w, cmd.anchor_pos_w, cmd.anchor_quat_w
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_anchor_ori_b_mf_nonflat(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state()
    robot_q = cmd.robot_anchor_quat_w[:, None, :].expand(-1, cmd.cfg.num_future_frames, -1)
    motion_q = future["body_quat_w"][:, :, cmd.motion_anchor_body_index]
    ori = quat_mul(quat_inv(robot_q), motion_q)
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, cmd.cfg.num_future_frames, -1)


def body_pos(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    n = len(cmd.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        cmd.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        cmd.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def body_ori(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    n = len(cmd.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        cmd.robot_anchor_pos_w[:, None, :].repeat(1, n, 1),
        cmd.robot_anchor_quat_w[:, None, :].repeat(1, n, 1),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def command_multi_future_nonflat(env, command_name: str = "motion") -> torch.Tensor:
    future = _motion(env, command_name).future_state()
    return torch.cat([future["dof_pos"], future["dof_vel"]], dim=-1)


def command_multi_future(env, command_name: str = "motion") -> torch.Tensor:
    return command_multi_future_nonflat(env, command_name).reshape(env.num_envs, -1)


def command_multi_future_lower_body(env, command_name: str = "motion") -> torch.Tensor:
    future = _motion(env, command_name).future_state()
    value = torch.cat([future["dof_pos"][..., :12], future["dof_vel"][..., :12]], dim=-1)
    return value.reshape(env.num_envs, -1)


def command_z(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    return cmd.anchor_pos_w[:, 2:3]


def command_z_multi_future_nonflat(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state()
    return future["body_pos_w"][:, :, cmd.motion_anchor_body_index, 2:3]


def command_num_frames(env, command_name: str = "motion") -> torch.Tensor:
    return _motion(env, command_name).command_num_frames


def vr_3point_local_target(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state(num_frames=1)
    pts = future["body_pos_w"][:, 0, cmd.vr_3point_body_indices_motion]
    pos_b, _ = subtract_frame_transforms(
        cmd.robot_anchor_pos_w[:, None, :].repeat(1, pts.shape[1], 1),
        cmd.robot_anchor_quat_w[:, None, :].repeat(1, pts.shape[1], 1),
        pts,
        future["body_quat_w"][:, 0, cmd.vr_3point_body_indices_motion],
    )
    return pos_b.reshape(env.num_envs, -1)


def vr_3point_local_target_multi_future(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    pos = vr_3point_local_target(env, command_name)
    return pos[:, None, :].repeat(1, cmd.cfg.num_future_frames, 1)


def vr_3point_local_orn_target(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state(num_frames=1)
    q = future["body_quat_w"][:, 0, cmd.vr_3point_body_indices_motion]
    robot_q = cmd.robot_anchor_quat_w[:, None, :].repeat(1, q.shape[1], 1)
    mat = matrix_from_quat(quat_mul(quat_inv(robot_q), q))
    return mat[..., :2].reshape(env.num_envs, -1)


def vr_3point_local_orn_target_multi_future(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    orn = vr_3point_local_orn_target(env, command_name)
    return orn[:, None, :].repeat(1, cmd.cfg.num_future_frames, 1)


def motion_anchor_ori_b_multi_future_current(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    ori = motion_anchor_ori_b(env, command_name)
    return ori[:, None, :].repeat(1, cmd.cfg.num_future_frames, 1)


def smpl_joints_multi_future_local_nonflat(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state(cmd.cfg.smpl_num_future_frames, cmd.cfg.smpl_dt_future_ref_frames)
    smpl = future.get("smpl_joints")
    if smpl is None:
        return torch.zeros(env.num_envs, cmd.cfg.smpl_num_future_frames, 72, device=env.device)
    return smpl.reshape(env.num_envs, cmd.cfg.smpl_num_future_frames, -1)


def smpl_root_ori_b_multi_future(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.future_state(cmd.cfg.smpl_num_future_frames, cmd.cfg.smpl_dt_future_ref_frames)
    q = future["body_quat_w"][:, :, cmd.motion_anchor_body_index]
    robot_q = cmd.robot_anchor_quat_w[:, None, :].repeat(1, q.shape[1], 1)
    mat = matrix_from_quat(quat_mul(quat_inv(robot_q), q))
    return mat[..., :2].reshape(env.num_envs, cmd.cfg.smpl_num_future_frames, -1)


def joint_pos_multi_future_wrist_for_smpl(env, command_name: str = "motion") -> torch.Tensor:
    future = _motion(env, command_name).future_state()
    wrist_ids = (19, 20, 21, 26, 27, 28)
    return future["dof_pos"][..., wrist_ids]


def soma_joints_multi_future_local_nonflat(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.soma_future_state()
    joints = future["soma_joints"]
    root_quat = future["soma_root_quat_w"].unsqueeze(-2).expand(-1, -1, joints.shape[-2], -1)
    joints_root = quat_apply(quat_inv(root_quat), joints)
    return joints_root.reshape(env.num_envs, cmd.cfg.smpl_num_future_frames, -1)


def soma_root_ori_b_multi_future(env, command_name: str = "motion") -> torch.Tensor:
    cmd = _motion(env, command_name)
    future = cmd.soma_future_state()
    q = future["soma_root_quat_w"]
    robot_q = cmd.robot_anchor_quat_w[:, None, :].repeat(1, q.shape[1], 1)
    mat = matrix_from_quat(quat_mul(quat_inv(robot_q), q))
    return mat[..., :2].reshape(env.num_envs, cmd.cfg.smpl_num_future_frames, -1)


def joint_pos_multi_future_wrist_for_soma(env, command_name: str = "motion") -> torch.Tensor:
    return joint_pos_multi_future_wrist_for_smpl(env, command_name)
