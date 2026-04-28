from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mj_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from sonic_mj.assets import (
    SONIC_G1_ACTION_SCALE,
    SONIC_G1_BODY_NAMES,
    SONIC_G1_JOINT_NAMES,
    get_sonic_g1_robot_cfg,
)
from sonic_mj.mdp import observations as obs
from sonic_mj.mdp import events as sonic_events
from sonic_mj.mdp import rewards, terminations
from sonic_mj.mdp.commands import SonicMotionCommandCfg


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _as_tuple(value):
    return tuple(value) if value is not None else None


def _event_params(term_cfg):
    return _cfg_get(term_cfg, "params", {}) or {}


def _make_sonic_events(manager_cfg):
    source_events = _cfg_get(manager_cfg, "events", {}) or {}
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mj_mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    add_joint_cfg = _cfg_get(source_events, "add_joint_default_pos", None)
    if add_joint_cfg is not None:
        params = _event_params(add_joint_cfg)
        events["add_joint_default_pos"] = EventTermCfg(
            func=sonic_events.randomize_joint_default_pos,
            mode=_cfg_get(add_joint_cfg, "mode", "startup"),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "pos_distribution_params": _as_tuple(
                    _cfg_get(params, "pos_distribution_params", (-0.01, 0.01))
                ),
                "operation": _cfg_get(params, "operation", "add"),
            },
        )

    base_com_cfg = _cfg_get(source_events, "base_com", None)
    if base_com_cfg is not None:
        params = _event_params(base_com_cfg)
        com_range = _cfg_get(params, "com_range", {})
        events["base_com"] = EventTermCfg(
            func=mj_mdp.dr.body_com_offset,
            mode=_cfg_get(base_com_cfg, "mode", "startup"),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "ranges": {
                    0: _as_tuple(_cfg_get(com_range, "x", (0.0, 0.0))),
                    1: _as_tuple(_cfg_get(com_range, "y", (0.0, 0.0))),
                    2: _as_tuple(_cfg_get(com_range, "z", (0.0, 0.0))),
                },
                "operation": "add",
            },
        )

    mass_cfg = _cfg_get(source_events, "randomize_rigid_body_mass", None)
    if mass_cfg is not None:
        params = _event_params(mass_cfg)
        asset_cfg = _cfg_get(params, "asset_cfg", {})
        events["randomize_rigid_body_mass"] = EventTermCfg(
            func=mj_mdp.dr.body_mass,
            mode=_cfg_get(mass_cfg, "mode", "startup"),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_cfg_get(asset_cfg, "body_names", ".*"),
                ),
                "ranges": _as_tuple(
                    _cfg_get(params, "mass_distribution_params", (0.8, 1.2))
                ),
                "operation": _cfg_get(params, "operation", "scale"),
            },
        )

    physics_material_cfg = _cfg_get(source_events, "physics_material", None)
    if physics_material_cfg is not None:
        params = _event_params(physics_material_cfg)
        events["physics_material"] = EventTermCfg(
            func=mj_mdp.dr.geom_friction,
            mode=_cfg_get(physics_material_cfg, "mode", "startup"),
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                "ranges": _as_tuple(_cfg_get(params, "static_friction_range", (0.3, 1.6))),
                "operation": "abs",
                "axes": [0],
            },
        )

    push_cfg = _cfg_get(source_events, "push_robot", None)
    if push_cfg is not None:
        params = _event_params(push_cfg)
        velocity_range = {
            key: _as_tuple(value)
            for key, value in (_cfg_get(params, "velocity_range", {}) or {}).items()
        }
        events["push_robot"] = EventTermCfg(
            func=mj_mdp.push_by_setting_velocity,
            mode=_cfg_get(push_cfg, "mode", "interval"),
            interval_range_s=_as_tuple(
                _cfg_get(push_cfg, "interval_range_s", (4.0, 6.0))
            ),
            params={"velocity_range": velocity_range},
        )

    return events


def make_sonic_mj_env_cfg(config) -> ManagerBasedRlEnvCfg:
    manager_cfg = config.manager_env
    base_cfg = manager_cfg.config
    motion_cfg = manager_cfg.commands.motion
    motion_lib_cfg = OmegaConf.to_container(motion_cfg.motion_lib_cfg, resolve=True)
    motion_file = motion_lib_cfg.get("motion_file", "")
    smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", "dummy")
    motion_lib_cfg["smpl_motion_file"] = smpl_motion_file or "dummy"
    motion_lib_cfg.setdefault("motion_file", motion_file)
    mjlab_cfg = _cfg_get(config, "mjlab", {})
    nconmax = _cfg_get(mjlab_cfg, "nconmax", 512)
    njmax = _cfg_get(mjlab_cfg, "njmax", 2048)

    actor_hist = int(_cfg_get(config, "actor_prop_history_length", 10))
    actor_action_hist = int(_cfg_get(config, "actor_actions_history_length", 10))
    critic_hist = int(_cfg_get(config, "critic_prop_history_length", 10))
    critic_action_hist = int(_cfg_get(config, "critic_actions_history_length", 10))

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "gravity_dir": ObservationTermCfg(func=obs.gravity_dir, history_length=actor_hist),
                "base_ang_vel": ObservationTermCfg(func=obs.base_ang_vel, history_length=actor_hist),
                "joint_pos": ObservationTermCfg(func=obs.joint_pos, history_length=actor_hist),
                "joint_vel": ObservationTermCfg(func=obs.joint_vel, history_length=actor_hist),
                "actions": ObservationTermCfg(func=obs.actions, history_length=actor_action_hist),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "command_multi_future": ObservationTermCfg(func=obs.command_multi_future),
                "motion_anchor_pos_b": ObservationTermCfg(func=obs.motion_anchor_pos_b),
                "motion_anchor_ori_b": ObservationTermCfg(func=obs.motion_anchor_ori_b),
                "body_pos": ObservationTermCfg(func=obs.body_pos),
                "body_ori": ObservationTermCfg(func=obs.body_ori),
                "base_lin_vel": ObservationTermCfg(func=obs.base_lin_vel, history_length=critic_hist),
                "base_ang_vel": ObservationTermCfg(func=obs.base_ang_vel, history_length=critic_hist),
                "joint_pos": ObservationTermCfg(func=obs.joint_pos, history_length=critic_hist),
                "joint_vel": ObservationTermCfg(func=obs.joint_vel, history_length=critic_hist),
                "actions": ObservationTermCfg(func=obs.actions, history_length=critic_action_hist),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "tokenizer": ObservationGroupCfg(
            terms={
                "encoder_index": ObservationTermCfg(func=obs.encoder_index),
                "command_multi_future_nonflat": ObservationTermCfg(
                    func=obs.command_multi_future_nonflat
                ),
                "command_z_multi_future_nonflat": ObservationTermCfg(
                    func=obs.command_z_multi_future_nonflat
                ),
                "motion_anchor_ori_b_mf_nonflat": ObservationTermCfg(
                    func=obs.motion_anchor_ori_b_mf_nonflat
                ),
                "command_multi_future_lower_body": ObservationTermCfg(
                    func=obs.command_multi_future_lower_body
                ),
                "vr_3point_local_target": ObservationTermCfg(func=obs.vr_3point_local_target),
                "vr_3point_local_orn_target": ObservationTermCfg(
                    func=obs.vr_3point_local_orn_target
                ),
                "motion_anchor_ori_b": ObservationTermCfg(func=obs.motion_anchor_ori_b),
                "command_z": ObservationTermCfg(func=obs.command_z),
                "smpl_joints_multi_future_local_nonflat": ObservationTermCfg(
                    func=obs.smpl_joints_multi_future_local_nonflat
                ),
                "smpl_root_ori_b_multi_future": ObservationTermCfg(
                    func=obs.smpl_root_ori_b_multi_future
                ),
                "joint_pos_multi_future_wrist_for_smpl": ObservationTermCfg(
                    func=obs.joint_pos_multi_future_wrist_for_smpl
                ),
            },
            concatenate_terms=False,
            enable_corruption=False,
        ),
    }

    return ManagerBasedRlEnvCfg(
        decimation=int(base_cfg.decimation),
        episode_length_s=float(base_cfg.episode_length_s),
        seed=int(config.seed),
        scene=SceneCfg(
            num_envs=int(config.num_envs),
            env_spacing=float(base_cfg.env_spacing),
            entities={"robot": get_sonic_g1_robot_cfg()},
        ),
        sim=SimulationCfg(
            nconmax=None if nconmax is None else int(nconmax),
            njmax=None if njmax is None else int(njmax),
            mujoco=MujocoCfg(timestep=float(base_cfg.sim_dt)),
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=SONIC_G1_JOINT_NAMES,
                scale=SONIC_G1_ACTION_SCALE,
                use_default_offset=True,
            )
        },
        commands={
            "motion": SonicMotionCommandCfg(
                resampling_time_range=(1.0e9, 1.0e9),
                debug_vis=False,
                entity_name="robot",
                motion_file=str(motion_file),
                smpl_motion_file=str(smpl_motion_file),
                motion_lib_cfg=motion_lib_cfg,
                anchor_body="pelvis",
                body_names=SONIC_G1_BODY_NAMES,
                reward_point_body=tuple(motion_cfg.reward_point_body),
                reward_point_body_offset=tuple(tuple(v) for v in motion_cfg.reward_point_body_offset),
                num_future_frames=int(motion_cfg.num_future_frames),
                dt_future_ref_frames=float(motion_cfg.dt_future_ref_frames),
                smpl_num_future_frames=int(motion_cfg.smpl_num_future_frames),
                smpl_dt_future_ref_frames=float(motion_cfg.smpl_dt_future_ref_frames),
                cat_upper_body_poses=bool(motion_cfg.cat_upper_body_poses),
                cat_upper_body_poses_prob=float(motion_cfg.cat_upper_body_poses_prob),
                freeze_frame_aug=bool(motion_cfg.freeze_frame_aug),
            )
        },
        observations=observations,
        rewards={
            "tracking_anchor_pos": RewardTermCfg(func=rewards.tracking_anchor_pos, weight=0.5),
            "tracking_anchor_ori": RewardTermCfg(func=rewards.tracking_anchor_ori, weight=0.5),
            "tracking_relative_body_pos": RewardTermCfg(
                func=rewards.tracking_relative_body_pos, weight=1.0
            ),
            "tracking_relative_body_ori": RewardTermCfg(
                func=rewards.tracking_relative_body_ori, weight=1.0
            ),
            "tracking_body_linvel": RewardTermCfg(func=rewards.tracking_body_linvel, weight=1.0),
            "tracking_body_angvel": RewardTermCfg(func=rewards.tracking_body_angvel, weight=1.0),
            "tracking_vr_5point_local": RewardTermCfg(
                func=rewards.tracking_vr_5point_local, weight=1.0
            ),
            "action_rate_l2": RewardTermCfg(func=rewards.action_rate_l2, weight=-0.1),
            "joint_limit": RewardTermCfg(func=rewards.joint_limit, weight=-10.0),
            "feet_acc": RewardTermCfg(
                func=rewards.feet_acc,
                weight=float(_cfg_get(manager_cfg.rewards.feet_acc, "weight", -2.5e-6)),
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
            "anchor_pos": TerminationTermCfg(
                func=terminations.anchor_pos, params={"threshold": 1.0}
            ),
            "anchor_ori_full": TerminationTermCfg(
                func=terminations.anchor_ori_full, params={"threshold": 1.57}
            ),
            "ee_body_pos": TerminationTermCfg(
                func=terminations.ee_body_pos, params={"threshold": 1.0}
            ),
            "foot_pos_xyz": TerminationTermCfg(
                func=terminations.foot_pos_xyz, params={"threshold": 1.0}
            ),
        },
        events=_make_sonic_events(manager_cfg),
        scale_rewards_by_dt=True,
    )
