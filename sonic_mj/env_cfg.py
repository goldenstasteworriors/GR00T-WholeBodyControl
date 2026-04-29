from __future__ import annotations

import math
from pathlib import Path

from omegaconf import OmegaConf

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mj_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains.config import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
from mjlab.terrains.primitive_terrains import BoxRandomGridTerrainCfg

from sonic_mj.assets import (
    SONIC_G1_ACTION_SCALE,
    SONIC_G1_BODY_NAMES,
    SONIC_G1_JOINT_NAMES,
    get_sonic_g1_robot_cfg,
)
from sonic_mj.mdp import curriculum as sonic_curriculum
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


def _term_params(group_cfg, term_name: str):
    return _cfg_get(_cfg_get(group_cfg, term_name, None), "params", {}) or {}


def _scale_range_to_pseudo_inertia_alpha(
    scale_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Convert mass/inertia scale range to mjlab pseudo-inertia log scale."""
    low, high = float(scale_range[0]), float(scale_range[1])
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"Mass scale range must be positive, got {scale_range}.")
    return 0.5 * math.log(low), 0.5 * math.log(high)


def _make_sonic_terrain_cfg(base_cfg, *, num_envs: int, seed: int) -> TerrainEntityCfg:
    terrain_type = _cfg_get(base_cfg, "terrain_type", "plane")
    env_spacing = float(_cfg_get(base_cfg, "env_spacing", 2.0))
    if terrain_type == "plane":
        return TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
            num_envs=num_envs,
        )
    if terrain_type != "trimesh":
        raise ValueError(f"Unsupported SonicMJ terrain_type: {terrain_type}")

    num_rows = int(_cfg_get(base_cfg, "rough_terrain_num_rows", 20))
    num_cols = int(_cfg_get(base_cfg, "rough_terrain_num_cols", 20))
    terrain_size = float(_cfg_get(base_cfg, "rough_terrain_size", 8.0))
    border_width = float(_cfg_get(base_cfg, "rough_terrain_border_width", 20.0))
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(terrain_size, terrain_size),
            border_width=border_width,
            num_rows=num_rows,
            num_cols=num_cols,
            sub_terrains={
                "boxes": BoxRandomGridTerrainCfg(
                    proportion=0.3,
                    grid_width=0.45,
                    grid_height_range=(0.001, 0.005),
                    platform_width=2.0,
                ),
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.05,
                    noise_range=(0.001, 0.005),
                    noise_step=0.02,
                    border_width=0.25,
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                ),
            },
        ),
        env_spacing=env_spacing,
        max_init_terrain_level=10,
        num_envs=num_envs,
    )


def _uses_variable_frame_masks(config) -> bool:
    backbone_cfg = _cfg_get(_cfg_get(_cfg_get(config, "algo", {}), "config", {}), "actor", {})
    backbone_cfg = _cfg_get(backbone_cfg, "backbone", {}) or {}
    for group_name in ("encoders", "decoders"):
        group = _cfg_get(backbone_cfg, group_name, {}) or {}
        for term_cfg in group.values():
            if len(_cfg_get(term_cfg, "mask", []) or []) > 0:
                return True
    return False


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
        mass_distribution_params = _as_tuple(
            _cfg_get(params, "mass_distribution_params", (0.8, 1.2))
        )
        operation = _cfg_get(params, "operation", "scale")
        if operation == "scale":
            mass_func = mj_mdp.dr.pseudo_inertia
            mass_params = {
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_cfg_get(asset_cfg, "body_names", ".*"),
                ),
                "alpha_range": _scale_range_to_pseudo_inertia_alpha(mass_distribution_params),
            }
        else:
            mass_func = mj_mdp.dr.body_mass
            mass_params = {
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_cfg_get(asset_cfg, "body_names", ".*"),
                ),
                "ranges": mass_distribution_params,
                "operation": operation,
            }
        events["randomize_rigid_body_mass"] = EventTermCfg(
            func=mass_func,
            mode=_cfg_get(mass_cfg, "mode", "startup"),
            params=mass_params,
        )

    physics_material_cfg = _cfg_get(source_events, "physics_material", None)
    if physics_material_cfg is not None:
        params = _event_params(physics_material_cfg)
        events["physics_material"] = EventTermCfg(
            func=sonic_events.randomize_physics_material,
            mode=_cfg_get(physics_material_cfg, "mode", "startup"),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_cfg_get(_cfg_get(params, "asset_cfg", {}), "body_names", ".*"),
                ),
                "static_friction_range": _as_tuple(
                    _cfg_get(params, "static_friction_range", (0.3, 1.6))
                ),
                "dynamic_friction_range": _as_tuple(
                    _cfg_get(params, "dynamic_friction_range", (0.3, 1.2))
                ),
                "restitution_range": _as_tuple(
                    _cfg_get(params, "restitution_range", (0.0, 0.5))
                ),
                "num_buckets": int(_cfg_get(params, "num_buckets", 64)),
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


def _normalise_param_path(path: str | list[str | int] | tuple[str | int, ...]) -> list[str | int]:
    if isinstance(path, str):
        parts: list[str | int] = []
        for part in path.replace("/", ".").split("."):
            if not part:
                continue
            parts.append(int(part) if part.isdigit() else part)
        return parts
    return list(path)


def _make_curriculum_param_path(params) -> list[str | int]:
    path = _cfg_get(params, "param_path", None)
    if path is not None:
        return _normalise_param_path(path)
    velocity_axis = _cfg_get(params, "velocity_axis", None)
    bound_index = _cfg_get(params, "bound_index", None)
    if velocity_axis is not None and bound_index is not None:
        return ["velocity_range", str(velocity_axis), int(bound_index)]
    target = _cfg_get(params, "target", None)
    if isinstance(target, str) and "velocity_range" in target:
        parts = target.replace("[", ".").replace("]", "").replace("'", "").replace('"', "")
        parts = [part for part in parts.replace("/", ".").split(".") if part]
        try:
            velocity_index = parts.index("velocity_range")
        except ValueError:
            velocity_index = -1
        if velocity_index >= 0:
            return [
                int(part) if part.isdigit() else part
                for part in parts[velocity_index:]
            ]
    raise ValueError(
        "SonicMJ curriculum terms need params.param_path or "
        "params.velocity_axis + params.bound_index to modify mjlab event params."
    )


def _make_sonic_curriculum(manager_cfg):
    source_curriculum = _cfg_get(manager_cfg, "curriculum", {}) or {}
    curriculum = {}
    term_specs = {
        "force_push_curriculum": "step",
        "force_push_linear_curriculum": "linear",
    }
    for term_name, mode in term_specs.items():
        term_cfg = _cfg_get(source_curriculum, term_name, None)
        if term_cfg is None:
            continue
        params = _event_params(term_cfg)
        curriculum[term_name] = CurriculumTermCfg(
            func=sonic_curriculum.event_param_curriculum,
            params={
                "event_name": _cfg_get(params, "event_name", "push_robot"),
                "param_path": _make_curriculum_param_path(params),
                "original_value": float(_cfg_get(params, "original_value", 0.0)),
                "values": list(_cfg_get(params, "values", [])),
                "num_steps": list(_cfg_get(params, "num_steps", [])),
                "mode": mode,
            },
        )
    return curriculum


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
    termination_cfg = _cfg_get(manager_cfg, "terminations", {}) or {}
    encoder_sample_probs = dict(_cfg_get(motion_cfg, "encoder_sample_probs", {}) or {})
    use_soma_encoder = "soma" in encoder_sample_probs
    use_variable_frames = bool(_cfg_get(motion_cfg, "variable_frames_enabled", False))
    expose_variable_frame_mask = use_variable_frames and _uses_variable_frame_masks(config)
    num_envs = int(config.num_envs)

    tokenizer_terms = {
        "encoder_index": ObservationTermCfg(func=obs.encoder_index),
        "command_multi_future_nonflat": ObservationTermCfg(func=obs.command_multi_future_nonflat),
        "command_z_multi_future_nonflat": ObservationTermCfg(
            func=obs.command_z_multi_future_nonflat
        ),
        "motion_anchor_ori_b_mf_nonflat": ObservationTermCfg(
            func=obs.motion_anchor_ori_b_mf_nonflat
        ),
        "command_multi_future_lower_body": ObservationTermCfg(func=obs.command_multi_future_lower_body),
        "vr_3point_local_target": ObservationTermCfg(func=obs.vr_3point_local_target),
        "vr_3point_local_orn_target": ObservationTermCfg(func=obs.vr_3point_local_orn_target),
        "motion_anchor_ori_b": ObservationTermCfg(func=obs.motion_anchor_ori_b),
        "command_z": ObservationTermCfg(func=obs.command_z),
        "smpl_joints_multi_future_local_nonflat": ObservationTermCfg(
            func=obs.smpl_joints_multi_future_local_nonflat
        ),
        "smpl_root_ori_b_multi_future": ObservationTermCfg(func=obs.smpl_root_ori_b_multi_future),
        "joint_pos_multi_future_wrist_for_smpl": ObservationTermCfg(
            func=obs.joint_pos_multi_future_wrist_for_smpl
        ),
    }
    if expose_variable_frame_mask:
        tokenizer_terms["command_num_frames"] = ObservationTermCfg(func=obs.command_num_frames)
    if use_soma_encoder:
        tokenizer_terms.update(
            {
                "soma_joints_multi_future_local_nonflat": ObservationTermCfg(
                    func=obs.soma_joints_multi_future_local_nonflat
                ),
                "soma_root_ori_b_multi_future": ObservationTermCfg(
                    func=obs.soma_root_ori_b_multi_future
                ),
                "joint_pos_multi_future_wrist_for_soma": ObservationTermCfg(
                    func=obs.joint_pos_multi_future_wrist_for_soma
                ),
            }
        )

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
            terms=tokenizer_terms,
            concatenate_terms=False,
            enable_corruption=False,
        ),
    }

    return ManagerBasedRlEnvCfg(
        decimation=int(base_cfg.decimation),
        episode_length_s=float(base_cfg.episode_length_s),
        seed=int(config.seed),
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=float(base_cfg.env_spacing),
            terrain=_make_sonic_terrain_cfg(base_cfg, num_envs=num_envs, seed=int(config.seed)),
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
                variable_frames_enabled=use_variable_frames,
                variable_frames_min=int(_cfg_get(motion_cfg, "variable_frames_min", 16)),
                variable_frames_step=int(_cfg_get(motion_cfg, "variable_frames_step", 4)),
                cat_upper_body_poses=bool(motion_cfg.cat_upper_body_poses),
                cat_upper_body_poses_prob=float(motion_cfg.cat_upper_body_poses_prob),
                freeze_frame_aug=bool(motion_cfg.freeze_frame_aug),
                freeze_frame_aug_prob=float(_cfg_get(motion_cfg, "freeze_frame_aug_prob", 0.1)),
                encoder_sample_probs=encoder_sample_probs or None,
                teleop_sample_prob_when_smpl=float(
                    _cfg_get(motion_cfg, "teleop_sample_prob_when_smpl", 0.0)
                ),
                start_from_first_frame=bool(_cfg_get(motion_cfg, "start_from_first_frame", False)),
                sample_unique_motions=bool(_cfg_get(motion_cfg, "sample_unique_motions", False)),
                use_paired_motions=bool(_cfg_get(motion_cfg, "use_paired_motions", False)),
                sample_from_n_initial_frames=_cfg_get(
                    motion_cfg, "sample_from_n_initial_frames", None
                ),
                contact_file=_cfg_get(motion_cfg, "contact_file", None),
                sample_before_contact=bool(
                    _cfg_get(motion_cfg, "sample_before_contact", False)
                ),
                sample_before_contact_margin=int(
                    _cfg_get(motion_cfg, "sample_before_contact_margin", 10)
                ),
                sample_before_contact_hand=str(
                    _cfg_get(motion_cfg, "sample_before_contact_hand", "right_hand")
                ),
                contact_frame_tolerance=int(_cfg_get(motion_cfg, "contact_frame_tolerance", 3)),
                pose_range=dict(_cfg_get(motion_cfg, "pose_range", {}) or {}),
                velocity_range=dict(_cfg_get(motion_cfg, "velocity_range", {}) or {}),
                joint_position_range=_as_tuple(
                    _cfg_get(motion_cfg, "joint_position_range", (-0.1, 0.1))
                ),
                joint_velocity_range=_as_tuple(
                    _cfg_get(motion_cfg, "joint_velocity_range", (0.0, 0.0))
                ),
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
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*ankle.*",))},
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
            "anchor_pos": TerminationTermCfg(
                func=terminations.anchor_pos,
                params=_term_params(termination_cfg, "anchor_pos"),
            ),
            "anchor_ori_full": TerminationTermCfg(
                func=terminations.anchor_ori_full,
                params={
                    key: value
                    for key, value in _term_params(termination_cfg, "anchor_ori_full").items()
                    if key != "asset_cfg"
                },
            ),
            "ee_body_pos": TerminationTermCfg(
                func=terminations.ee_body_pos,
                params=_term_params(termination_cfg, "ee_body_pos"),
            ),
            "foot_pos_xyz": TerminationTermCfg(
                func=terminations.foot_pos_xyz,
                params=_term_params(termination_cfg, "foot_pos_xyz"),
            ),
        },
        events=_make_sonic_events(manager_cfg),
        curriculum=_make_sonic_curriculum(manager_cfg),
        scale_rewards_by_dt=True,
    )
