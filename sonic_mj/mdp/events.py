from __future__ import annotations

from typing import Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def _as_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _as_joint_ids(env, asset, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if isinstance(asset_cfg.joint_ids, slice):
        return torch.arange(asset.num_joints, device=env.device, dtype=torch.long)
    return torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)


def _as_geom_ids(env, asset, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if not isinstance(asset_cfg.geom_ids, slice):
        return torch.tensor(asset_cfg.geom_ids, device=env.device, dtype=torch.long)

    geom_ids = torch.arange(asset.num_geoms, device=env.device, dtype=torch.long)
    if isinstance(asset_cfg.body_ids, slice):
        return geom_ids

    body_ids = torch.tensor(asset_cfg.body_ids, device=env.device, dtype=torch.long)
    model_geom_ids = asset.indexing.geom_ids[geom_ids].to(device=env.device, dtype=torch.long)
    geom_body_ids = env.sim.model.geom_bodyid[model_geom_ids]
    body_mask = torch.isin(geom_body_ids, asset.indexing.body_ids[body_ids])
    return geom_ids[body_mask]


def randomize_joint_default_pos(
    env,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | list[float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
):
    asset = env.scene[asset_cfg.name]
    env_ids_tensor = _as_env_ids(env, env_ids)
    joint_ids = _as_joint_ids(env, asset, asset_cfg)

    if pos_distribution_params is None:
        return

    low, high = float(pos_distribution_params[0]), float(pos_distribution_params[1])
    random_value = sample_uniform(
        low,
        high,
        (len(env_ids_tensor), len(joint_ids)),
        device=env.device,
    )
    current = asset.data.default_joint_pos[env_ids_tensor[:, None], joint_ids]

    if operation == "add":
        randomized = current + random_value
    elif operation == "scale":
        randomized = current * random_value
    elif operation == "abs":
        randomized = random_value
    else:
        raise ValueError(f"Unsupported joint default position operation: {operation}")

    asset.data.default_joint_pos[env_ids_tensor[:, None], joint_ids] = randomized

    if hasattr(env, "action_manager"):
        action_term = env.action_manager.get_term("joint_pos")
        action_names = list(action_term.target_names)
        joint_names = list(asset.joint_names)
        shared_action_ids = []
        shared_asset_ids = []
        selected_joint_ids = set(joint_ids.detach().cpu().tolist())
        for action_id, joint_name in enumerate(action_names):
            if joint_name in joint_names:
                asset_joint_id = joint_names.index(joint_name)
                if asset_joint_id in selected_joint_ids:
                    shared_action_ids.append(action_id)
                    shared_asset_ids.append(asset_joint_id)
        if shared_action_ids and isinstance(action_term.offset, torch.Tensor):
            action_term.offset[env_ids_tensor[:, None], shared_action_ids] = (
                asset.data.default_joint_pos[env_ids_tensor[:, None], shared_asset_ids]
            )


def randomize_physics_material(
    env,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float] | list[float] = (0.3, 1.6),
    dynamic_friction_range: tuple[float, float] | list[float] = (0.3, 1.2),
    restitution_range: tuple[float, float] | list[float] = (0.0, 0.5),
    num_buckets: int = 64,
):
    """Best-effort mjlab mapping of IsaacLab rigid-body material randomization.

    MuJoCo exposes one sliding friction coefficient plus optional torsional/rolling
    friction axes, but not separate PhysX-style static/dynamic friction or restitution.
    We keep the original bucketed sampling semantics and map static friction to the
    sliding axis. Dynamic friction is placed on the torsional axis for high-dimensional
    contacts; restitution is intentionally not mapped to solver parameters.
    """
    del restitution_range

    asset = env.scene[asset_cfg.name]
    env_ids_tensor = _as_env_ids(env, env_ids)
    geom_ids = _as_geom_ids(env, asset, asset_cfg)
    if len(geom_ids) == 0:
        return

    num_buckets = max(int(num_buckets), 1)
    static_low, static_high = float(static_friction_range[0]), float(static_friction_range[1])
    dynamic_low, dynamic_high = float(dynamic_friction_range[0]), float(dynamic_friction_range[1])
    friction_buckets = torch.empty((num_buckets, 2), device=env.device)
    friction_buckets[:, 0] = sample_uniform(
        static_low,
        static_high,
        (num_buckets,),
        device=env.device,
    )
    friction_buckets[:, 1] = sample_uniform(
        dynamic_low,
        dynamic_high,
        (num_buckets,),
        device=env.device,
    )

    bucket_ids = torch.randint(
        0,
        num_buckets,
        (len(env_ids_tensor), len(geom_ids)),
        device=env.device,
    )
    sampled_friction = friction_buckets[bucket_ids]

    model_geom_ids = asset.indexing.geom_ids[geom_ids].to(device=env.device, dtype=torch.long)
    env_grid, geom_grid = torch.meshgrid(env_ids_tensor, model_geom_ids, indexing="ij")
    env.sim.model.geom_friction[env_grid, geom_grid, 0] = sampled_friction[..., 0]
    if env.sim.model.geom_friction.shape[-1] > 1:
        env.sim.model.geom_friction[env_grid, geom_grid, 1] = sampled_friction[..., 1]
