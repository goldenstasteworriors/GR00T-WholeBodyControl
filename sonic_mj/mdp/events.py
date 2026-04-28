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
