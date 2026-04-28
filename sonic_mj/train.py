from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv

from sonic_mj.env_cfg import make_sonic_mj_env_cfg
from sonic_mj.wrapper import SonicMjEnvWrapper


def create_mjlab_manager_env(config, device: str):
    env_cfg = make_sonic_mj_env_cfg(config)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    return SonicMjEnvWrapper(env, config.manager_env.config)

