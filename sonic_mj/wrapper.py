from __future__ import annotations

import copy

import easydict
from omegaconf import OmegaConf
import torch


def _to_easydict(value):
    if isinstance(value, dict):
        return easydict.EasyDict({key: _to_easydict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_easydict(item) for item in value]
    return value


def _container_or_empty(value):
    return value if isinstance(value, dict) else {}


class SonicMjEnvWrapper:
    def __init__(self, env, config):
        self.env = env
        self.env.wrapper = self
        if hasattr(self.env.observation_space, "spaces"):
            self.env.observation_space = self.env.observation_space.spaces
        self.config = _to_easydict(OmegaConf.to_container(config, resolve=False))
        self.device = env.device
        self.num_envs = env.num_envs
        self.is_manager_env = True
        self.is_evaluating = False
        self.motion_command = env.command_manager.get_term("motion")
        self._motion_lib = self.motion_command.motion_lib
        self.obs_buf_dict = {}
        self.extras = {}
        self.config.num_envs = self.num_envs
        self.config["obs"] = _to_easydict(_container_or_empty(self.config.get("obs", {})))
        self.config["obs"]["obs_dict"] = _to_easydict(self.config["obs"].get("obs_dict", {}))
        self.config["obs"]["obs_dims"] = _to_easydict(self.config["obs"].get("obs_dims", {}))
        self.config["obs"]["group_obs_dims"] = _to_easydict(
            self.config["obs"].get("group_obs_dims", {})
        )
        self.config["obs"]["group_obs_names"] = _to_easydict(
            self.config["obs"].get("group_obs_names", {})
        )
        self.config["robot"] = _to_easydict(_container_or_empty(self.config.get("robot", {})))
        self.config["robot"]["algo_obs_dim_dict"] = _to_easydict(
            self.config["robot"].get("algo_obs_dim_dict", {})
        )
        self.config.robot.setdefault("actions_dim", env.action_space.shape[-1])
        self.config["rewards"] = _to_easydict(_container_or_empty(self.config.get("rewards", {})))
        self.config.rewards.setdefault("num_critics", 1)
        self.use_symmetry = False

    def reset(self, flatten_dict_obs=True):
        obs, _info = self.env.reset()
        processed = self.process_raw_obs(obs, flatten_dict_obs=flatten_dict_obs)
        self.obs_buf_dict = copy.deepcopy(processed)
        return processed

    def reset_all(self, *_, **__):
        return self.reset(flatten_dict_obs=True)

    def step(self, actions):
        if isinstance(actions, dict) or hasattr(actions, "keys"):
            env_actions = actions["actions"]
        else:
            env_actions = actions
        clip_value = self.config.get("action_clip_value", None)
        if clip_value is not None and clip_value > 0:
            env_actions = torch.clip(env_actions, -float(clip_value), float(clip_value))
        obs, rew, terminated, truncated, extras = self.env.step(env_actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        processed = self.process_raw_obs(obs, flatten_dict_obs=True)
        self.obs_buf_dict = copy.deepcopy(processed)
        extras.setdefault("log", {})
        extras["time_outs"] = truncated
        extras["episode"] = {}
        extras["to_log"] = {
            k: (
                v.float()
                if isinstance(v, torch.Tensor)
                else torch.tensor(v, device=self.device, dtype=torch.float32)
            )
            for k, v in extras["log"].items()
        }
        extras["env_actions"] = env_actions.detach().cpu()
        self.extras = extras
        return processed, rew, dones, extras

    def process_raw_obs(self, obs, flatten_dict_obs=True):
        if not flatten_dict_obs:
            return obs
        out = {}
        out["actor_obs"] = obs["policy"]
        out["critic_obs"] = obs["critic"]
        if isinstance(obs.get("tokenizer"), dict):
            out["tokenizer"] = torch.cat(
                [value.reshape(value.shape[0], -1) for value in obs["tokenizer"].values()],
                dim=-1,
            )
        elif "tokenizer" in obs:
            out["tokenizer"] = obs["tokenizer"]
        return out

    def get_env_state_dict(self):
        return {}

    def load_env_state_dict(self, state_dict):  # noqa: ARG002
        return None

    def set_is_evaluating(self, is_evaluating=True, *_, **__):
        self.is_evaluating = is_evaluating
        if hasattr(self.motion_command, "is_evaluating"):
            self.motion_command.is_evaluating = is_evaluating

    def set_is_training(self):
        self.set_is_evaluating(False)

    def resample_motion(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.motion_command._resample_command(env_ids)

    def reinit_dr(self):
        return None

    def render_results(self):
        return None

    def end_render_results(self):
        return None

    def close(self):
        self.env.close()
