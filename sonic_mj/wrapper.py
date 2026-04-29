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
        self.start_idx = 0
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
        if "actions_dim" not in self.config.robot:
            self.config.robot.actions_dim = env.action_space.shape[-1]
        self.config["rewards"] = _to_easydict(_container_or_empty(self.config.get("rewards", {})))
        self.config.rewards.setdefault("num_critics", 1)
        self.use_symmetry = False

    def reset(self, flatten_dict_obs=True):
        obs, _info = self.env.reset()
        self.motion_command.sync_after_sim_forward()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.env.observation_manager.reset(env_ids)
        obs = self.env.observation_manager.compute(update_history=True)
        self.env.obs_buf = obs
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
        return {"motion_lib": self._motion_lib.get_state_dict()}

    def load_env_state_dict(self, state_dict):
        if "motion_lib" in state_dict:
            self._motion_lib.load_state_dict(state_dict["motion_lib"])
            if getattr(self._motion_lib, "use_adaptive_sampling", False):
                self.resample_motion()

    def set_is_evaluating(self, is_evaluating=True, global_rank=0, *_, **__):
        self.is_evaluating = is_evaluating
        if hasattr(self.motion_command, "set_is_evaluating"):
            self.motion_command.set_is_evaluating(is_evaluating)
        elif hasattr(self.motion_command, "is_evaluating"):
            self.motion_command.is_evaluating = is_evaluating
        if is_evaluating:
            self.begin_seq_motion_samples(global_rank=global_rank)

    def set_is_training(self):
        self.set_is_evaluating(False)
        if hasattr(self._motion_lib, "load_motions_for_training"):
            loaded = self._motion_lib.load_motions_for_training(
                max_num_seqs=min(self.num_envs, self.motion_command.max_num_load_motions)
            )
            if loaded:
                self.reset_all()

    def resample_motion(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.motion_command._resample_command(env_ids)

    def begin_seq_motion_samples(self, global_rank=0):
        self.start_idx = int(global_rank) * self.num_envs
        if hasattr(self._motion_lib, "load_motions_for_evaluation"):
            self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all()

    def forward_motion_samples(self, global_rank=0, world_size=1):
        del global_rank
        self.start_idx += int(world_size) * self.num_envs
        if hasattr(self._motion_lib, "load_motions_for_evaluation"):
            self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all()

    def reinit_dr(self):
        return None

    def get_env_data(self, key):
        if key == "ref_body_pos_extend":
            return self.motion_command.robot_body_pos_w
        if key == "rigid_body_pos_extend":
            return self.motion_command.body_pos_w
        if hasattr(self.env, "get_env_data"):
            return self.env.get_env_data(key)
        raise KeyError(f"Unsupported SonicMJ env data key: {key}")

    @property
    def motion_ids(self):
        return self.motion_command.motion_ids

    def sync_and_compute_adaptive_sampling(self, accelerator, sync_across_gpus=False):
        if self._motion_lib is not None:
            self._motion_lib.sync_and_compute_adaptive_sampling(
                accelerator, sync_across_gpus=sync_across_gpus
            )

    def render_results(self):
        return None

    def end_render_results(self):
        return None

    def close(self):
        self.env.close()
