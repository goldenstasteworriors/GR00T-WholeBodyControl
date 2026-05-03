from __future__ import annotations

import copy

import easydict
from omegaconf import OmegaConf
import torch

from sonic_mj.assets import (
    SONIC_G1_BODY_NAMES,
    SONIC_G1_JOINT_NAMES,
    SONIC_G1_MOTION_DOF_TO_MUJOCO,
)


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
        self._train_only_event_names = tuple(self.config.get("train_only_events", []) or [])
        self._disabled_train_only_events = {}

    def _set_train_only_events_enabled(self, enabled: bool) -> None:
        if not self._train_only_event_names or not hasattr(self.env, "event_manager"):
            return
        if enabled:
            for event_name in self._train_only_event_names:
                self._restore_event_term(event_name)
        else:
            for event_name in self._train_only_event_names:
                self._remove_event_term(event_name)

    def _remove_event_term(self, event_name: str) -> None:
        if event_name in self._disabled_train_only_events:
            return
        manager = self.env.event_manager
        for mode, names in list(manager._mode_term_names.items()):
            if event_name not in names:
                continue
            index = names.index(event_name)
            term_cfg = manager._mode_term_cfgs[mode][index]
            removed = {
                "mode": mode,
                "index": index,
                "term_cfg": term_cfg,
                "class_index": None,
                "interval_time_left": None,
                "reset_last_step": None,
                "reset_triggered_once": None,
            }
            names.pop(index)
            manager._mode_term_cfgs[mode].pop(index)
            class_terms = manager._mode_class_term_cfgs.get(mode, [])
            for class_index, class_term_cfg in enumerate(class_terms):
                if class_term_cfg is term_cfg:
                    removed["class_index"] = class_index
                    class_terms.pop(class_index)
                    break
            if mode == "interval":
                removed["interval_time_left"] = manager._interval_term_time_left.pop(index)
            elif mode == "reset":
                removed["reset_last_step"] = manager._reset_term_last_triggered_step_id.pop(index)
                removed["reset_triggered_once"] = manager._reset_term_last_triggered_once.pop(index)
            self._disabled_train_only_events[event_name] = removed
            return

    def _restore_event_term(self, event_name: str) -> None:
        removed = self._disabled_train_only_events.pop(event_name, None)
        if removed is None:
            return
        manager = self.env.event_manager
        mode = removed["mode"]
        manager._mode_term_names.setdefault(mode, [])
        manager._mode_term_cfgs.setdefault(mode, [])
        manager._mode_class_term_cfgs.setdefault(mode, [])
        index = min(removed["index"], len(manager._mode_term_names[mode]))
        term_cfg = removed["term_cfg"]
        manager._mode_term_names[mode].insert(index, event_name)
        manager._mode_term_cfgs[mode].insert(index, term_cfg)
        if removed["class_index"] is not None:
            class_index = min(
                removed["class_index"],
                len(manager._mode_class_term_cfgs[mode]),
            )
            manager._mode_class_term_cfgs[mode].insert(class_index, term_cfg)
        if mode == "interval":
            manager._interval_term_time_left.insert(index, removed["interval_time_left"])
        elif mode == "reset":
            manager._reset_term_last_triggered_step_id.insert(index, removed["reset_last_step"])
            manager._reset_term_last_triggered_once.insert(index, removed["reset_triggered_once"])

    def get_order_diagnostics(self):
        robot = self.env.scene["robot"]
        action_joint_names = ()
        try:
            action_joint_names = tuple(self.env.action_manager.get_term("joint_pos").target_names)
        except Exception:
            pass
        obs_shapes = {}
        for group_name, space in self.env.observation_space.items():
            if hasattr(space, "shape"):
                obs_shapes[group_name] = tuple(space.shape)
            elif hasattr(space, "spaces"):
                obs_shapes[group_name] = {
                    term_name: tuple(term_space.shape)
                    for term_name, term_space in space.spaces.items()
                }
        diagnostics = {
            "robot_joint_names": tuple(robot.joint_names),
            "robot_body_names": tuple(robot.body_names),
            "motion_body_names": tuple(self.motion_command.cfg.body_names),
            "motion_dof_to_mujoco": tuple(SONIC_G1_MOTION_DOF_TO_MUJOCO),
            "action_joint_names": action_joint_names,
            "policy_joint_pos_order": tuple(robot.joint_names),
            "policy_joint_vel_order": tuple(robot.joint_names),
            "policy_action_order": action_joint_names,
            "action_dim": int(self.env.action_space.shape[-1]),
            "observation_shapes": obs_shapes,
        }
        diagnostics["checks"] = {
            "robot_joints_match_sonic_mujoco": diagnostics["robot_joint_names"]
            == SONIC_G1_JOINT_NAMES,
            "robot_bodies_match_sonic_mujoco": diagnostics["robot_body_names"]
            == SONIC_G1_BODY_NAMES,
            "motion_bodies_match_sonic_mujoco": diagnostics["motion_body_names"]
            == SONIC_G1_BODY_NAMES,
            "action_joints_match_sonic_mujoco": diagnostics["action_joint_names"]
            == SONIC_G1_JOINT_NAMES,
            "policy_joint_pos_order_matches_sonic_mujoco": diagnostics["policy_joint_pos_order"]
            == SONIC_G1_JOINT_NAMES,
            "policy_joint_vel_order_matches_sonic_mujoco": diagnostics["policy_joint_vel_order"]
            == SONIC_G1_JOINT_NAMES,
            "policy_action_order_matches_sonic_mujoco": diagnostics["policy_action_order"]
            == SONIC_G1_JOINT_NAMES,
            "motion_dof_mapping_identity": diagnostics["motion_dof_to_mujoco"]
            == tuple(range(len(SONIC_G1_JOINT_NAMES))),
            "action_dim_is_29": diagnostics["action_dim"] == 29,
        }
        return diagnostics

    def print_order_diagnostics(self):
        diagnostics = self.get_order_diagnostics()
        print("[SonicMJ] structured order diagnostics:")
        for key in (
            "robot_joint_names",
            "robot_body_names",
            "motion_body_names",
            "action_joint_names",
            "policy_joint_pos_order",
            "policy_joint_vel_order",
            "policy_action_order",
            "motion_dof_to_mujoco",
            "action_dim",
            "observation_shapes",
            "checks",
        ):
            print(f"[SonicMJ]   {key}: {diagnostics[key]}")
        return diagnostics

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
        self._set_train_only_events_enabled(not is_evaluating)
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
                max_num_seqs=self.motion_command.max_num_load_motions
            )
            if loaded:
                self.motion_command.refresh_after_motion_lib_reload()
                self.reset_all()

    def resample_motion(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.motion_command._resample_command(env_ids)

    def begin_seq_motion_samples(self, global_rank=0):
        self.start_idx = int(global_rank) * self.num_envs
        if hasattr(self._motion_lib, "load_motions_for_evaluation"):
            self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
            self.motion_command.refresh_after_motion_lib_reload()
        self.reset_all()

    def forward_motion_samples(self, global_rank=0, world_size=1):
        del global_rank
        self.start_idx += int(world_size) * self.num_envs
        if hasattr(self._motion_lib, "load_motions_for_evaluation"):
            self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
            self.motion_command.refresh_after_motion_lib_reload()
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
