# PKU SonicMJ 4-GPU Training Debug

记录时间：2026-05-19

## 运行位置

- 服务器：PKU，`ubuntu@27.190.15.128`
- 服务器项目目录：`/home/nvme02/GR00T/GR00T`
- 本次日志文件：`/home/nvme02/GR00T/GR00T/run_logs/test_4gpu_4096_20260519_132447.log`
- W&B run：`https://wandb.ai/2004kjy666-huazhong-university-of-science-and-technology/TRL_G1_Track/runs/0bli55ow`

## 运行命令

注意：真实 `WANDB_API_KEY` 已省略，避免把密钥写入仓库文件。

```bash
mkdir -p /home/nvme02/GR00T/GR00T/run_logs /home/nvme02/GR00T/GR00T/wandb && WANDB_API_KEY='<WANDB_API_KEY>' nohup bash -c 'cd /home/nvme02/GR00T/GR00T && WANDB_MODE=online CUDA_VISIBLE_DEVICES=4,5,6,7 UV_CACHE_DIR=/home/nvme02/GR00T/GR00T/uv-cache ./.tools/uv/uv run python -m accelerate.commands.launch --num_processes=4 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab use_wandb=True wandb.wandb_dir=/home/nvme02/GR00T/GR00T/wandb num_envs=2048 headless=True ++algo.config.num_learning_iterations=20 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane' > /home/nvme02/GR00T/GR00T/run_logs/test_4gpu_4096_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 训练进展

本次训练已经成功进入 PPO 训练循环，并跑到第 8 个 learning iteration 后失败。

```text
Learning iteration 1
Mean rewards: -21.86654
Total timesteps: 196608

Learning iteration 2
Mean rewards: -20.96198
Total timesteps: 393216

Learning iteration 3
Mean rewards: -18.03822
Total timesteps: 589824

Learning iteration 4
Mean rewards: -19.94408
Total timesteps: 786432

Learning iteration 5
Mean rewards: -16.01417
Total timesteps: 983040

Learning iteration 6
Mean rewards: -19.53630
Total timesteps: 1179648

Learning iteration 7
Mean rewards: -19.34243
Total timesteps: 1376256

Learning iteration 8
Mean rewards: -19.45314
Total timesteps: 1572864
```

失败前最后一个完整 iteration 日志：

```text
Learning iteration 8

Computation: 16196 steps/s (Collection: 8.866s, Learning 3.273s)
Mean action noise std: 0.05
Mean entropy: -43.88948
Mean rewards: -19.45314
Mean length: 12.03250
Env/Episode_Reward/tracking_anchor_pos: 0.0096
Env/Episode_Reward/tracking_anchor_ori: 0.0080
Env/Episode_Reward/tracking_relative_body_pos: 0.0197
Env/Episode_Reward/tracking_relative_body_ori: 0.0059
Env/Episode_Reward/tracking_body_linvel: 0.0122
Env/Episode_Reward/tracking_body_angvel: 0.0087
Env/Episode_Reward/tracking_vr_5point_local: 0.0252
Env/Episode_Reward/action_rate_l2: -0.0003
Env/Episode_Reward/joint_limit: -0.0008
Env/Episode_Reward/feet_acc: -2.1081
Env/Metrics/motion/error_anchor_pos: 0.1035
Env/Episode_Termination/time_out: 0.0000
Env/Episode_Termination/anchor_pos: 0.8333
Env/Episode_Termination/anchor_ori_full: 20.1250
Env/Episode_Termination/ee_body_pos: 78.0000
Env/Episode_Termination/foot_pos_xyz: 102.3333
Total episodes: 65536
Total timesteps: 1572864
Iteration time: 12.14s
Total time: 109.73s
ETA: 164.6s
Logging Directory: logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260519_132511
```

## 完整报错

```text
Error executing job with overrides: ['+exp=manager/universal_token/all_modes/sonic_release', 'use_mjlab=True', 'sim_type=mjlab', 'use_wandb=True', 'wandb.wandb_dir=/home/nvme02/GR00T/GR00T/wandb', 'num_envs=2048', 'headless=True', '++algo.config.num_learning_iterations=20', '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered', '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered', 'manager_env.config.terrain_type=plane']
Traceback (most recent call last):
  File "/home/nvme02/GR00T/GR00T/gear_sonic/train_agent_trl.py", line 480, in main
    trainer.train()
  File "/home/nvme02/GR00T/GR00T/gear_sonic/trl/trainer/ppo_trainer.py", line 1722, in train
    obs_dict = self._rollout_step(model, obs_dict)
  File "/home/nvme02/GR00T/GR00T/gear_sonic/trl/trainer/ppo_trainer.py", line 912, in _rollout_step
    policy_state_dict = self.policy_step(policy_model, obs_dict, cur_dones=dones)
  File "/home/nvme02/GR00T/GR00T/gear_sonic/trl/trainer/ppo_trainer.py", line 837, in policy_step
    policy_state_dict = policy_model.rollout(
  File "/home/nvme02/GR00T/GR00T/gear_sonic/trl/modules/actor_critic_modules.py", line 432, in rollout
    "actions": self.distribution.sample(),
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/torch/distributions/normal.py", line 74, in sample
    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
RuntimeError: normal expects all elements of std >= 0.0

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
W0519 13:40:58.063311 3492651 .venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 3494484 closing signal SIGTERM
W0519 13:40:58.064070 3492651 .venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 3494486 closing signal SIGTERM
W0519 13:40:58.064403 3492651 .venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:900] Sending process 3494490 closing signal SIGTERM
E0519 13:41:03.288824 3492651 .venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 2 (pid: 3494488) of binary: /home/nvme02/GR00T/GR00T/.venv/bin/python3
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1415, in <module>
    main()
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1411, in main
    launch_command(args)
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1396, in launch_command
    multi_gpu_launcher(args)
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1023, in multi_gpu_launcher
    distrib_run.run(args)
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/nvme02/GR00T/GR00T/.venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 270, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
gear_sonic/train_agent_trl.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-05-19_13:40:58
  host      : instance-afs92r3e
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 3494488)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```

## 初步判断

- 本次不是 OOM：4 卡每卡 `num_envs=2048` 已进入训练并完成 8 个 learning iteration。
- 失败类型是策略动作分布的 `std` 出现非法负值，触发 PyTorch `Normal.sample()` 检查。
- 相关代码位置：
  - `gear_sonic/train_agent_trl.py:480`
  - `gear_sonic/trl/trainer/ppo_trainer.py:1722`
  - `gear_sonic/trl/trainer/ppo_trainer.py:912`
  - `gear_sonic/trl/trainer/ppo_trainer.py:837`
  - `gear_sonic/trl/modules/actor_critic_modules.py:432`
