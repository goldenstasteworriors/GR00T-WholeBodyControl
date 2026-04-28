# SonicMJ 迁移进展记录

更新时间：2026-04-28

## 当前状态

- 已新增 `sonic_mj` 包，提供 mjlab 原生 `ManagerBasedRlEnv` 路径，用于承载 SONIC G1 29-DOF 训练闭环。
- `gear_sonic/train_agent_trl.py` 已支持 `use_mjlab=True` / `sim_type=mjlab` 分支；mjlab 路径不再依赖 IsaacLab 初始化。
- G1 MuJoCo 资产使用 SONIC 默认 XML：
  `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`。
- 已保持 SONIC MuJoCo canonical joint/body order，action dim 保持 29。
- `JointPositionAction` 语义保持为：
  `target_joint_pos = default_joint_pos + action * action_scale`。
- `TrackingCommand` 已迁移 motion_lib 加载、motion id/start time 采样、当前帧和 future reference 输出、encoder index 采样。
- 已接入 policy、critic、tokenizer 观测组，核心 reward、termination 已按 mjlab manager 配置接入。
- 已接入第一批训练必要 events：
  - `reset_scene_to_default`
  - `add_joint_default_pos`
  - `base_com`
  - `randomize_rigid_body_mass`
  - `physics_material`
  - `push_robot`
- `randomize_rigid_body_mass` 已从 mjlab `dr.body_mass` 改为优先使用 `dr.pseudo_inertia`：
  - SONIC 的 `mass_distribution_params` scale range 会转换为 mjlab pseudo-inertia 的 `alpha_range`。
  - 当前 mass 与 inertia 会一起按密度尺度随机化，不再触发只改 mass、不改 inertia 的物理一致性警告。
- `feet_acc` 已从占位零奖励改为 mjlab `joint_acc_l2`，只惩罚 `.*ankle.*` 关节加速度；`sonic_release` 实验覆盖权重保持 `-2.5e-6`。
- termination 已进一步对齐原 SONIC 严格配置：
  - `anchor_pos` 使用 adaptive root height error。
  - `anchor_ori_full` 使用 squared quaternion error。
  - `ee_body_pos` 使用 ankles + wrists 的 adaptive height error。
  - `foot_pos_xyz` 使用 feet 的 3D position error。
  - `TrackingCommand` 已维护 `running_ref_root_height` EMA。
- 进一步修正了 mjlab reset/reference 对齐：
  - `SonicMjEnvWrapper.reset()` 在 mjlab `sim.forward()` 后同步 motion command，并重置 observation history 后重新计算首帧观测。
  - `TrackingCommand` 不再让 motion_lib 触发 IsaacLab body/dof 重排分支；后续诊断确认 motion_lib 输出已是 SONIC MuJoCo joint order，因此 mjlab 写入保持 identity order。
  - motion_lib quaternion 在 `TrackingCommand` 内显式从 `xyzw` 转为 mjlab/MuJoCo 使用的 `wxyz`。
  - `G1_ISAACLAB_TO_MUJOCO_BODY` / `G1_MUJOCO_TO_ISAACLAB_BODY` 已按 body 名称重新核对为互逆映射。
- 已修正 mjlab reset 时的 motion DOF 写入顺序：
  - 诊断确认 motion_lib 由 SONIC XML FK 生成的 `dof_pos` 已经是 SONIC MuJoCo joint order。
  - `SONIC_G1_MOTION_DOF_TO_MUJOCO` 改为 identity，不再错误重排到 alphabetic motor order。
  - reset 后 body target 与 mjlab robot FK 最大误差从约 `0.57m` 降到 `1e-6m` 量级。
  - `ee_body_pos`、`foot_pos_xyz` 首帧不再因为 FK/order 错位触发。
- 已为 mjlab simulation 增加 `nconmax=512`、`njmax=2048` 配置，短训练中未再看到 `nefc overflow`。
- 已继续对齐 `TrackingCommand` 的 adaptive sampling / evaluation mode 行为：
  - mjlab command 初始化后读取 `motion_lib.use_adaptive_sampling`。
  - reset 重采样时支持 adaptive `sample_motion_ids_and_time_steps()`。
  - step 更新时调用 `motion_lib.update_adaptive_sampling(reset_terminated, motion_ids, cur_time_steps)`。
  - evaluation mode 下按 env id 确定性遍历 motion，并从第 0 帧开始。
  - `SonicMjEnvWrapper` 已补回 `sync_and_compute_adaptive_sampling()`、`get_env_state_dict()`、`load_env_state_dict()`，保持 TRL trainer 与 motion_lib 状态保存/恢复接口一致。
- 已继续补齐 `TrackingCommand` 的原 SONIC reset 语义：
  - reset 写入 MuJoCo 前会按 `pose_range`、`velocity_range`、`joint_position_range`、`joint_velocity_range` 对 root pose/velocity 和 joint pos/vel 做训练期随机化。
  - evaluation mode 会跳过上述 reset 随机化，保持可视化/评估起始帧稳定。
  - 支持 `start_from_first_frame`、`sample_from_n_initial_frames`、`sample_unique_motions`、`use_paired_motions` 的基础采样语义。
  - `encoder_sample_probs` 已从配置传入 mjlab command，`encoder_index` 改为与原 SONIC 一致的 multi-hot 形式；SMPL encoder 被采样时会同步激活 G1，并按 `teleop_sample_prob_when_smpl` 额外采样 teleop。
- 已提交并推送到远程：
  - branch：`SONICMJ`
  - commit：`1b9cb72 Add SonicMJ mjlab migration path`
  - remote：`origin/SONICMJ`

## 已验证结果

- 编译检查通过：
  `uv run python -m py_compile ...`
- reset/step smoke test 通过：
  - `actor_obs`: `(2, 930)`
  - `critic_obs`: `(2, 1789)`
  - action dim: `29`
- tiny training 通过：
  - `num_envs=2`
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
- AGENTS.md 要求的小规模训练命令已跑通：
  - `num_envs=16`
  - `headless=True`
  - `num_learning_iterations=10`
  - 最终完成 `Learning iteration 10`
  - total timesteps: `3840`
- events 迁移后再次短训练通过：
  - `num_envs=16`
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
  - EventManager 能正常加载 reset/startup/interval 事件
  - action dim 仍为 `29`
  - policy obs 仍为 `930`
  - critic obs 仍为 `1789`
- `feet_acc` 迁移后 tiny training 通过：
  - `num_envs=2`
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
  - RewardManager 正常加载 `feet_acc`
  - 日志出现非零 `Env/Episode_Reward/feet_acc`
- termination 对齐后 tiny training 通过：
  - `num_envs=2`
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
  - TerminationManager 正常执行 adaptive height / foot xyz 逻辑
- reset/reference 诊断：
  - reset 后 `joint_pos` 与 mjlab robot joint state 最大误差为 `0.0`。
  - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action dim `29`。
  - 修正前 pelvis relative reference 曾出现 stale body pose；wrapper reset 同步后 pelvis 误差稳定为 `0.0`。
  - motion_lib DOF 顺序映射与 quaternion 转换后，tiny training 仍可完成 `Learning iteration 1`，`feet_acc` episode reward 从约 `-2.8680` 降到约 `-0.1962`。
  - 2026-04-28 复查 DOF 写入顺序：当前 `SONIC_G1_MOTION_DOF_TO_MUJOCO=identity` 后，reset 后 body FK 最大误差为 `[9.56e-07, 1.83e-07]`，`ee_body_pos=False`、`foot_pos_xyz=False`。
- DOF 顺序修正后训练验证：
  - `num_envs=2`
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
  - 完成 `Learning iteration 1`，termination 日志中 `ee_body_pos=0.0000`、`foot_pos_xyz=0.0000`。
  - 该极小 batch 仍会出现 PPO `batch_size` 不能整除 `num_mini_batches` 的提示，属于配置尺寸过小导致。
- DOF 顺序修正后 `num_envs=16` 短训练验证：
  - `num_learning_iterations=1`
  - `num_steps_per_env=2`
  - 完成 `Learning iteration 1`
  - total timesteps: `32`
  - termination 日志中 `anchor_pos=0.0000`、`anchor_ori_full=0.0000`、`ee_body_pos=0.0000`、`foot_pos_xyz=0.0000`。
- adaptive sampling 对齐后验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/wrapper.py sonic_mj/env_cfg.py sonic_mj/mdp/rewards.py sonic_mj/mdp/terminations.py` 通过。
  - `num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2` tiny training 通过。
  - RewardManager 中 `feet_acc` 权重仍为 `-2.5e-06`，action dim / observation 构建未被破坏。
- mass/inertia 随机化改用 `pseudo_inertia` 后验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/events.py sonic_mj/mdp/commands.py sonic_mj/mdp/rewards.py sonic_mj/mdp/terminations.py sonic_mj/wrapper.py` 通过。
  - `num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2` tiny training 通过。
  - EventManager 的 DR 字段包含 `body_mass`、`body_inertia`、`body_iquat`，确认 `pseudo_inertia` 生效。
  - 日志未再出现 `dr.body_mass only randomizes mass and leaves the inertia tensor unchanged` 警告。
- reset 随机化与 encoder multi-hot 对齐后验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/mdp/observations.py sonic_mj/env_cfg.py` 通过。
  - `num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2` tiny training 通过。
  - `num_envs=16`、`num_learning_iterations=1`、`num_steps_per_env=2` 短训练通过。
  - action dim 仍为 `29`，policy obs 仍为 `930`，critic obs 仍为 `1789`，tokenizer `encoder_index` 仍为 `(3,)`。
  - termination 日志中 `anchor_pos=0.0000`、`anchor_ori_full=0.0000`、`ee_body_pos=0.0000`、`foot_pos_xyz=0.0000`。

## 关键实现位置

- mjlab env 构建入口：`sonic_mj/train.py`
- mjlab env 配置：`sonic_mj/env_cfg.py`
- G1 资产、关节顺序、body 顺序、action scale：`sonic_mj/assets.py`
- TRL 兼容 wrapper：`sonic_mj/wrapper.py`
- motion command：`sonic_mj/mdp/commands.py`
- observations：`sonic_mj/mdp/observations.py`
- rewards：`sonic_mj/mdp/rewards.py`
- terminations：`sonic_mj/mdp/terminations.py`
- events：`sonic_mj/mdp/events.py`
- 训练入口分支：`gear_sonic/train_agent_trl.py`
- UniversalTokenModule 兼容修复：`gear_sonic/trl/modules/universal_token_modules.py`
- mjlab/uv 配置：`pyproject.toml`、`uv.lock`、`.python-version`

## 已运行的重要命令

```bash
uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input /home/ykj/Downloads/dataset/bones-seed/g1/csv/210531 \
  --output data/motion_lib_bones_seed/robot_smoke \
  --fps 30 \
  --fps_source 120 \
  --individual \
  --num_workers 8
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  checkpoint=null \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=10 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  checkpoint=null \
  num_envs=2 \
  headless=True \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_steps_per_env=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  checkpoint=null \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_steps_per_env=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

## 待解决问题

### 高优先级

- termination 已按原 SONIC term 类型、body 选择和阈值来源对齐；body 映射、quaternion 顺序、reset 首帧 stale pose 和 DOF 写入顺序已修正。当前 reset FK 误差已到 `1e-6m` 量级，`ee_body_pos`、`foot_pos_xyz` 首帧误触发问题已解决；后续仍需在更长训练中观察随机动作导致的真实 tracking failure 分布。
- `physics_material` 目前只映射为 MuJoCo `geom_friction` 的滑动摩擦随机化；Isaac 里的 dynamic friction 和 restitution 还没有完整等价迁移。
- `randomize_rigid_body_mass` 当前已对 `scale` 操作使用 mjlab `dr.pseudo_inertia`，比 Isaac 只缩放 mass 的事件更物理一致；后续长训练仍需观察该差异是否影响 SONIC 原始训练分布。
- `TrackingCommand` 已初步对齐 adaptive sampling 和 evaluation mode；后续仍需核对 motion cache、paired motions、contact-based initialization 与更长训练下的 adaptive sampling 分布是否和原始 `gear_sonic` 一致。
- `TrackingCommand` 已补齐基础 reset 随机化、start-time override、unique/paired motion 采样和 encoder multi-hot；后续还需迁移 contact-based initialization、variable future frames、SOMA encoder 的完整 4-encoder 采样分支。

### 中优先级

- rough terrain/trimesh terrain 尚未迁移；当前只保证 plane 训练闭环。
- curriculum 尚未完整迁移，尤其是与 push/randomization 相关的课程项。
- events/domain randomization 还需要更细致对齐原 IsaacLab：
  - physics material bucket 语义
  - reset-time vs startup-time 随机化
  - body/geom/name 选择是否完全一致
- `open3d` 当前未安装，mesh 加载做成缺失时跳过；mesh 可视化和精细 FK 相关能力后续需要补齐环境依赖或替代实现。
- 需要继续核对 motion_lib body/joint order、mjlab robot body/joint order、policy obs/action order 在更大数据集和 checkpoint 加载时是否完全一致。

### 低优先级

- 清理或降噪第三方 warning：
  - `PPOConfig` TRL import path warning
  - `torch.cuda.amp.autocast` deprecation warning
  - mjlab/mujoco_warp 部分 API deprecation warning
- 可补充 README 或训练文档，说明 mjlab 分支的启动命令、uv 环境和已知限制。
- 后续如需长期训练，建议增加脚本化 smoke test，但测试脚本创建后必须按 AGENTS.md 要求测试完删除。

## 下一步建议

1. 对 events 做一次原 SONIC vs mjlab 的逐项对照表，明确哪些是等价实现、近似实现、暂未实现。
2. 用 `sonic_release` checkpoint 或同结构 checkpoint 做加载 smoke，确认网络 key、obs dim、action dim 无破坏。
3. 再跑更长一点的小规模训练，观察是否还有 `nefc overflow`、NaN、异常 reset 或 reward 退化。
4. 继续核对 motion cache、paired motions、contact-based initialization 与原始 `gear_sonic` 的行为一致性。
