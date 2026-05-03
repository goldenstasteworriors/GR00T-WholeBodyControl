# SonicMJ 迁移进展记录

更新时间：2026-05-03

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
- `physics_material` 已从单轴 `geom_friction` 随机化改为 SONIC 专用 mjlab event：
  - 保留原 Isaac 配置的 `num_buckets` bucket 采样语义。
  - 按 `asset_cfg.body_names` 选择对应 body 下的 MuJoCo geoms。
  - `static_friction_range` 映射到 MuJoCo sliding friction。
  - `dynamic_friction_range` 作为 MuJoCo torsional friction 的 best-effort 映射，仅在高维接触启用时生效。
  - `restitution_range` 暂不映射到 MuJoCo solver 参数，避免无依据改变接触求解稳定性。
- events 的 `asset_cfg` 映射已进一步对齐原 Hydra 配置：
  - `add_joint_default_pos`、`base_com`、`randomize_rigid_body_mass`、`physics_material` 会从原 event params 透传 robot/body/joint/geom/site/actuator 名称选择。
  - `sonic_release` 的 `randomize_rigid_body_mass` 现在明确保留 `.*wrist_yaw.*|torso_link` body 选择，不再退回默认全身匹配。
  - `randomize_joint_default_pos` 补齐原 SONIC 的 `distribution` 参数，并缓存 `default_joint_pos_nominal` 兼容原 event 语义。
  - `base_com` / `randomize_rigid_body_mass` 会透传 mjlab 支持的 `distribution`，并在配置声明时透传 `axes` / `shared_random`。
  - `push_robot` 现在也透传 `asset_cfg`，默认仍作用于 `robot`，与原 SONIC event 的资产选择语义保持一致。
- `randomize_rigid_body_mass` 已从 mjlab `dr.body_mass` 改为优先使用 `dr.pseudo_inertia`：
  - SONIC 的 `mass_distribution_params` scale range 会转换为 mjlab pseudo-inertia 的 `alpha_range`。
  - 当前 mass 与 inertia 会一起按密度尺度随机化，不再触发只改 mass、不改 inertia 的物理一致性警告。
- `feet_acc` 已从占位零奖励改为 mjlab `joint_acc_l2`，只惩罚 `.*ankle.*` 关节加速度；`sonic_release` 实验覆盖权重保持 `-2.5e-6`。
- 已补齐基础 push/randomization curriculum 迁移入口：
  - 新增 mjlab 原生 curriculum term，保留原 SONIC `step_curriculum` / `linear_curriculum` 数值规则。
  - 支持通过 `force_push_curriculum` / `force_push_linear_curriculum` 修改 mjlab `EventManager` 中已有 event 参数。
  - `param_path` 统一归一为 list 路径，避免字符串路径在 env cfg 与 curriculum term 之间重复解析时行为不一致。
  - 当前已验证可更新 `push_robot.params.velocity_range.x[0]`，后续可按相同路径扩展其它 push/randomization 参数。
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
- 已继续修正 motion batch reload 语义：
  - `use_paired_motions=True` 时 mjlab 与原 SONIC 一致加载完整 unique motion set，不再只加载 `num_envs` 条。
  - `SonicMotionCommand.refresh_after_motion_lib_reload()` 会在 evaluation/training 重新加载 motion 后刷新 `use_adaptive_sampling`、`motion_num_steps`、contact first-frame lookup、per-env first-contact 缓存和当前 reference state。
  - `SonicMjEnvWrapper.begin_seq_motion_samples()`、`forward_motion_samples()`、`set_is_training()` 会在 motion_lib reload 后调用上述刷新，避免 contact/object 缓存停留在旧 motion batch。
- 已补齐 TRL eval callback 需要的 mjlab wrapper 评估序列接口：
  - `SonicMjEnvWrapper` 维护 `start_idx`，与原 `ManagerEnvWrapper` 对齐。
  - `set_is_evaluating(True, global_rank=...)` 会调用 `motion_lib.load_motions_for_evaluation(start_idx=...)` 并 reset。
  - `forward_motion_samples(global_rank, world_size)` 会按 `world_size * num_envs` 推进 evaluation motion window。
  - `set_is_training()` 会重新加载 training motion subset 并在需要时 reset。
  - 已补齐 `motion_ids`、`get_env_data("ref_body_pos_extend")`、`get_env_data("rigid_body_pos_extend")` 和 `config.robot.actions_dim` 兼容面，供 `ImEvalCallback` 的评估循环读取。
- `eval_agent_trl.py` 已支持 mjlab evaluation 路径：
  - 顶层不再在缺少 IsaacLab 时直接退出；只有 IsaacSim evaluation 路径仍要求 IsaacLab。
  - `use_mjlab=True` / `sim_type=mjlab` 时跳过 Isaac AppLauncher，复用 `train_agent_trl.create_manager_env()` 创建 SonicMJ env。
  - `ImEvalCallback` 的 object scene 检测已兼容 mjlab scene 缺少 `rigid_objects` 的情况。
  - `ImEvalCallback` 在 uv 环境缺少 `smpl_sim` 时提供本地 MPJPE fallback，保证 `mpjpe_g/mpjpe_l/mpjpe_pa` 指标链路不断；fallback 中 `mpjpe_pa` 暂按 local MPJPE 近似。
  - SonicMJ command 暴露 `cmd_body_names`，wrapper 的 eval `get_env_data()` 已与原 `ManagerEnvWrapper` 的 body position 张量语义对齐。
  - `SonicMjEnvWrapper.set_is_evaluating()` 会在训练中的 eval callback 阶段临时移除 `manager_env.config.train_only_events`，退出 evaluation 后恢复，避免 `push_robot` 等训练专用 interval event 污染评估轨迹。
- 已继续补齐 `TrackingCommand` 的原 SONIC reset 语义：
  - reset 写入 MuJoCo 前会按 `pose_range`、`velocity_range`、`joint_position_range`、`joint_velocity_range` 对 root pose/velocity 和 joint pos/vel 做训练期随机化。
  - evaluation mode 会跳过上述 reset 随机化，保持可视化/评估起始帧稳定。
  - 支持 `start_from_first_frame`、`sample_from_n_initial_frames`、`sample_unique_motions`、`use_paired_motions` 的基础采样语义。
  - `encoder_sample_probs` 已从配置传入 mjlab command，`encoder_index` 改为与原 SONIC 一致的 multi-hot 形式；SMPL encoder 被采样时会同步激活 G1，并按 `teleop_sample_prob_when_smpl` 额外采样 teleop。
- 已补齐 `sonic_bones_seed` 需要的 SOMA 4-encoder 基础路径：
  - `TrackingCommand` 支持 SOMA encoder index、motion_has_soma availability-aware 采样，以及采到 SOMA 时同步激活 G1 做 latent alignment。
  - mjlab tokenizer 在配置包含 `encoder_sample_probs.soma` 时才加入 SOMA 观测项，避免影响 `sonic_release` 3-encoder 观测形状。
  - 新增 SOMA joints/root orientation/wrist joint tokenizer 观测，SOMA root quat 按原 SONIC 语义执行 Y-up 到 Z-up 转换和 BVH base rotation removal。
  - SOMA future state 改为直接使用 motion_lib 的 SOMA getter 和离散 future frame index，兼容 dummy SOMA 数据。
- 已补齐 `TrackingCommand` 的 variable future frames 与 contact-before 初始化基础语义：
  - 支持 `variable_frames_enabled` / `variable_frames_min` / `variable_frames_step`，reset 时为每个 env 重采样有效 future frame 数。
  - 新增 `command_num_frames` tokenizer 观测；仅当 actor backbone 的 encoder/decoder 配置声明 `mask` 特征时暴露，避免破坏现有 `sonic_release` tokenizer 维度。
  - 支持 `contact_file`、`sample_before_contact`、`sample_before_contact_margin`、`sample_before_contact_hand` 配置。
  - `contact_file` 支持单 pkl 或目录 pkl；缺省时会尝试从 motion_lib 的 `_motion_object_in_contact_left/right` 标签推导 first contact frame。
- 已修正 variable-frame token mask 链路：
  - 原 `UniversalTokenModule` 将 frame mask reshape 为 `max_num_tokens * 2**down_t`，在 SONIC 默认 `num_future_frames=10`、`max_num_tokens=2` 时会触发形状错误。
  - 现在 token mask 按 `num_frames * max_num_tokens / num_future_frames` 线性映射有效 token 数，不再要求 future frame 数与 token 分组整除。
  - `command_num_frames` 已在带 mask 的 mjlab tokenizer 中验证为 `(1,)`，默认无 mask 配置仍不暴露该项，保持 `sonic_release` tokenizer 维度不变。
- 已进一步补齐 contact-based initialization 诊断缓存：
  - mjlab `TrackingCommand` 会过滤 contact keys 到当前 loaded motion keys。
  - 支持 contact/motion 帧数容差校验，默认 `contact_frame_tolerance=3`。
  - 构建 `_motion_contact_flags`、`_first_contact_lookup` 和 `_per_env_first_contact`，便于后续真实 contact/object 数据 smoke 与调试。
- 已提交并推送到远程：
  - branch：`SONICMJ`
  - commit：`1b9cb72 Add SonicMJ mjlab migration path`
  - remote：`origin/SONICMJ`
- 已补齐 mjlab rough terrain 基础迁移：
  - `manager_env.config.terrain_type=trimesh` 会创建 mjlab `TerrainEntityCfg(terrain_type="generator")`。
  - 按原 SONIC `ROUGH_TERRAINS_CFG` 当前启用项映射 `boxes` 和 `random_rough` 两个子地形。
  - 默认保留原 SONIC `20x20`、`size=8.0`、`border_width=20.0` 配置。
  - 增加 `rough_terrain_num_rows`、`rough_terrain_num_cols`、`rough_terrain_size`、`rough_terrain_border_width` override，便于 smoke test 使用小地形快速验证。
- 已补齐顺序核对诊断入口：
  - `SonicMotionCommand` 初始化时打印 robot joint/body order、motion body order、motion DOF mapping 和 order checks。
  - `SonicMjEnvWrapper.get_order_diagnostics()` 返回结构化诊断，覆盖 mjlab robot joint/body order、motion body order、action term joint order、policy `joint_pos` / `joint_vel` / `actions` 顺序、action dim 和 observation shapes。
  - `SonicMjEnvWrapper.print_order_diagnostics()` 可在 smoke/eval 中直接打印完整诊断。
  - action manager 在 command 初始化阶段尚未完成构建时，command 侧 action order check 标记为 `pending_action_manager_init`；wrapper 诊断在 env 完整创建后执行最终校验。
- 已修正 mjlab reward 配置读取：
  - `tracking_anchor_pos`、`tracking_anchor_ori`、`tracking_relative_body_pos`、`tracking_relative_body_ori`、`tracking_body_linvel`、`tracking_body_angvel`、`tracking_vr_5point_local`、`action_rate_l2`、`joint_limit`、`feet_acc` 均从原 Hydra reward term 读取 `weight`。
  - tracking reward 的 `command_name` / `std` 参数从原配置透传到 mjlab reward term。
  - `joint_limit`、`feet_acc` 会把原 Isaac `asset_cfg` 转成 mjlab `SceneEntityCfg`。
  - 修正了此前 `tracking_vr_5point_local` 使用硬编码 `weight=1.0`、默认 `std=0.3` 的偏差；当前 `sonic_release` 读取为原 SONIC 的 `weight=2.0`、`std=0.1`。

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
- SOMA 4-encoder 基础路径验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/mdp/observations.py sonic_mj/env_cfg.py` 通过。
  - `git diff --check` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - `sonic_release` tokenizer `encoder_index` 仍为 `(3,)`，policy obs 仍为 `930`，critic obs 仍为 `1789`，action dim 仍为 `29`。
  - `sonic_bones_seed` + dummy SOMA tiny training 通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - `sonic_bones_seed` tokenizer `encoder_index` 为 `(4,)`，新增 SOMA term 形状：
    - `soma_joints_multi_future_local_nonflat`: `(10, 78)`
    - `soma_root_ori_b_multi_future`: `(10, 6)`
    - `joint_pos_multi_future_wrist_for_soma`: `(10, 6)`
  - `UniversalTokenModule` 已初始化 `g1`、`teleop`、`smpl`、`soma` 四个 encoder。
  - 已使用真实 SOMA BVH 数据 `/home/ykj/Downloads/dataset/bones-seed/soma_uniform/bvh/210531` 转换出 smoke PKL：
    `data/motion_lib_bones_seed/soma_uniform_smoke/210531`。
  - 转换结果：210 个 BVH 全部转换成功，0 failed；与 `data/motion_lib_bones_seed/robot_smoke` 的 motion key 覆盖为 210/210。
  - 真实 SOMA PKL smoke training 通过：
    `++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_uniform_smoke`。
  - 抽查转换产物 `soma_joints` 为非零数据，例如 shape `(1027, 26, 3)`、`abs_mean=0.1964`、`abs_max=0.9877`；`soma_root_quat` shape `(1027, 4)`。
- variable frames / contact-before 基础路径验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/mdp/observations.py sonic_mj/env_cfg.py` 通过。
  - `git diff --check` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - 新开关 smoke 通过：
    `++manager_env.commands.motion.variable_frames_enabled=true`
    `++manager_env.commands.motion.variable_frames_min=4`
    `++manager_env.commands.motion.variable_frames_step=2`
    `++manager_env.commands.motion.sample_before_contact=true`
  - 现有 `sonic_release` tokenizer 未声明 variable-frame mask，因此 smoke 中未额外暴露 `command_num_frames`；policy obs 仍为 `930`，critic obs 仍为 `1789`，action dim 仍为 `29`。
- variable-frame mask 完整链路修正后验证：
  - 复现并修复了带 mask 配置下的 token mask reshape 错误：
    `RuntimeError: shape '[-1, 2, 4]' is invalid for input of size 20`。
  - `uv run python -m py_compile gear_sonic/trl/modules/universal_token_modules.py sonic_mj/env_cfg.py sonic_mj/mdp/commands.py sonic_mj/mdp/observations.py` 通过。
  - `git diff --check` 通过。
  - compose 检查确认带 mask 配置时 tokenizer terms 包含 `command_num_frames`，且 `variable_frames_enabled=True`、`variable_frames_min=4`、`variable_frames_step=2`。
  - 带 mask 的 `sonic_release` tiny training 通过：
    `num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`、`terrain_type=plane`、
    `encoders.g1.mask=[command_multi_future_nonflat]`、
    `decoders.g1_kin.mask=[command_multi_future_nonflat]`。
  - 无 mask 的 `sonic_release` tiny training 回归通过；tokenizer 中未出现 `command_num_frames`，policy obs 仍为 `930`，critic obs 仍为 `1789`，action dim 仍为 `29`。
- contact 诊断缓存补齐后验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/env_cfg.py sonic_mj/mdp/observations.py` 通过。
  - `git diff --check` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - `sample_before_contact=true` fallback smoke 通过；当前本机 `data` 与 `/home/ykj/Downloads/dataset/bones-seed` 下未找到明显真实 contact 文件，因此真实 contact/object 数据校验仍未完成。
- physics material bucket 迁移后验证：
  - `uv run python -m py_compile sonic_mj/mdp/events.py sonic_mj/env_cfg.py` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - EventManager 正常加载 `physics_material` startup event。
  - action dim 仍为 `29`，policy obs 仍为 `930`，critic obs 仍为 `1789`。
  - termination 日志中 `anchor_pos=0.0000`、`anchor_ori_full=0.0000`、`ee_body_pos=0.0000`、`foot_pos_xyz=0.0000`。
- curriculum 迁移后验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/curriculum.py sonic_mj/mdp/events.py` 通过。
  - 带 `force_push_curriculum` override 的 `sonic_release` tiny training 通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - `CurriculumManager` 正常加载 `force_push_curriculum`。
  - 训练日志出现 `Env/Curriculum/force_push_curriculum: -0.5000`，确认 curriculum term 能写入并上报状态。
- curriculum 路径归一修正后回归验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/curriculum.py sonic_mj/mdp/events.py` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - 带 `force_push_curriculum` override 的 tiny training 回归通过，`CurriculumManager` 正常加载，日志仍上报 `Env/Curriculum/force_push_curriculum: -0.5000`。
  - `sonic_release` `num_envs=16`、`num_learning_iterations=10`、默认 `num_steps_per_env=24` 短训练回归通过，完成 `Learning iteration 10`，total timesteps `3840`，未出现 NaN、`nefc overflow` 或训练崩溃。
- eval wrapper 接口补齐后验证：
  - `uv run python -m py_compile sonic_mj/wrapper.py sonic_mj/mdp/commands.py sonic_mj/env_cfg.py sonic_mj/mdp/curriculum.py sonic_mj/mdp/events.py` 通过。
  - inline smoke 创建 `sonic_release` mjlab env，`num_envs=2`，`motion_file=data/motion_lib_bones_seed/robot_smoke`，`smpl_motion_file=dummy`。
  - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action space `(2, 29)`。
  - `set_is_evaluating(True, global_rank=0)` 后 `start_idx=0`，`motion_ids=[0, 1]`。
  - `forward_motion_samples(global_rank=0, world_size=1)` 后 `start_idx=2`，`motion_ids=[0, 1]`，说明 evaluation window 已前移且局部 motion id 仍按 env 顺序遍历。
  - `set_is_training()` 后 `motion_command.is_evaluating=False`，可重新进入 training motion subset。
  - `git diff --check` 通过。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`，完成 `Learning iteration 1`，total timesteps `4`。
  - tiny training 中 action dim 仍为 `29`，policy obs 仍为 `930`，critic obs 仍为 `1789`，termination 日志均为 `0.0000`。
- eval callback 数据接口补齐后验证：
  - `uv run python -m py_compile sonic_mj/wrapper.py sonic_mj/mdp/commands.py sonic_mj/env_cfg.py` 通过。
  - inline smoke 中 `env.config.robot.actions_dim=29`，`get_env_data("ref_body_pos_extend")=(2, 30, 3)`，`get_env_data("rigid_body_pos_extend")=(2, 30, 3)`。
  - `env.step({"obs": obs, "actions": zeros, "step": 0})` 能按 eval callback actor_state 形式执行，返回 reward/done shape 均为 `(2,)`。
  - `set_is_evaluating(True, global_rank=0)` 后 `start_idx=0`、`motion_ids=[0, 1]`；`forward_motion_samples(global_rank=0, world_size=1)` 后 `start_idx=2`、`motion_ids=[0, 1]`。
  - `sonic_release` tiny training 回归通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`，完成 `Learning iteration 1`，total timesteps `4`。
- 2026-04-29 使用当前 `data/` 下真实/本地数据重新验证 mjlab 路径：
  - 数据计数：
    - `data/motion_lib_bones_seed/robot_smoke`：210 个 robot motion pkl。
    - `data/motion_lib_bones_seed/soma_uniform_smoke`：210 个 SOMA motion pkl。
    - `data/smpl_filtered`：131455 个 SMPL pkl。
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/wrapper.py sonic_mj/mdp/commands.py sonic_mj/mdp/observations.py sonic_mj/mdp/rewards.py sonic_mj/mdp/terminations.py sonic_mj/mdp/events.py sonic_mj/mdp/curriculum.py gear_sonic/train_agent_trl.py` 通过。
  - 直接使用 `sonic_release` 默认 `terrain_type=trimesh` 做 smoke 时，mjlab terrain generation 本机耗时约 `107.8s`；快速 smoke 建议显式加 `manager_env.config.terrain_type=plane`，避免把验证时间耗在 rough terrain 生成上。
  - `sonic_release` inline reset/step smoke 通过：
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=data/smpl_filtered`
    - `manager_env.config.terrain_type=plane`
    - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、`actions_dim=29`。
    - `motion_lib.smpl_data_keys` 覆盖当前 210 个 robot smoke motion key。
    - 单步 step 返回 reward/done shape 均为 `(2,)`。
  - `sonic_release` tiny PPO training 通过：
    - `num_envs=2`
    - `num_learning_iterations=1`
    - `num_steps_per_env=2`
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=data/smpl_filtered`
    - `manager_env.config.terrain_type=plane`
    - 完成 `Learning iteration 1`，total timesteps `4`。
  - `sonic_bones_seed` + SOMA 4-encoder tiny PPO training 通过：
    - `num_envs=2`
    - `num_learning_iterations=1`
    - `num_steps_per_env=2`
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `soma_motion_file=data/motion_lib_bones_seed/soma_uniform_smoke`
    - `manager_env.config.terrain_type=plane`
    - tokenizer `encoder_index=(4,)`，SOMA tokenizer terms 正常出现。
    - `UniversalTokenModule` 初始化 `g1`、`teleop`、`smpl`、`soma` 四个 encoder。
    - 完成 `Learning iteration 1`，total timesteps `4`。
  - 极小 batch 仍会出现 PPO 提示：`batch_size` 不能整除 `num_mini_batches`，属于 `num_envs=2`、`num_steps_per_env=2` 的 smoke 配置尺寸过小导致，不影响本次闭环验证。
- checkpoint 加载 smoke：
  - 临时在 `/tmp/sonicmj_checkpoint_smoke` 用 `++callbacks.model_save.save_last_frequency=1` 生成同结构 `last.pt`。
  - 随后以 `checkpoint=/tmp/sonicmj_checkpoint_smoke/last.pt` 启动 mjlab `sonic_release` tiny training。
  - 日志确认 `Loading checkpoint from /tmp/sonicmj_checkpoint_smoke/last.pt` 和 `Loaded checkpoint from step 1`。
  - 加载后训练完成 `Learning iteration 1`，total timesteps `4`，policy/value/env_state_dict 加载链路未报错。
  - 临时目录 `/tmp/sonicmj_checkpoint_smoke` 和 `/tmp/sonicmj_checkpoint_smoke_load` 已删除。
- 完整 eval callback smoke：
  - 临时在 `/tmp/sonicmj_eval_callback_smoke` 生成同结构 `last.pt`。
  - 使用 `gear_sonic/eval_agent_trl.py`、`use_mjlab=True`、`sim_type=mjlab`、`eval_callbacks=im_eval`、`run_eval_loop=False` 启动评估。
  - 使用 `filter_motion_keys` 将 smoke 限制为 2 条 motion，`num_envs=2`。
  - callback 成功完成 evaluation loop，写出 `/tmp/sonicmj_eval_callback_out/metrics_eval.json`。
  - metrics 中包含 `eval/all/mpjpe_g`、`eval/all/mpjpe_l`、`eval/all_metrics_dict.motion_keys`、`sampling_prob` 等 eval 后处理所需字段。
  - 因临时 checkpoint 基本未训练，2 条 motion 中有失败终止，`eval/success/*` 为 NaN；这属于 smoke policy 质量问题，不是链路错误。
  - 临时目录 `/tmp/sonicmj_eval_callback_smoke` 和 `/tmp/sonicmj_eval_callback_out` 已删除。
- rough terrain 小网格 smoke：
  - `uv run python -m py_compile sonic_mj/env_cfg.py` 通过。
  - `git diff --check` 通过。
  - 使用 `terrain_type=trimesh` 默认路径，并 override `rough_terrain_num_rows=1`、`rough_terrain_num_cols=1`。
  - tiny training 通过：`num_envs=2`、`num_learning_iterations=1`、`num_steps_per_env=2`。
  - 日志确认 `Terrain generation took 0.0056 seconds`，mjlab rough terrain generator 正常创建。
  - action dim 仍为 `29`，policy obs 仍为 `930`，critic obs 仍为 `1789`。
  - termination 日志中 `anchor_pos=0.0000`、`anchor_ori_full=0.0000`、`ee_body_pos=0.0000`、`foot_pos_xyz=0.0000`。
- reward 配置读取修正后验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/rewards.py` 通过。
  - 直接 compose `sonic_release` mjlab env cfg，确认 RewardTermCfg：
    - `tracking_vr_5point_local`: `weight=2.0`、`params={"command_name": "motion", "std": 0.1}`
    - `feet_acc`: `weight=-2.5e-06`、`asset_cfg.joint_names=[".*ankle.*"]`
  - inline env reset/step smoke 通过：`actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action dim `29`、reward/done shape 均为 `(2,)`。
  - RewardManager 表中 `tracking_vr_5point_local` 权重为 `2.0`，step 后读取 term params 为 `{"command_name": "motion", "std": 0.1}`。
  - 一次 `accelerate` tiny training 在 motion_lib 初始化后超过常规等待时间无进一步输出，已停止该进程；随后用 inline env reset/step smoke 完成验证。
- event `asset_cfg` 透传修正后验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/events.py` 通过。
  - `git diff --check` 通过。
  - compose `sonic_release` mjlab env cfg，确认 event 映射：
    - `add_joint_default_pos`: `joint_names=[".*"]`，保留 `operation=add` 和 `distribution=uniform`。
    - `base_com`: `body_names="torso_link"`。
    - `randomize_rigid_body_mass`: `body_names=".*wrist_yaw.*|torso_link"`，`mode=startup`。
    - `physics_material`: `body_names=".*"`。
  - inline env reset/step smoke 通过：
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=dummy`
    - `manager_env.config.terrain_type=plane`
    - reset/step 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action dim `29`。
    - EventManager 正常加载 `reset_scene_to_default`、`add_joint_default_pos`、`base_com`、`randomize_rigid_body_mass`、`physics_material`、`push_robot`。
    - step 返回 reward/done shape 均为 `(2,)`。
- event 剩余参数透传修正后验证：
  - `uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/events.py` 通过。
  - `git diff --check` 通过。
  - compose `sonic_release` mjlab env cfg，确认：
    - `base_com` 保留 `distribution=uniform`、`operation=add`。
    - `randomize_rigid_body_mass` 使用 `pseudo_inertia` 时保留 `distribution=uniform`。
    - `push_robot` params 中包含默认 `asset_cfg=robot`。
  - inline env reset/step smoke 通过：
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=dummy`
    - `manager_env.config.terrain_type=plane`
    - reset/step 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action dim `29`。
    - step 返回 reward/done shape 均为 `(2,)`。
- 顺序诊断入口补齐后验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/wrapper.py` 通过。
  - `git diff --check` 通过。
  - inline `sonic_release` mjlab env reset/step smoke 通过：
    - `num_envs=2`
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=dummy`
    - `manager_env.config.terrain_type=plane`
    - `get_order_diagnostics()["checks"]` 全部为 `True`：
      robot joint/body order、motion body order、action joint order、policy `joint_pos` / `joint_vel` / `actions` order、motion DOF identity mapping、action dim 29。
    - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`。
    - step 返回 reward/done shape 均为 `(2,)`。
- paired motion / motion reload 缓存修正后验证：
  - `uv run python -m py_compile sonic_mj/mdp/commands.py sonic_mj/wrapper.py` 通过。
  - `git diff --check` 通过。
  - inline `sonic_release` mjlab env smoke 通过：
    - `num_envs=2`
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=dummy`
    - `manager_env.config.terrain_type=plane`
    - `manager_env.commands.motion.use_paired_motions=true`
    - `manager_env.commands.motion.sample_before_contact=true`
    - 初始化时 `num_unique=210`、`num_loaded=210`、`max_num_load_motions=210`。
    - `set_is_evaluating(True, global_rank=0)` 后 eval batch 为前 2 条 motion；`forward_motion_samples(global_rank=0, world_size=1)` 后推进到第 3/4 条 motion。
    - `set_is_training()` 后重新加载 210 条 training motion，contact lookup 缓存长度与当前 loaded motion keys 保持一致。
    - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`；step 返回 reward/done shape 均为 `(2,)`。
- eval callback train-only event 切换验证：
  - `uv run python -m py_compile sonic_mj/wrapper.py sonic_mj/env_cfg.py sonic_mj/mdp/events.py sonic_mj/mdp/commands.py` 通过。
  - `git diff --check` 通过。
  - inline `sonic_release` mjlab env smoke 通过：
    - `num_envs=2`
    - `motion_file=data/motion_lib_bones_seed/robot_smoke`
    - `smpl_motion_file=dummy`
    - `manager_env.config.terrain_type=plane`
    - 初始 `EventManager.active_terms["interval"]=["push_robot"]`。
    - `set_is_evaluating(True, global_rank=0)` 后 `interval` terms 变为空，确认 `push_robot` 被临时移除。
    - `set_is_evaluating(False)` 后 `interval` terms 恢复为 `["push_robot"]`。
    - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`、action dim `29`；step 返回 reward/done shape 均为 `(2,)`。
- 2026-05-03 迁移工作树恢复与 event 对照推进：
  - 当前 `main` 未包含 `origin/SONICMJ` 中的 SonicMJ 迁移文件，已从 `origin/SONICMJ` 恢复迁移工作树：
    `sonic_mj/`、`process.md`、`uv.lock`、`.python-version`、`AGENTS.md`、mjlab train/eval 入口、eval callback 兼容、UniversalTokenModule mask 修复和相关 pyproject/base 配置。
  - `uv run` 环境可导入 `mjlab`、`torch`、`mujoco`；base Python 环境缺少 `mjlab`，因此后续验证继续使用项目 uv 环境。
  - 新增 `docs/sonic_mj_event_migration.md`，把当前 event 迁移标为：
    - 等价：`reset_scene_to_default`、`add_joint_default_pos`、`base_com`、`push_robot`
    - 近似：`randomize_rigid_body_mass`、`physics_material`
    - 已知未映射：`physics_material.restitution_range`
  - event 对照结论：当前训练必要 event 已覆盖，剩余风险主要是 MuJoCo 与 PhysX 接触/材质语义差异和 mass pseudo-inertia 分布差异。
  - 恢复后重新验证：
    - `uv run python -m py_compile ...` 覆盖 `sonic_mj`、train/eval 入口、eval callback 和 UniversalTokenModule，通过。
    - `git diff --check` 通过。
    - Hydra compose 检查可构建 `sonic_release` mjlab env cfg，events 为 `reset_scene_to_default`、`add_joint_default_pos`、`base_com`、`randomize_rigid_body_mass`、`physics_material`、`push_robot`。
    - inline reset/step smoke 使用 `num_envs=2`、`terrain_type=plane`、`motion_file=data/motion_lib_bones_seed/robot_smoke`、`smpl_motion_file=dummy` 通过。
    - reset 后 `actor_obs=(2, 930)`、`critic_obs=(2, 1789)`，action space `(2, 29)`，step 返回 reward/done shape 均为 `(2,)`。
    - `get_order_diagnostics()["checks"]` 全部为 `True`，确认 robot/motion/action/policy 顺序和 action dim 没有因恢复工作树回退。

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
- curriculum：`sonic_mj/mdp/curriculum.py`
- event 迁移对照：`docs/sonic_mj_event_migration.md`
- 训练入口分支：`gear_sonic/train_agent_trl.py`
- eval 入口分支：`gear_sonic/eval_agent_trl.py`
- eval callback 兼容：`gear_sonic/trl/callbacks/im_eval_callback.py`
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
uv run python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
  --input /home/ykj/Downloads/dataset/bones-seed/soma_uniform/bvh/210531 \
  --output data/motion_lib_bones_seed/soma_uniform_smoke/210531 \
  --fps 30 \
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
uv run python -m py_compile sonic_mj/env_cfg.py sonic_mj/mdp/rewards.py
```

```bash
uv run python - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path
from gear_sonic.utils.config_utils import register_rl_resolvers
from sonic_mj.env_cfg import make_sonic_mj_env_cfg

register_rl_resolvers()
with initialize_config_dir(version_base=None, config_dir=str(Path('gear_sonic/config').resolve())):
    cfg = compose(config_name='base', overrides=[
        '+exp=manager/universal_token/all_modes/sonic_release',
        'use_mjlab=True',
        'sim_type=mjlab',
        'checkpoint=null',
        'num_envs=2',
        'headless=True',
        '++algo.config.num_learning_iterations=1',
        '++algo.config.num_steps_per_env=2',
        '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke',
        '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy',
    ])
env_cfg = make_sonic_mj_env_cfg(cfg)
for name, term in env_cfg.rewards.items():
    print(name, 'weight=', term.weight, 'params=', term.params or {})
PY
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bones_seed \
  use_mjlab=True \
  sim_type=mjlab \
  checkpoint=null \
  num_envs=2 \
  headless=True \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_steps_per_env=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=dummy
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bones_seed \
  use_mjlab=True \
  sim_type=mjlab \
  checkpoint=null \
  num_envs=2 \
  headless=True \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_steps_per_env=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_uniform_smoke
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
  ++manager_env.commands.motion.variable_frames_enabled=true \
  ++manager_env.commands.motion.variable_frames_min=4 \
  ++manager_env.commands.motion.variable_frames_step=2 \
  ++manager_env.commands.motion.sample_before_contact=true \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/eval_agent_trl.py \
  +checkpoint=/tmp/sonicmj_eval_callback_smoke/last.pt \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  ++eval_callbacks=im_eval \
  ++run_eval_loop=False \
  ++num_envs=2 \
  ++eval_output_dir=/tmp/sonicmj_eval_callback_out \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy \
  '++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[rage_professionall_001__A002,walk_forward_hips_amplified_002__A001_M]'
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
  ++manager_env.config.rough_terrain_num_rows=1 \
  ++manager_env.config.rough_terrain_num_cols=1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

## 待解决问题

### 高优先级

- termination 已按原 SONIC term 类型、body 选择和阈值来源对齐；body 映射、quaternion 顺序、reset 首帧 stale pose 和 DOF 写入顺序已修正。当前 reset FK 误差已到 `1e-6m` 量级，`ee_body_pos`、`foot_pos_xyz` 首帧误触发问题已解决；后续仍需在更长训练中观察随机动作导致的真实 tracking failure 分布。
- `physics_material` 已保留 bucket 采样，并完成 static friction 到 MuJoCo sliding friction 的明确映射；Isaac 的 dynamic friction/restitution 仍没有 MuJoCo 完全等价表达，目前 dynamic friction 只作为 torsional friction best-effort 映射，restitution 暂不映射。
- `randomize_rigid_body_mass` 当前已对 `scale` 操作使用 mjlab `dr.pseudo_inertia`，比 Isaac 只缩放 mass 的事件更物理一致；后续长训练仍需观察该差异是否影响 SONIC 原始训练分布。
- `TrackingCommand` 已初步对齐 adaptive sampling、evaluation mode、paired motion 完整加载和 motion reload 后的 contact 缓存刷新；后续仍需核对真实 contact/object 数据下的 contact-based initialization，以及更长训练下的 adaptive sampling 分布是否和原始 `gear_sonic` 一致。
- `TrackingCommand` 已补齐基础 reset 随机化、start-time override、unique/paired motion 采样、encoder multi-hot、SOMA 4-encoder 基础采样、contact-before 初始化、contact 诊断缓存和 variable future frames 基础语义；`command_num_frames` 到 tokenizer mask 的 mjlab 完整链路已用带 mask actor config 跑通，后续仍需用真实 contact/object 数据验证 contact-based initialization。
- eval callback 已可在 mjlab 路径跑通并产出 metrics；当前 uv 环境缺少 `smpl_sim` 时使用本地 MPJPE fallback，`mpjpe_pa` 只是 local MPJPE 近似，若需要与原 Isaac/SMPL eval 完全一致，仍需按项目环境安装/接入官方 `smpl_sim`。
- 训练中的 eval callback 现在会在 `set_is_evaluating(True)` 阶段临时禁用 `train_only_events` 并在返回训练模式时恢复；当前已验证 `push_robot` 不会在 evaluation interval 中触发。

### 中优先级

- rough terrain/trimesh terrain 已补齐基础 generator 路径，并通过小网格 smoke；默认 `20x20` 大地形尚未做完整训练回归，后续需要确认启动耗时和接触稳定性。
- curriculum 已补齐基础 push/randomization event 参数更新入口；尚未用真实课程配置覆盖所有 push/randomization 参数组合。
- events/domain randomization 还需要更细致对齐原 IsaacLab：
  - physics material dynamic friction / restitution 的物理近似差异
  - reset-time vs startup-time 随机化
  - body/geom/name 选择、push asset 选择和 mjlab 支持的 distribution/axes/shared_random 参数已按当前 SONIC event 配置透传；后续如果新增 hand/object event，还需要逐项复查对应资产命名。
- `open3d` 当前未安装，mesh 加载做成缺失时跳过；mesh 可视化和精细 FK 相关能力后续需要补齐环境依赖或替代实现。
- SOMA 已用 `/home/ykj/Downloads/dataset/bones-seed/soma_uniform` 的 210531 子集完成真实数据 smoke；后续仍需转换/验证全量 SOMA 数据，并在更长训练中观察 SOMA encoder 采样比例和数值稳定性。
- 已新增结构化顺序诊断入口，当前 smoke 中 motion_lib body/joint order、mjlab robot body/joint order、policy obs/action order 均核对通过；后续仍需在更大数据集和 checkpoint 加载场景下周期性调用该诊断确认未回退。

### 低优先级

- 清理或降噪第三方 warning：
  - `PPOConfig` TRL import path warning
  - `torch.cuda.amp.autocast` deprecation warning
  - mjlab/mujoco_warp 部分 API deprecation warning
- 可补充 README 或训练文档，说明 mjlab 分支的启动命令、uv 环境和已知限制。
- 后续如需长期训练，建议增加脚本化 smoke test，但测试脚本创建后必须按 AGENTS.md 要求测试完删除。

## 下一步建议

1. 再跑更长一点的小规模训练，观察是否还有 `nefc overflow`、NaN、异常 reset 或 reward 退化。
2. 继续核对 motion cache、paired motions、contact-based initialization 与原始 `gear_sonic` 的行为一致性，尤其是真实 contact/object 数据下的 first-contact key 匹配。
3. 转换全量 `/home/ykj/Downloads/dataset/bones-seed/soma_uniform/bvh`，重跑更大 `sonic_bones_seed` smoke，检查 SOMA encoder 采样比例和 SOMA 观测数值。
4. 准备或找到带 tokenizer `mask` 字段的 variable-frame actor 配置，验证 `command_num_frames` 到 `UniversalTokenModule` frame/token mask 的完整训练链路。
5. 如需要严格复现原 eval 的 PA-MPJPE，按项目 uv 环境补齐 `smpl_sim` 依赖并回归对比 fallback 与官方指标。
