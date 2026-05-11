# PROGRESS

记录 workflow agent 的进度、测试、阻塞和恢复点。

## 2026-05-08T21:19:39+08:00 - workflow 恢复检查

### 已读取
- `AGENTS.md`：确认 SonicMJ 迁移约束；项目迁移环境使用 uv，不使用 conda；不得修改 CUDA/显卡驱动/系统级 GPU 环境；workflow 运行时必须先处理 `PERSON.md` open 条目。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md`：原状态中用户需求、验收条件、Python 环境均未补全。
- `PLAN.md`：尚未生成计划。
- `task.json`：`request_text=null`，`plan_approved=false`，`last_result=init`，环境字段为空。
- `.workflow/runtime/resume-prompt.txt`：内容与当前恢复指令一致。
- `.workflow/artifacts`：恢复前不存在 artifact 文件。
- `README.md` / `docs/README.md` / `pyproject.toml` / `gear_sonic/pyproject.toml`：确认根项目 `sonicmj` 使用 Python `>=3.10,<3.11`，根 `pyproject.toml` 有 uv 配置和 `uv.lock`；README 中原上游训练仍描述 Isaac Lab，但本项目 `AGENTS.md` 明确 SonicMJ 的 mjlab 迁移使用 uv。
- git 状态：分支 `SONICMJ...origin/SONICMJ`；已有未提交改动 `AGENTS.md`，新增 workflow 文件 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 判断
- 当前没有 `PERSON.md` open 条目需要处理。
- 由于 `request_text=null` 且 `PLAN.md` 未生成，不能直接执行代码迁移或测试；否则会绕过 workflow 的任务确认和计划阶段。
- Python 环境可按项目事实登记为 `uv` + `.venv` + `existing`，但还不能安装依赖或启动训练。
- 最合理下一步是阻塞等待用户明确本轮项目目标，或确认按 `AGENTS.md` 的 SonicMJ/mjlab 迁移顺序生成计划。

### 本轮命令与结果
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md`：成功读取 workflow 必读文档。
- `jq . task.json`：成功读取轻量运行摘要。
- `find .workflow ...`：恢复前仅有 `.workflow/runtime/resume-prompt.txt`。
- `git status --short --branch` / `git status --porcelain=v1`：确认当前工作区已有 workflow 初始化文件和 `AGENTS.md` 修改。
- `rg --files -g 'README*' -g 'pyproject.toml' -g 'uv.lock' ...`：定位必要 README 和环境配置。
- `sed -n ... README.md docs/README.md pyproject.toml gear_sonic/pyproject.toml`：确认根项目 uv 环境事实。
- `mkdir -p .workflow/artifacts`：创建 artifact 目录用于记录阻塞问题。

### blocked_reason
缺少可执行项目任务：`task.json.request_text` 为空，`TASK.md` 只有占位信息，`PLAN.md` 尚未生成。需要用户提供具体要推进的 SonicMJ/mjlab 迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成计划”。

## 2026-05-08T21:23:04+08:00 - grounding / plan / review 完成，等待 PLAN 确认

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复 workflow 状态，并确认本轮用户已明确要求继续 SonicMJ 到 mjlab 迁移。
- `README.md`、`docs/README.md`、`pyproject.toml`：确认项目环境事实为 uv + `.venv`，根项目依赖含 `mjlab==1.2.0`。
- `sonic_mj/assets.py`、`sonic_mj/env_cfg.py`、`sonic_mj/mdp/commands.py`、`sonic_mj/mdp/observations.py`、`sonic_mj/mdp/rewards.py`、`sonic_mj/mdp/terminations.py`、`sonic_mj/mdp/events.py`、`sonic_mj/wrapper.py`、`sonic_mj/train.py`：确认当前 mjlab 迁移骨架和待验证面。
- `gear_sonic/train_agent_trl.py`：确认 `use_mjlab=True` / `sim_type=mjlab` 已接入 `sonic_mj.train.create_mjlab_manager_env`。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`：确认第一阶段目标是 G1、teleop、SMPL 3 encoder。
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` 与 `gear_sonic/envs/manager_env/mdp/`：作为原 IsaacLab 语义参考。
- Zotero：搜索 `GEAR SONIC humanoid behavior foundation model`，找到 SONIC 论文 item `GT7EYT6G`，URL `https://arxiv.org/pdf/2511.07820`。

### 已完成
- 更新 `TASK.md`：补全用户需求、硬约束、当前代码事实、参考资料和验收条件。
- 写入 `.workflow/artifacts/grounding.md`：记录项目现状、关键证据、参考资料和风险。
- 写入 `.workflow/artifacts/paper_project_map.json`：结构化映射 SONIC 论文、原实现、目标代码和验证点。
- 写入 `.workflow/artifacts/open_questions.json`：按 `{ "questions": [...] }` 格式记录非 blocking 的计划确认问题；当前没有阻塞生成计划的问题。
- 生成 `PLAN.md`：包含 M0-M8 里程碑、完成标准、测试命令和 review 点。
- 写入 Codex 多角色 review：
  - `.workflow/artifacts/reviews/scope.md`
  - `.workflow/artifacts/reviews/milestones.md`
  - `.workflow/artifacts/reviews/architecture.md`
  - `.workflow/artifacts/reviews/testing.md`
  - `.workflow/artifacts/reviews/executor.md`
- 更新 `task.json`：写入本轮 request，标记 `plan_ready_waiting_user_confirmation`。

### 判断
- 没有 blocking open question；但 workflow 规则要求当前阶段停在等待用户确认 `PLAN.md`。
- 尚未进入代码修改或测试执行；本轮只改 workflow 文档和 artifacts。
- 后续执行必须使用 `uv run`，不得安装到 base，不得触碰 CUDA/显卡驱动/系统 GPU 组件。

### 本轮命令与结果
- `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh resume --workspace "$PWD"`：成功，输出 `phase=resumed`，`plan_approved=false`。
- `sed -n ...`：读取 workflow 必读文档、README、pyproject 和关键代码文件成功。
- `find .workflow -maxdepth 4 -type f`：确认已有 `.workflow/artifacts/open_questions.json` 和 runtime 文件。
- `git status --short`：确认已有未提交 workflow 初始化文件和 `AGENTS.md` 修改；本轮未回退任何已有改动。
- `zotero_search_items` / `zotero_item_fulltext`：找到并读取 SONIC 论文条目。
- `mkdir -p .workflow/artifacts/reviews`：创建 review artifact 目录。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后，从 `PLAN.md` 的 M0 开始执行环境确认、静态检查、asset/order 验证、reset/step smoke、obs shape 对齐和短训练 smoke。

## 2026-05-08T21:22:12+08:00 - 短生命周期 agent 恢复判断

### 已读取
- `AGENTS.md`：确认本项目 SonicMJ 迁移使用 uv + `.venv`，不得修改 CUDA/显卡驱动/系统级 GPU 组件，workflow 恢复必须先处理 `PERSON.md` open 条目。
- `PERSON.md`：无实际 open 条目，只有模板。
- `TASK.md`：仍只记录“恢复 workflow”占位需求，未包含可执行开发任务。
- `PLAN.md`：仍未生成计划。
- `PROGRESS.md`：上一轮已记录缺少任务目标并阻塞。
- `task.json`：`request_text=null`，`plan_approved=false`，环境为 `uv` + `.venv` + `existing`。
- `.workflow/artifacts/open_questions.json`：存在 `blocking: true` 的 `Q0001`，要求用户提供本轮任务或确认按 `AGENTS.md` 建议迁移顺序生成计划。
- `README.md` / `docs/README.md` / `pyproject.toml` / `.python-version`：确认上游 README 仍描述 Isaac Lab 训练，项目根配置为 `sonicmj`、Python 3.10、uv 环境；当前 SonicMJ 迁移约束以 `AGENTS.md` 为准。

### 判断
- 当前用户消息是恢复 workflow 的运行指令，没有回答 `Q0001`。
- `open_questions.json` 中已有 blocking 问题，按 workflow 规则必须停止等待用户补充，不能擅自生成计划、改代码或跑训练测试。
- 当前最合理下一步是保持阻塞状态，并要求用户提供具体迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成 PLAN.md”。

### 本轮命令与结果
- `sed -n '1,220p' /home/ykj/tool/WORKFLOW/skills/workflow/SKILL.md`：成功读取 workflow skill 规则；未修改 workflow 源码仓库。
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md`：成功读取 workflow 必读文档。
- `jq . task.json`：成功读取运行摘要，确认任务请求仍为空。
- `find .workflow -maxdepth 3 -type f -print | sort`：确认 artifact/runtime 文件存在。
- `git status --short --branch`：确认分支 `SONICMJ...origin/SONICMJ`，已有未提交 workflow 文件与 `AGENTS.md` 修改。
- `rg --files -g 'README*' -g 'docs/**' -g 'pyproject.toml' -g 'uv.lock' -g '.python-version'`：定位必要 README、docs 和环境配置。
- `sed -n ... .workflow/artifacts/open_questions.json README.md pyproject.toml .python-version docs/README.md`：确认阻塞问题和 uv/Python 环境事实。
- `date --iso-8601=seconds`：记录本轮时间戳。

### blocked_reason
`.workflow/artifacts/open_questions.json` 存在 blocking 问题 `Q0001`，且当前仍缺少可执行项目任务。需要用户提供本轮要推进的具体 SonicMJ/mjlab 迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成 PLAN.md”。

## 2026-05-08T21:42:33+08:00 - PLAN.md 代码阶段 M0-M8 执行完成

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认 `plan_approved=true`，环境为 `uv` + `.venv` + `existing`。
- `PERSON.md`：没有实际 open 条目，只有模板；无需同步新增需求。
- `README.md`、`docs/README.md`、`pyproject.toml`：确认根项目使用 uv，`mjlab==1.2.0`，不使用 conda，不安装依赖到 base。
- `.workflow/artifacts/open_questions.json`：仅有非 blocking 问题；本轮按已确认 PLAN 执行。

### M0 - 基线恢复与环境确认
- `uv run python --version`：通过，输出 `Python 3.10.20`。
- `uv run python -c "import mjlab, torch, mujoco; ..."`：通过，输出 `ok mjlab unknown torch 2.11.0+cu130 mujoco 3.8.1`。
- `git status --short --branch`：分支 `SONICMJ...origin/SONICMJ`；已有 `M AGENTS.md` 和未跟踪 workflow 文档/目录。本轮未回退任何已有改动。
- 数据路径检查：仓库内存在 `data/motion_lib_bones_seed/robot_smoke`、`data/motion_lib_bones_seed/soma_uniform_smoke`、`data/smpl_filtered`，可用于 smoke。

### M1 - 静态对齐与导入健康检查
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`：通过。
- Hydra compose `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16` 并调用 `make_sonic_mj_env_cfg`：通过。
- 构造结果：`ManagerBasedRlEnvCfg`，actions `['joint_pos']`，commands `['motion']`，observations `['policy', 'critic', 'tokenizer']`，action dim `29`。
- 诊断脚本中第一次访问 `cfg.experiment_name` 时触发 `HydraConfig was not set`；这是非 Hydra 正式入口访问 `${hydra:runtime...}` 字段导致，调整脚本不解析该字段后通过，不需要改项目代码。

### M2 - G1 资产、顺序和动作闭环
- XML 静态核对命令解析 `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`：通过。
- 结果：XML actuator count `29`，`SONIC_G1_JOINT_NAMES` count `29`，body count `30`，actuator order 与 `sonic_mj/assets.py` canonical joint order 完全一致。
- 按 regex 展开 `SONIC_G1_ACTION_SCALE` 和 `SONIC_G1_DEFAULT_JOINT_POS` 后均为 29 维；action scale 全部为 `0.5`。
- reset smoke 中 `env.print_order_diagnostics()` 输出所有 order checks 为 True：robot joints/body、motion bodies、action joints、policy `joint_pos`/`joint_vel`/`actions` 顺序均匹配 SONIC MuJoCo order，action dim 为 29。

### M3 - MotionCommand reset/step smoke
- 命令：用 `sonic_release` + `robot_smoke` + `smpl_filtered` + `terrain_type=plane` 创建 `create_mjlab_manager_env(cfg, 'cpu')`，执行 reset 和零动作 step。
- 结果：通过。motion command 加载 210 个 robot motion，reset obs shape 为 `actor_obs (2, 930)`、`critic_obs (2, 1789)`、`tokenizer (2, 1767)`；step 返回 reward shape `(2,)`、done shape `(2,)`、info keys `['env_actions', 'episode', 'log', 'time_outs', 'to_log']`。

### M4 - Observation 与 3 encoder 对齐
- 命令：读取 `sonic_release` 的 `algo.config.actor.backbone.encoders`，与 mjlab tokenizer terms 对比。
- 结果：通过。G1、teleop、SMPL 所有输入项均存在，无 missing：
  `command_multi_future_nonflat`、`motion_anchor_ori_b_mf_nonflat`、`command_multi_future_lower_body`、`vr_3point_local_target`、`vr_3point_local_orn_target`、`motion_anchor_ori_b`、`smpl_joints_multi_future_local_nonflat`、`smpl_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_smpl`。

### M5 - Rewards / Terminations / Events / Curriculum 训练必要项
- reset/step smoke 中确认 event manager active terms：reset `reset_scene_to_default`；startup `add_joint_default_pos`、`base_com`、`randomize_rigid_body_mass`、`physics_material`；interval `push_robot`。
- reward terms active：`tracking_anchor_pos`、`tracking_anchor_ori`、`tracking_relative_body_pos`、`tracking_relative_body_ori`、`tracking_body_linvel`、`tracking_body_angvel`、`tracking_vr_5point_local`、`action_rate_l2`、`joint_limit`、`feet_acc`。
- termination terms active：`time_out`、`anchor_pos`、`anchor_ori_full`、`ee_body_pos`、`foot_pos_xyz`。
- 20 步 CPU rollout（4 env，零动作）：通过；reward finite，last obs shapes `actor_obs (4, 930)`、`critic_obs (4, 1789)`、`tokenizer (4, 1767)`，done_count `5`。

### M6 - TRL 小规模训练 smoke
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16 headless=True ++algo.config.num_learning_iterations=10 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：通过，完成 learning iteration 1 到 10。
- 关键输出：mjlab backend 使用 `cuda:0` 现有设备；初始化 G1、teleop、SMPL 三个 encoder；PPO loop 正常采样和更新；第 10 iteration total timesteps `3840`，mean reward `-16.74488`，日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_214120`。
- 未修改 CUDA、显卡驱动或系统 GPU 组件。

### M7 - SOMA / sonic_bones_seed 第二阶段兼容
- 查找配置：存在 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`，SOMA 数据 smoke 路径存在 `data/motion_lib_bones_seed/soma_uniform_smoke`。
- compose 检查：`sonic_bones_seed` + `robot_smoke` + `soma_uniform_smoke` + `smpl_filtered` 通过；tokenizer 自动包含 `soma_joints_multi_future_local_nonflat`、`soma_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_soma`，所有 encoder input 无 missing，action dim 29。
- CPU reset/step smoke：通过；reset obs shapes `actor_obs (2, 930)`、`critic_obs (2, 1789)`、`tokenizer (2, 2668)`；step reward finite。

### M8 - 文档、清理和交付状态
- 本轮未创建临时测试脚本，所有测试均为 inline `uv run python - <<'PY' ... PY`。
- 本轮未修改 Python 源码；只更新 workflow 文档/摘要。
- 训练 smoke 产生日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_214120`，该目录未出现在 `git status --short` 中，应为 ignored 或已按项目规则排除。
- 最终 `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 未验证项
- 未用默认完整数据路径 `data/motion_lib_bones_seed/robot_filtered`、`data/bones_seed_smpl`、`data/motion_lib_bones_seed/soma_filtered` 跑训练；本轮使用仓库 smoke 数据以降低成本。
- 未验证 rough/trimesh terrain 训练；smoke 使用 `manager_env.config.terrain_type=plane`。
- 未验证多 GPU、多进程、大规模 `num_envs=4096` 训练、checkpoint resume/finetune、ONNX export、render/eval、W&B 在线曲线。
- 未修改或重写 `UniversalTokenModule`、PPO trainer、checkpoint key。

## 2026-05-08T23:06:32+08:00 - PLAN.md 代码阶段补充执行：默认训练尝试与非 smoke 正常训练

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认用户已要求进入 PLAN.md 代码阶段；`PERSON.md` 没有实际 open 条目。
- `.workflow/artifacts/open_questions.json`：仅有非 blocking 的计划确认问题；本轮用户已确认继续执行。
- `README.md`、`docs/README.md`、`pyproject.toml`：再次确认根项目使用 uv + `.venv`，依赖包含 `mjlab==1.2.0`；未安装任何依赖。

### M0 / M1 复核
- `uv run python --version`：通过，输出 `Python 3.10.20`。
- `uv run python -c "import mjlab, torch, mujoco; ..."`：通过，输出 `ok mjlab unknown torch 2.11.0+cu130 mujoco 3.8.1`。
- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv`：GPU 为 `NVIDIA GeForce RTX 5070 Ti Laptop GPU`，总显存 `12227 MiB`，运行前空闲约 `9539 MiB`。
- 数据路径复核：默认完整目录 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl` 不存在；`data/motion_lib_bones_seed/robot_medium`、`data/motion_lib_bones_seed/soma_uniform_medium`、`data/smpl_filtered`、`/home/ykj/Downloads/dataset/bones-seed`、`/home/ykj/commonly_used/body_models` 存在。
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`：通过。

### M7 默认正常训练路径尝试
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab headless=True`
- 结果：进入训练入口并创建日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_230045`，但默认不覆盖 `terrain_type` 时卡在 mjlab rough terrain 生成阶段；运行约 3 分钟后手动 `SIGINT` 中断以避免无界等待。
- 证据：中断堆栈显示 `Terrain generation took 109.7336 seconds`，卡点位于 `mjlab/terrains/terrain_generator.py` 的 `TerrainGenerator.compile`；不是 CUDA/驱动修改问题，也没有进入 PPO iteration。
- 结论：默认完整训练路径在本机当前配置下未完成，主要限制为默认 rough terrain 生成耗时不可接受；此外默认完整数据目录 `robot_filtered` / `bones_seed_smpl` 也不存在，后续按 PLAN 使用最大可用 medium 数据和 plane terrain 验证正常训练代码路径。

### M7 非 smoke 正常训练
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=64 headless=True ++algo.config.num_learning_iterations=100 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_medium ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：通过，完成 100 个 learning iterations，真实 PPO 采样和更新正常执行。
- 训练事实：`num_envs=64`，加载 `data/motion_lib_bones_seed/robot_medium` 共 970 个 motion 文件，启动时采样 64 个 motions；使用 `cuda:0`，未修改 CUDA/显卡驱动/系统 GPU 组件。
- 日志与 checkpoint：`logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_230400`；保存 `last.pt`，大小约 `432M`；`config.yaml` 已生成。
- 第 100 iteration：total episodes `6400`，total timesteps `153600`，mean rewards `-16.70631`，mean length `8.99000`，computation `1270 steps/s`，total time `122.01s`，reward finite，未出现 NaN/OOM。

### 清理与状态
- 本轮没有创建临时测试脚本。
- 没有修改 Python 源码；只更新 workflow 文档和 `task.json` 摘要。
- `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。
- 检查训练进程：无遗留本轮训练进程。

### 最终未验证项
- 默认不覆盖参数的 rough terrain 正常训练未完成；本机运行中 terrain 生成 109 秒后仍在 `TerrainGenerator.compile`，已记录为当前默认配置耗时限制。
- 默认完整数据目录 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl` 不存在，因此未用官方完整过滤数据跑通默认训练。
- 未验证 `num_envs=4096`、多 GPU、多进程、checkpoint resume/finetune、ONNX export、render/eval、W&B 在线曲线。
- 未修改或重写 `UniversalTokenModule`、PPO trainer、checkpoint key。

## 2026-05-08T23:09:14+08:00 - 补充执行：SOMA / sonic_bones_seed 训练入口

### 命令
- `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_bones_seed use_mjlab=True sim_type=mjlab num_envs=32 headless=True ++algo.config.num_learning_iterations=20 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_medium ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_uniform_medium ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`

### 结果
- 通过，完成 20 个 learning iterations，真实 PPO 采样和更新正常执行。
- 训练事实：`num_envs=32`，加载 `data/motion_lib_bones_seed/robot_medium` 共 970 个 robot motion 文件，SOMA tokenizer terms 已激活，使用 `cuda:0`，未修改 CUDA/显卡驱动/系统 GPU 组件。
- 日志目录：`logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_bones_seed_test-20260508_230821`。
- 第 20 iteration：total episodes `640`，total timesteps `15360`，mean rewards `-20.23965`，mean length `10.17000`，computation `508 steps/s`，total time `30.84s`，reward finite，未出现 NaN/OOM。
- 该 20-iteration 短训练未生成 `last.pt`；目录中存在 `config.yaml` 和 `meta.yaml`，未达到当前保存 checkpoint 的间隔。
- 检查训练进程：无遗留 `gear_sonic/train_agent_trl.py` / `accelerate launch` 训练进程。

## 2026-05-11T12:05:55+08:00 - H20 服务器训练兼容化 grounding / plan / review 完成

### 已读取
- `AGENTS.md`：确认本项目 SonicMJ 的 mjlab 迁移使用 uv，不使用 conda；不得修改 CUDA、显卡驱动、Isaac Sim 底层安装或系统级 GPU 环境；外部 `mjlab`、`InstinctMJ`、原始仓库只读参考。
- `PERSON.md`：没有实际 open 条目。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复历史迁移验证上下文，并将本轮目标重定向为 H20 服务器训练环境兼容与 smoke test。
- `README.md`、`docs/README.md`、`pyproject.toml`、`uv.lock`：确认根项目使用 uv；当前锁文件包含 `torch 2.11.0`、CUDA 13 wheel、`mjlab==1.2.0`、`mujoco-warp>=3.8.0,<3.9`。
- `gear_sonic/train_agent_trl.py`、`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`、`sonic_mj/`：确认 mjlab backend 入口、3 encoder 训练配置和历史可训练代码路径。

### 已完成
- 更新 `TASK.md`：记录服务器事实 `nvcc 11.6`、Driver `570.124.06`、reported CUDA `12.8`、GPU `NVIDIA H20`；保留不修改 CUDA/驱动/系统 GPU 组件、只用项目 uv 环境的硬约束；补充服务器 smoke 验收条件。
- 写入 `.workflow/artifacts/grounding.md`：记录当前锁文件 CUDA 13 依赖与 H20 服务器 Driver 570/CUDA 12.8 的兼容风险，以及历史本机训练验证不能替代服务器验证。
- 写入 `.workflow/artifacts/paper_project_map.json`：保留 SONIC 论文、原实现、目标 `sonic_mj` 代码和服务器验证点之间的结构化映射。
- 写入 `.workflow/artifacts/open_questions.json`：当前没有 blocking 问题；记录等待确认 `PLAN.md`、是否允许服务器上重建项目 `.venv`、smoke 数据路径偏好的非阻塞问题。
- 生成 `PLAN.md`：M0-M8 覆盖服务器基线、当前 lock 兼容性判定、必要时项目级 CUDA 12.x 依赖修复、compile/compose、reset/step、obs/order、PPO smoke、默认数据加载检查和文档清理。
- 写入 Codex 多角色 review：`scope.md`、`milestones.md`、`architecture.md`、`testing.md`、`executor.md`。

### 判断
- 没有阻塞生成计划的问题；服务器信息足够制定执行计划。
- 计划阶段不应直接修改依赖或运行训练；按用户要求停在等待 `PLAN.md` 确认。
- 确认后应先在服务器执行只读 M0/M1；只有现有 CUDA 13 lock 在 Driver 570/CUDA 12.8 上失败时，才通过项目级 uv 依赖和 lock 文件切换到 CUDA 12.x 兼容组合。

### 本轮命令与结果
- `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh run --workspace "$PWD" --request ...`：workflow 子进程生成本轮 `TASK.md`、`PLAN.md` 和 artifacts；子进程完成后未正常退出，已终止该已完成的 workflow 子进程，未修改 workflow 源码。
- `sed -n ... TASK.md PLAN.md PROGRESS.md .workflow/artifacts/grounding.md`：核对本轮文档内容。
- `python -m json.tool .workflow/artifacts/open_questions.json`：通过，JSON 格式有效。
- `python -m json.tool .workflow/artifacts/paper_project_map.json`：通过，JSON 格式有效。
- `git status --short --branch`：当前分支 `SONICMJ...origin/SONICMJ`；已有 `M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后按 `PLAN.md` 从 M0 开始执行；如果用户允许，可在服务器项目目录内用 uv 重建 `.venv`，但不得修改 base、系统 CUDA、驱动或 GPU 底层组件。

## 2026-05-11T12:20:00+08:00 - 更新 PLAN：加入预备服务器 GPU 前置测试阶段

### 用户新增要求
- 在 `PLAN.md` 中加入一台可修改的预备服务器，用于完成真正使用 GPU 前的测试。
- 预备服务器登录入口为 `ssh -p 37716 root@region-42.seetacloud.com`；认证信息由用户提供，不写入项目文档。
- 预备服务器可以随便修改；最好把它配置成最终 H20 训练服务器将使用的项目依赖配置。
- 最终 H20 服务器仍为 Driver `570.124.06`、reported CUDA `12.8`、`nvcc 11.6`、GPU `NVIDIA H20`；等用户开启 GPU 后再运行需要 GPU 的 reset/step 与训练 smoke。

### 已完成
- 更新 `PLAN.md` 的状态说明和“执行环境分层”章节。
- 将 M0-M3 调整为可先在预备服务器执行的 GPU 前置测试：环境基线、依赖解析、uv lock/env 修复、import、Hydra compose、静态检查。
- 将 M4-M7 明确标记为最终 H20 GPU 开启后执行：mjlab GPU reset/step、obs tensor 检查、PPO 训练 smoke、默认数据真实 env reset。
- 更新计划确认项：预备服务器先行；最终 H20 GPU 开启后再继续 GPU 阶段。

### 当前状态
- 尚未登录预备服务器，也未修改远端环境。
- 尚未修改 `pyproject.toml` / `uv.lock`。
- 下一步仍需用户确认按更新后的 `PLAN.md` 执行。

## 2026-05-11T14:39:24+08:00 - 预备服务器 GPU 前置测试完成

### 预备服务器基线
- 登录入口：`ssh -p 37716 root@region-42.seetacloud.com`；认证信息未写入项目文件。
- 系统：Ubuntu 20.04.4，host `autodl-container-e9b742b627-c588c445`。
- GPU 状态：`nvidia-smi` 返回 `No devices were found`，本阶段按 GPU 前置测试处理，不作为失败。
- CUDA 工具链：`nvcc --version` 为 CUDA `11.3, V11.3.109`；未修改 CUDA、显卡驱动或系统 GPU 组件。
- Python/uv：远端用户级安装 `uv 0.11.13`；用 uv 安装 CPython `3.10.20`。
- 磁盘：`/root/autodl-tmp` 50G；项目部署到 `/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`。

### 依赖修复
- 原因：根 `uv.lock` 原先解析到 `torch 2.11.0` 和 CUDA 13 wheel；最终 H20 服务器 Driver `570.124.06` / reported CUDA `12.8` 不适合作为 CUDA 13 wheel 的目标运行环境。
- 修改 `pyproject.toml`：在根项目依赖中显式加入 `torch==2.7.1`。
- 重新运行 `uv lock`：CUDA 13 相关包被移除，锁定到 `torch 2.7.1` / `torchvision 0.22.1` / `triton 3.3.1` 和 NVIDIA CUDA 12.6 wheel 组合。
- 兼容判断：最终 H20 的 Driver 570 可运行 CUDA 12.6 wheel；`nvcc 11.6` 只是本机编译工具链版本，本轮不依赖系统 nvcc 编译 PyTorch。

### 执行过的命令与结果
- 远端只读基线：`nvcc --version`、`nvidia-smi`、`python3 --version`、`df -h`、`which git rsync uv`。
  - 结果：无 GPU；系统 Python 3.8.10；远端原先无 uv；`git`/`rsync` 可用。
- 远端安装 uv 与 Python：
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - `uv python install 3.10`
  - 结果：`uv 0.11.13`、CPython `3.10.20` 可用。
- 项目同步：
  - `rsync -az --delete` 同步代码、配置和小资产到远端；排除 `.git/`、`.venv*/`、`logs_rl/`、`data/`、`sonic_release/` 等大目录。
  - 结果：远端项目目录创建成功；大数据未同步，符合 GPU 前置测试范围。
- 依赖解析：
  - 初始测试 `torch==2.7.1` + PyTorch cu128 可解析，但官方 cu128 源下载在预备服务器和本机都过慢。
  - 改用 PyPI `torch==2.7.1` 的 CUDA 12.6 wheel 组合后，本机 `uv sync --frozen` 成功。
  - 远端直连 wheel 下载仍慢；为完成前置验证，将本机已完成的 `.venv/lib/python3.10/site-packages` 同步到远端，并保留远端自己的 `.venv/bin/python` / `pyvenv.cfg`，随后修正 editable path 到远端项目路径。
- 本机验证：
  - `uv run python - <<'PY' ... import torch/mujoco/mujoco_warp/mjlab ... PY`
  - `uv run python -m compileall -q sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：`torch 2.7.1+cu126`，`torch.version.cuda 12.6`，`torch.cuda.is_available() False`；`mujoco 3.8.1`、`mujoco_warp 3.8.0`、`mjlab` 导入成功；compileall 通过。
- 预备服务器验证：
  - `.venv/bin/python - <<'PY' ... import torch/mujoco/mujoco_warp/mjlab ... PY`
  - `.venv/bin/python -m compileall -q sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：`python 3.10.20`，`torch 2.7.1+cu126`，`torch.version.cuda 12.6`，`torch.cuda.is_available() False`；`mujoco 3.8.1`、`mujoco_warp 3.8.0`、`mjlab` 导入成功；compileall 通过。
- 预备服务器 Hydra / env cfg 检查：
  - compose `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab checkpoint=null num_envs=2 headless=True manager_env.config.terrain_type=plane`。
  - 调用 `make_sonic_mj_env_cfg(cfg)`。
  - 结果：`num_envs 2`；actions `['joint_pos']`；observations `['policy', 'critic', 'tokenizer']`；commands `['motion']`；rewards `10`；terminations `['time_out', 'anchor_pos', 'anchor_ori_full', 'ee_body_pos', 'foot_pos_xyz']`；terrain `plane`。

### 当前状态与未验证项
- M0-M3 预备服务器 GPU 前置测试完成。
- 没有创建临时测试脚本。
- 没有运行需要 GPU 的 env reset/step 或 PPO 训练；预备服务器没有可见 GPU。
- 未同步完整数据目录 `data/` 和 `sonic_release/`，因此未做默认 motion 数据加载或 checkpoint 相关检查。
- 最终 H20 GPU 开启后，需要继续 M4-M7：`torch.cuda` 初始化、mjlab GPU reset/step、obs/order 诊断、`num_envs=16` / `num_learning_iterations=10` PPO smoke。
- 检查遗留进程：无 `gear_sonic/train_agent_trl.py`、`accelerate launch`、`uv sync` 或 `rsync` 遗留进程。
- `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，`M pyproject.toml`，`M uv.lock`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

## 2026-05-11T15:03:52+08:00 - AutoDL H20 GPU 阶段测试完成

### 服务器与数据放置
- 服务器入口：`ssh -p 37716 root@region-42.seetacloud.com`；认证信息未写入项目文件。
- 项目目录：`/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`，数据和项目均放在 AutoDL 数据盘 `/root/autodl-tmp` 下，未放到系统盘 `/root`。
- GPU：2 张 `NVIDIA H20`，每张约 `97871 MiB`；`nvidia-smi` 显示 Driver `580.105.08`、reported CUDA `13.0`。
- `nvcc`：该 AutoDL 服务器上未找到 `nvcc`；本轮未安装或修改 CUDA、显卡驱动、系统级 GPU 组件。
- 项目环境：远端 `.venv/bin/python` 为 Python `3.10.20`；`torch 2.7.1+cu126`、`torch.version.cuda 12.6`、`torch.cuda.is_available() True`、`torch.cuda.device_count() 2`。

### 已同步数据
- 同步 `gear_sonic/data/assets/` 到远端，修复 G1 MuJoCo XML 缺失问题。
- 同步 `data/motion_lib_bones_seed/robot_smoke` 和 `data/motion_lib_bones_seed/robot_medium`。
- 从本地按 `robot_smoke/robot_medium` 引用生成临时 SMPL 文件列表并同步匹配文件到远端 `data/smpl_filtered/`；匹配 `874` 个文件，远端大小约 `949M`。临时文件 `/tmp/sonicmj_smpl_subset.txt` 已删除。
- 同步默认 robot 数据 `data/motion_lib_bones_seed/robot_filtered` 到远端，远端大小约 `7.9G`，训练时加载 `129785` 个 robot motion 文件。
- 本地和远端 `data/bones_seed_smpl` 均为空或缺失，因此完整默认 SMPL 数据语义尚未验证。

### GPU reset/step 与顺序诊断
- 命令要点：
  - `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab checkpoint=null num_envs=2 headless=True`
  - `++algo.config.num_learning_iterations=1 ++algo.config.num_steps_per_env=2`
  - `++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke`
  - `++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy`
  - `manager_env.config.terrain_type=plane`
  - `create_mjlab_manager_env(cfg, "cuda:0")`
- 结果：
  - 加载 `210` 个 `robot_smoke` motions。
  - `actor_obs (2, 930)`，`critic_obs (2, 1789)`，`actions_dim 29`。
  - joint/body/motion/action/policy order 诊断全部通过；motion dof mapping 为 identity；action dim 为 29。
  - 单步 step 成功，reward/done shape 为 `(2,)`，reward finite，mean reward `-0.47829321026802063`。

### 单卡 PPO smoke
- 首次命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m accelerate.commands.launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=2 headless=True ++algo.config.num_learning_iterations=1 ++algo.config.num_steps_per_env=2 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：
  - 训练进程可启动并完成 1 iteration，但出现配置错误：`batch_size must be a multiple of num_mini_batches, inexact division: 2 / 4 = 0.5`。
  - 判定为 smoke 参数太小导致的 batch 配置问题，不是 GPU/环境不可用。
- 修正命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m accelerate.commands.launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 修正结果：
  - 退出码 `0`，完成 learning iteration `1` 和 `2`。
  - 初始化 `g1` / `teleop` / `smpl` encoders 和 `g1_dyn` / `g1_kin` decoders。
  - total episodes `8`，total timesteps `32`，未出现 OOM/NaN。

### 双卡 PPO smoke
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：
  - 退出码 `0`，Accelerate 进入 multi-GPU 模式。
  - rank 0 使用 `cuda:0`，rank 1 使用 `cuda:1`；两个 rank 分别完成 env 创建、采样和训练。
  - 完成 learning iteration `1` 和 `2`；total episodes `16`，total timesteps `64`，mean rewards `-6.04001`。
  - 未出现 OOM/NaN。
  - 退出时有 PyTorch distributed 的非致命清理警告：`destroy_process_group() was not called before program exit`。

### 默认 robot 数据双卡训练
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 manager_env.config.terrain_type=plane`
- 说明：
  - 未覆盖 `motion_file`，因此使用 `sonic_release.yaml` 默认 robot motion 路径 `data/motion_lib_bones_seed/robot_filtered`。
  - 未覆盖 `smpl_motion_file`，因此尝试使用默认 `data/bones_seed_smpl`；该目录缺失/为空，当前 motion lib 对缺失 SMPL 路径会容忍并使用 `None` 数据项。
- 结果：
  - 退出码 `0`，两个 rank 均加载 `129785` 个 robot motion 文件。
  - 完成 learning iteration `1` 和 `2`；total episodes `16`，total timesteps `64`。
  - mean reward：iteration 1 为 `-2.69212`，iteration 2 为 `-12.68186`。
  - 未出现 OOM/NaN。
  - 同样存在非致命 `destroy_process_group()` 清理警告。

### 清理与当前状态
- 远端检查命令：
  - `pgrep -af "gear_sonic/train_agent_trl.py|accelerate.commands.launch|accelerate launch|rsync|uv sync" || true; nvidia-smi`
- 结果：
  - 无遗留训练、rsync 或 uv sync 进程；`pgrep` 只匹配到本次检查命令自身。
  - `nvidia-smi` 显示两张 H20 显存占用均为 `0MiB`，无运行中的 GPU 进程。

### 结论与未验证项
- 已通过：服务器 GPU import、mjlab reset/step、order diagnostics、单卡 PPO smoke、双卡 PPO smoke、默认 `robot_filtered` 数据的双卡短训练。
- 已证明：当前项目依赖组合 `torch 2.7.1+cu126` 可在该 AutoDL H20 GPU 服务器上运行，不需要修改 CUDA/驱动。
- 未完整验证：`sonic_release.yaml` 的完整默认 SMPL 数据路径 `data/bones_seed_smpl`，因为本地和远端该目录为空或不存在。
- 未验证：默认大规模 `num_envs=4096`、长时间训练、checkpoint resume/finetune、rough/trimesh terrain、render/eval、ONNX export、W&B 在线记录。

## 2026-05-11T15:32:27+08:00 - 补齐 AutoDL `data/bones_seed_smpl` 并完成默认数据双卡训练

### 数据补齐
- 本机检查：
  - `/home/ykj/Downloads/dataset/bones-seed` 总大小约 `650G`，主要是官方 Bones-SEED 原始/衍生数据，不直接整目录同步。
  - 本机 `data/smpl_filtered` 为可用于默认 SMPL motion 的 `.pkl` 目录，大小约 `31G`，文件数 `131455`。
  - 本机 `data/motion_lib_bones_seed/robot_filtered` 的 motion stem 数为 `129785`；`data/smpl_filtered` 的非 metadata stem 数为 `131454`。
  - 对齐检查显示 `robot_filtered` 有 `7` 个 motion stem 在本机 `data/smpl_filtered` 中不存在：`kneeling_loop_002__A098_M`、`neutral_button press_001__A543`、`neutral_button press_001__A543_M`、`neutral_button press_001__A544`、`neutral_button press_001__A544_M`、`neutral_button press_001__A545`、`neutral_button press_001__A545_M`。
  - 在 `/home/ykj/Downloads/dataset/bones-seed` 中也未找到这 7 个同名 `.pkl`，因此不是同步遗漏；当前 `MotionLibBase` 对缺失 SMPL 文件会将对应条目置为 `None`。
- 远端整理：
  - 远端项目仍位于 `/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`。
  - 将此前由本轮同步的远端小样本 `data/smpl_filtered` 迁移为 `data/bones_seed_smpl`，再把本机完整 `data/smpl_filtered/` 增量同步到远端默认路径。
  - 为兼容此前 smoke 命令，远端保留 `data/smpl_filtered -> bones_seed_smpl` 符号链接。
  - 同步命令：`rsync -az --info=progress2 --partial --inplace -e 'ssh -o UserKnownHostsFile=/tmp/sonicmj_known_hosts -o StrictHostKeyChecking=no -p 37716' data/smpl_filtered/ root@region-42.seetacloud.com:/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl/data/bones_seed_smpl/`
- 远端同步结果：
  - `data/bones_seed_smpl` 大小约 `31G`，`.pkl` 文件数 `131455`。
  - `data/smpl_filtered` 是指向 `bones_seed_smpl` 的符号链接。
  - `/root/autodl-tmp` 数据盘剩余约 `2.0G`，使用率 `97%`。该状态能完成小规模 smoke，但正式训练前建议清理不需要的数据或换更大数据盘。

### 完整默认路径双卡训练
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16 headless=True ++algo.config.num_learning_iterations=10 manager_env.config.terrain_type=plane`
- 关键点：
  - 未覆盖 `motion_file`，使用 `sonic_release.yaml` 默认 `data/motion_lib_bones_seed/robot_filtered`。
  - 未覆盖 `smpl_motion_file`，使用 `sonic_release.yaml` 默认 `data/bones_seed_smpl`。
  - 双卡多进程：rank 0 使用 `cuda:0`，rank 1 使用 `cuda:1`。
  - 每个 rank 创建 `16` 个 mjlab env。
- 训练结果：
  - 退出码 `0`，完成 learning iteration `1` 到 `10`。
  - 两个 rank 均加载 `129785` 个 robot motion 文件，并从默认完整 SMPL 路径读取对应 SMPL 数据。
  - 初始化 `g1`、`teleop`、`smpl` 三个 encoder；tokenizer 观测中包含 `smpl_joints_multi_future_local_nonflat (10, 72)`、`smpl_root_ori_b_multi_future (10, 6)`、`joint_pos_multi_future_wrist_for_smpl (10, 6)`。
  - policy obs shape 为 `(930,)`，critic obs shape 为 `(1789,)`，action dim 为 `29`。
  - 第 10 iteration：computation `434 steps/s`，total episodes `320`，total timesteps `7680`，mean rewards `-24.92989`，mean length `10.49500`。
  - 未出现 OOM、NaN 或 CUDA 初始化错误。
  - 退出时仍有 PyTorch distributed 非致命清理警告：`destroy_process_group() was not called before program exit`。
- 日志目录：
  - `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260511_152916`

### 清理与状态
- 远端检查命令：
  - `pgrep -af "gear_sonic/train_agent_trl.py|accelerate.commands.launch|accelerate launch|rsync|uv sync" || true`
  - `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
  - `df -h /root/autodl-tmp`
- 结果：
  - 无遗留训练、rsync 或 uv sync 进程；`pgrep` 只匹配到本次检查命令自身。
  - 两张 H20 显存占用均为 `0 MiB`，GPU 利用率 `0%`。
  - 数据盘 `/root/autodl-tmp` 剩余约 `2.0G`。

### 当前结论
- 完整默认路径的小规模多卡训练已通过：默认 robot 数据 + 默认 `data/bones_seed_smpl` + 双 H20 + mjlab backend + Universal Token 三 encoder。
- 仍未验证：正式规模 `num_envs=4096`、长时间训练、checkpoint resume/finetune、rough/trimesh terrain、render/eval、ONNX export、W&B 在线记录。
- 数据盘空间已接近满载，正式训练前不建议继续往该 AutoDL 数据盘放大数据或大 checkpoint。

## 2026-05-11T16:02:06+08:00 - AutoDL H20 每卡 env 容量测试

### 测试设置
- 目标：估计当前双 H20 AutoDL 服务器上默认真实数据训练时每张卡可开的 env 数。
- 解释：当前 `accelerate --num_processes=2` 下，`num_envs=N` 会让每个 rank / 每张卡各自创建 `N` 个 env，因此这里记录的是每卡 env 数。
- 命令共同设置：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1`
  - `.venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py`
  - `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab headless=True`
  - `++algo.config.num_learning_iterations=1`
  - `manager_env.config.terrain_type=plane`
- 数据路径：未覆盖 `motion_file` / `smpl_motion_file`，使用默认真实数据 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl`。

### 结果
- `num_envs=4096`（每卡 4096，双卡总 8192）：
  - 失败，退出码 `1`。
  - 两个 rank 均创建了 4096 env 并加载 `129785` 个 motion 文件。
  - 失败点：Warp CUDA graph 创建时 OOM。
  - 关键错误：`Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, /builds/omniverse/warp/warp/native/warp.cu:2899)`。
- `num_envs=2048`（每卡 2048，双卡总 4096）：
  - 通过，退出码 `0`。
  - 完成 learning iteration `1`。
  - computation `12761 steps/s`，collection `6.250s`。
  - total episodes `4096`，total timesteps `98304`，mean rewards `-22.35138`。
- `num_envs=3072`（每卡 3072，双卡总 6144）：
  - 通过，退出码 `0`。
  - 完成 learning iteration `1`。
  - computation `11792 steps/s`，collection `10.570s`。
  - total episodes `6144`，total timesteps `147456`，mean rewards `-18.76445`。
- `num_envs=3584`：
  - 用户要求停止继续测试，因此手动 SIGTERM 中断；不作为容量结论。

### 当前建议
- 保守建议按用户要求先使用 `num_envs=2048`，即每卡 2048、双卡总 4096。
- 已知 `num_envs=3072` 也能完成 1 iteration，但正式训练建议先用 2048 留显存和磁盘余量。
- `num_envs=4096` 每卡不可用，会在 Warp CUDA graph 创建阶段 OOM。
- 停止后检查：无遗留训练进程，双 H20 显存占用为 `0 MiB`。
