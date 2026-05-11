# TASK: SonicMJ 服务器训练兼容化

## 项目元信息
- 项目名：GR00T-WholeBodyControl / SonicMJ
- 目标代码库：`/home/ykj/project/SONICMJ/GR00T-WholeBodyControl`
- python_env_kind：`uv`
- python_env：`.venv`
- python_setup_mode：`existing`，如需重建服务器环境必须使用 uv 和项目隔离环境
- 当前阶段：完成 grounding、计划和 Codex 多角色 review；等待用户确认 `PLAN.md`

## 本轮用户请求
- [requirement] 把当前 SonicMJ 项目变为服务器可以训练的版本，并能在服务器上完成训练 smoke test。
- [requirement] 服务器信息：`nvcc 11.6`；`nvidia-smi Driver Version 570.124.06`；`CUDA Version 12.8`；GPU 为 `NVIDIA H20`。
- [requirement] 不修改 CUDA、显卡驱动、系统级 GPU 组件或服务器底层环境。
- [requirement] SonicMJ 的 mjlab 迁移使用 uv 管理环境，不使用 conda 或 base 环境安装依赖。
- [requirement] 当前阶段只完成 grounding、计划和 Codex 多角色 review，然后停在等待用户确认 `PLAN.md`。

## 硬约束
- [constraint] 回答和项目文档使用中文。
- [constraint] 改动范围限制在当前项目；`/home/ykj/project/SONICMJ/mjlab`、`InstinctMJ`、原始外部 `GR00T-WholeBodyControl` 只读参考，必须修改外部仓库时先询问用户。
- [constraint] 不修改 workflow 源码仓库 `/home/ykj/tool/WORKFLOW`。
- [constraint] 不修改 CUDA、显卡驱动、Isaac Sim 底层安装或系统级 GPU 环境。
- [constraint] 依赖只能安装在项目 `.venv` 或项目隔离 uv 环境中，不得安装到 base。
- [constraint] 不修改 `UniversalTokenModule`、PPO trainer、checkpoint key 或网络结构，除非先获得用户确认。
- [constraint] 不破坏 IsaacLab 原路径；未使用 `use_mjlab=True` / `sim_type=mjlab` 时原入口保持原行为。
- [constraint] 临时测试脚本如需创建，测试完成后必须删除；优先使用 inline 命令。
- [constraint] 所有结论、命令结果、阻塞和下一步必须写回 `PROGRESS.md` 或 `.workflow/artifacts`。

## 服务器兼容目标
- [requirement] 明确当前 `uv.lock` / `.venv` 依赖与服务器驱动能力的兼容风险，特别是当前本地锁定的 `torch 2.11.0+cu130`、`cuda-toolkit 13.0.x`、`nvidia-*cu13`、`mujoco-warp>=3.8.0,<3.9` 与服务器 `CUDA runtime 12.8` / `nvcc 11.6` 的关系。
- [requirement] 服务器环境修复只能通过项目级 uv 依赖、锁文件或环境文档实现，不通过修改系统 CUDA/驱动解决。
- [requirement] 服务器 smoke test 至少覆盖：uv 环境导入、GPU 可见性、mjlab/MuJoCo 初始化、Hydra compose、SonicMJ env cfg 构造、reset、单步 step、order diagnostics、短 PPO 训练。
- [requirement] 如 cu13 wheel 在 H20 + Driver 570.124.06 上不可用，计划应切换到可由 Driver 570 支持的项目级 CUDA 12.x PyTorch / 依赖组合，并重新锁定 uv 环境。
- [requirement] 保留 SONIC 训练语义：motion_lib PKL 格式、future reference frames、encoder sampling、adaptive sampling、Universal Token policy 输入输出约定。
- [requirement] 默认 G1 29-DOF action dim 保持 29，canonical order 使用 SONIC MuJoCo order。

## 当前代码事实
- `pyproject.toml`：根项目 `sonicmj`，Python `>=3.10,<3.11`，依赖包含 `mjlab==1.2.0`、`mujoco-warp>=3.8.0,<3.9`、`accelerate`、`trl==0.28.0`，uv sources 指向 `mujoco` index 与 PyPI。
- `uv.lock`：当前锁定 `torch 2.11.0`，并包含 `cuda-toolkit 13.0.2`、`nvidia-cuda-runtime 13.0.96`、`nvidia-cudnn-cu13`、`nvidia-nccl-cu13` 等 CUDA 13 wheel。
- 本地 `.venv` 当前事实：`Python 3.10.20`、`torch 2.11.0+cu130`、`torch.version.cuda=13.0`、`mujoco 3.8.1`、`mjlab 1.2.0`、`torch.cuda.is_available=True`。
- `gear_sonic/train_agent_trl.py`：`use_mjlab=True` 或 `sim_type=mjlab` 时进入 `sonic_mj.train.create_mjlab_manager_env`。
- `sonic_mj/` 已存在 mjlab backend，历史记录显示本机曾完成 compile、reset/step、SOMA smoke、`sonic_release` 100 iteration 中等规模 PPO 训练。
- 数据目录当前事实：
  - 存在 `data/motion_lib_bones_seed/robot_filtered`、`robot_medium`、`robot_smoke`。
  - 存在 `data/motion_lib_bones_seed/soma_uniform_medium`、`soma_uniform_smoke`。
  - 存在 `data/smpl_filtered` 和 `data/bones_seed_smpl`。
  - 当前未发现 `data/motion_lib_bones_seed/soma_filtered`。

## 参考资料
- 原训练入口：`gear_sonic/train_agent_trl.py`。
- 默认 3 encoder 实验：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`。
- SOMA 实验：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`。
- mjlab backend：`sonic_mj/train.py`、`sonic_mj/env_cfg.py`、`sonic_mj/wrapper.py`、`sonic_mj/mdp/`。
- G1 资产和顺序：`sonic_mj/assets.py`、`gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`。
- 历史验证记录：`PROGRESS.md` 中 2026-05-08 的 compile、reset/step、PPO smoke、100 iteration 训练和 SOMA 短训练结果。
- 论文：Zotero item `GT7EYT6G`，SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control，arXiv:2511.07820。

## 验收条件
- [acceptance] 在不修改服务器 CUDA/驱动的前提下，项目 uv 环境可在 H20 服务器上创建或复用。
- [acceptance] 记录 `uv run python --version`、`torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`、`torch.cuda.get_device_name(0)`、`mujoco.__version__`、`mjlab` 版本。
- [acceptance] 若当前 cu13 锁文件不能在 Driver 570.124.06 + CUDA runtime 12.8 上运行，必须把依赖改为项目级 CUDA 12.x 兼容组合并更新 `pyproject.toml` / `uv.lock`，不得要求修改驱动。
- [acceptance] `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py` 通过。
- [acceptance] `sonic_release` + `use_mjlab=True sim_type=mjlab` 能构造 mjlab env cfg，并进入 mjlab backend。
- [acceptance] 环境可 reset，单步 step 不报错；短 rollout 中 rewards/dones/obs 均无 NaN、shape/device 错误。
- [acceptance] action dim 为 29；robot joint/body order、motion body order、action joint order、policy `joint_pos`/`joint_vel`/`actions` order 均有打印和检查。
- [acceptance] `policy`、`critic`、`tokenizer` obs key、shape 和 `sonic_release`/Universal Token 训练预期一致。
- [acceptance] 在服务器 H20 上完成一个训练 smoke test：`num_envs=16`、`headless=True`、`num_learning_iterations=10`，使用 mjlab backend 和可用数据路径，reward finite，无 NaN/OOM。
- [acceptance] 记录服务器 smoke test 命令、日志目录、GPU/依赖版本、结果和未验证项到 `PROGRESS.md`。
