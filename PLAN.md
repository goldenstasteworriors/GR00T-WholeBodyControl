# PLAN: SonicMJ H20 服务器训练 smoke 兼容化

状态：预备服务器 GPU 前置测试已完成；当前等待用户开启最终 H20 GPU 后，再进入 H20 GPU reset/step 与训练 smoke。

## 执行环境分层
- 预备服务器：可用于真正使用 GPU 前的测试和环境修复，登录入口为 `ssh -p 37716 root@region-42.seetacloud.com`；认证信息由用户本轮消息提供，不写入项目文件。
- 预备服务器权限：用户明确允许在该服务器上随便修改；仍优先把修改限制在项目目录、uv 环境和用户级工具内，不主动修改系统 CUDA/显卡驱动，除非用户后续明确要求。
- 预备服务器目标：尽量配置成最终 H20 训练服务器可复用的项目环境，重点对齐 Python 3.10、uv、Driver 570 / CUDA 12.8 可运行的 PyTorch / mjlab / mujoco-warp 依赖组合。
- 最终 H20 训练服务器：`nvcc 11.6`，`nvidia-smi Driver Version 570.124.06`，reported `CUDA Version 12.8`，GPU 为 `NVIDIA H20`；最终 GPU 测试等待用户确认 GPU 已开启后执行。
- 阶段边界：在用户明确“GPU 已开启”前，不要求完成 `torch.cuda` 初始化、mjlab GPU env reset/step 或 PPO 训练；先完成 clone/sync、uv lock/env、import、Hydra compose、静态检查和 CPU/无 GPU 可执行的配置验证。

## M0 - 恢复上下文与服务器基线
- 目标：确认当前执行机器是预备服务器还是最终 H20 服务器，并记录环境信息。
- 完成标准：
  - 记录 `git status --short --branch`，不回退非本轮改动。
  - 记录 `uv --version`、`uv run python --version`。
  - 在最终 H20 服务器上记录 `nvidia-smi` 中 Driver `570.124.06`、CUDA `12.8`、GPU `NVIDIA H20`。
  - 在最终 H20 服务器上记录 `nvcc --version` 为 `11.6`，仅作为事实，不尝试修改。
  - 在预备服务器上记录实际 `nvidia-smi` / `nvcc --version`；如果没有 GPU 或 GPU 未开启，记录为 GPU 前置阶段，不作为失败。
- 命令：
  - `git status --short --branch`
  - `uv --version`
  - `uv run python --version`
  - `nvidia-smi`
  - `nvcc --version`
- Review 点：最终 H20 服务器所有命令只读；预备服务器允许修改，但优先不碰 CUDA、驱动、系统 GPU 组件；不得安装到 base。

## M1 - 当前 uv 锁文件兼容性判定
- 目标：先在预备服务器完成 `.venv` / `uv.lock` 的非 GPU 导入与依赖解析，再在最终 H20 GPU 开启后验证 CUDA 初始化。
- 完成标准：
  - 预备服务器记录 `torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`；无 GPU 时不要求 `torch.cuda.get_device_name(0)`。
  - 最终 H20 服务器 GPU 开启后记录 `torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`、`torch.cuda.get_device_name(0)`。
  - 记录 `mujoco.__version__`、`mjlab` 版本、`mujoco_warp` 是否可导入。
  - 若 `torch 2.11.0+cu130` 或 CUDA 13 wheel 因 Driver 570 不兼容失败，分类为项目依赖问题，进入 M2。
  - 若导入和 GPU 初始化成功，记录当前 lock 在服务器可用，跳过 M2 直接 M3。
- 命令：
  - `uv run python - <<'PY' ... import torch/mujoco/mjlab/mujoco_warp and print versions/device ... PY`
  - `uv pip list | rg 'torch|cuda|nvidia|mujoco|mjlab|warp'`
- Review 点：不要通过 `LD_LIBRARY_PATH` 指向系统 CUDA 做隐式修补；先让项目 uv 环境自洽。

## M2 - 项目级 CUDA 12.x 依赖修复
- 目标：如果 M1 失败，先在预备服务器把项目依赖切到 Driver 570 / CUDA 12.8 可支持的 Python wheel 组合，再把同一组 `pyproject.toml` / `uv.lock` 用于最终 H20 服务器。
- 完成标准：
  - 在 `pyproject.toml` 中明确服务器兼容的 `torch` / `torchvision` / CUDA 12.x wheel 来源或约束。
  - 用 uv 更新 `uv.lock`，依赖安装只发生在项目 `.venv` 或新建项目 venv。
  - 最终 H20 服务器不改系统 CUDA、驱动、base 环境。
  - 预备服务器可以调整用户级工具、项目目录和项目 `.venv`；如确需系统级改动，先记录原因并避免影响最终服务器假设。
  - 重新运行 M1 的非 GPU 导入检查并通过；最终 H20 GPU 开启后补跑 GPU 初始化。
- 建议命令：
  - `uv lock`
  - `uv sync --frozen` 或按实际修复路径使用 `uv sync`
  - `uv run python - <<'PY' ... torch.cuda smoke ... PY`
- Review 点：优先选 CUDA 12.x PyTorch 官方 wheel；若 `mujoco-warp` 对 `torch` 版本有约束，必须以 `mjlab==1.2.0` 可运行为准。任何改动都记录到 `PROGRESS.md`。

## M3 - 静态健康检查
- 目标：在预备服务器确认依赖修复后项目代码仍可导入和编译；该阶段不依赖真实 GPU。
- 完成标准：
  - `compileall` 通过。
  - Hydra compose `sonic_release` + `use_mjlab=True sim_type=mjlab` 成功。
  - 构造 `ManagerBasedRlEnvCfg` 成功，actions/commands/observations 包含预期项。
- 命令：
  - `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`
  - `uv run python - <<'PY' ... hydra compose sonic_release and make_sonic_mj_env_cfg ... PY`
- Review 点：不修改 trainer、UniversalTokenModule、checkpoint key 或网络结构。

## M4 - H20 上 reset/step 与顺序诊断
- 目标：用户开启最终 H20 GPU 后，证明 mjlab backend 能在服务器 GPU 上创建环境并完成最小仿真闭环。
- 完成标准：
  - 使用 `robot_smoke` + `smpl_filtered` + `terrain_type=plane` 创建小规模 env。
  - reset 成功，单步零动作/随机动作成功。
  - reward、done、obs tensor finite，无 device/shape 错误。
  - `env.print_order_diagnostics()` 或等价诊断输出 action dim 29，joint/body/action/policy order 检查通过。
- 命令：
  - `uv run python - <<'PY' ... create_mjlab_manager_env(config,'cuda:0'), reset, step, diagnostics ... PY`
- Review 点：SONIC 默认 MuJoCo order 不变；`JointPositionAction` 语义不变。

## M5 - Observation / encoder 对齐
- 目标：先在预备服务器做配置级和可导入级 obs/encoder 对齐检查；最终 H20 GPU 开启后补跑真实 env obs tensor 检查。
- 完成标准：
  - `policy`、`critic`、`tokenizer` obs shape 与历史记录同类配置一致。
  - `sonic_release` 的 G1、teleop、SMPL encoder inputs 全部存在。
  - 不出现静默 padding 或 key rename。
- 命令：
  - `uv run python - <<'PY' ... print obs dict keys/shapes and compare encoder inputs ... PY`
- Review 点：obs key/name/order 稳定，不破坏 checkpoint 可加载性。

## M6 - 服务器训练 smoke
- 目标：用户开启最终 H20 GPU 后，在 H20 上完成最小真实 PPO 训练闭环。
- 完成标准：
  - `sonic_release` + mjlab backend 完成 `num_envs=16`、`num_learning_iterations=10`。
  - `WANDB_MODE=disabled`，避免联网依赖影响 smoke。
  - reward finite，无 NaN/OOM。
  - 记录日志目录、iteration 进度、total timesteps、GPU 版本信息。
- 命令：
  - `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16 headless=True ++algo.config.num_learning_iterations=10 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- Review 点：这是服务器可训练 smoke 的核心验收；失败必须分类为依赖、数据、代码或资源问题。

## M7 - 默认数据路径加载检查
- 目标：先在预备服务器确认默认 3 encoder 数据路径、LFS/rsync/权限和配置可读；最终 H20 GPU 开启后补做真实 env reset。
- 完成标准：
  - `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl` 存在并可被 motion command 加载，或明确记录缺失/权限问题。
  - 不要求默认 `num_envs=4096` 完整训练作为本轮验收。
- 命令：
  - `uv run python - <<'PY' ... compose sonic_release with default motion paths, create env with small num_envs and reset ... PY`
- Review 点：如果默认 rough/trimesh terrain 过慢，允许用 `terrain_type=plane` 做数据加载检查，并记录未验证 rough terrain。

## M8 - 文档、清理与交付
- 目标：把所有服务器结论写回项目文档，保证可恢复。
- 完成标准：
  - `PROGRESS.md` 记录每条命令、结果、失败原因、日志目录和未验证项。
  - `.workflow/artifacts/open_questions.json` 更新剩余非阻塞问题。
  - 如修改 `pyproject.toml` / `uv.lock`，最终列出修改文件和理由。
  - 临时测试脚本已删除；训练进程无遗留。
  - `git status --short --branch` 已记录。
- 命令：
  - `git status --short --branch`
  - `pgrep -af 'gear_sonic/train_agent_trl.py|accelerate launch' || true`
- Review 点：不提交、不推送，除非用户明确要求。

## 当前 GPU 阶段结果
- M0-M3 已完成：预备服务器 / AutoDL 项目环境、依赖导入、Hydra compose 和静态检查通过。
- M4 已完成：AutoDL H20 上 mjlab GPU reset/step 和 SONIC order diagnostics 通过，action dim 为 29。
- M5 已完成：真实 env 输出 `actor_obs (2, 930)`、`critic_obs (2, 1789)`，policy/action/order 检查通过。
- M6 已完成：单卡 PPO smoke 和双卡 PPO smoke 均完成 2 个 learning iterations，无 OOM/NaN。
- M7 已完成：默认 `data/motion_lib_bones_seed/robot_filtered` 和默认 `data/bones_seed_smpl` 已同步到 AutoDL 数据盘，完成 `sonic_release` 默认路径双卡训练；加载 `129785` 个 robot motion 文件，`data/bones_seed_smpl` 含 `131455` 个 `.pkl` 文件。
- 默认数据注意事项：`robot_filtered` 有 7 个 motion stem 在本机和远端 `data/bones_seed_smpl` 中都没有同名 SMPL `.pkl`，当前 loader 会将这些条目置为 `None`；这不是同步遗漏。

## 剩余输入
- AutoDL `/root/autodl-tmp` 当前剩余约 `2.0G`，正式训练前建议清理数据盘或换更大数据盘，避免 checkpoint / 日志写满磁盘。
- 如需接近正式训练规模，还需单独确认目标 `num_envs`、训练时长、日志策略和是否启用 W&B 在线记录。
