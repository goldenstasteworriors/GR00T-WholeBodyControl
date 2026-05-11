# Grounding: SonicMJ 服务器训练兼容化

时间：2026-05-11T12:01:50+08:00

## 已读取材料
- `AGENTS.md`：确认项目范围、uv 环境约束、不得修改 CUDA/驱动、SONIC MuJoCo order、mjlab 迁移验证要求和 workflow 规则。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md` / `PLAN.md` / `PROGRESS.md` / `task.json`：确认历史任务是本机迁移验证，本轮新目标是 H20 服务器训练兼容 smoke，并停在 `PLAN.md` 确认。
- `README.md` / `docs/README.md` / `pyproject.toml` / `uv.lock`：确认上游 README 仍描述 IsaacLab 训练；本项目根配置使用 uv 管理 SonicMJ/mjlab 环境。
- `gear_sonic/train_agent_trl.py`：确认 `use_mjlab=True` 或 `sim_type=mjlab` 时进入 `sonic_mj.train.create_mjlab_manager_env`。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`：默认 3 encoder 训练配置，`num_envs=4096`，默认 terrain 为 `trimesh`。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`：SOMA encoder 参考配置，默认 `soma_motion_file=data/motion_lib_bones_seed/soma_filtered`。
- `sonic_mj/`：当前 mjlab backend 代码，包括 env cfg、motion command、obs/reward/termination/event/curriculum 和 TRL wrapper。
- `.workflow/artifacts/*`：读取已有 grounding、paper map、open questions 和多角色 review，用于保留历史证据但重置本轮目标。

## 服务器目标事实
- 用户提供服务器信息：`nvcc 11.6`；`nvidia-smi Driver Version 570.124.06`；`CUDA Version 12.8`；GPU 为 `NVIDIA H20`。
- 硬约束：不能修改 CUDA、显卡驱动、系统级 GPU 组件；所有依赖调整必须发生在项目 uv 环境或项目文件内。
- 关键判断：`nvidia-smi CUDA Version 12.8` 表示驱动最高支持的 CUDA runtime 能力，不等于必须使用系统 CUDA toolkit；PyTorch wheel 自带 CUDA runtime，但 wheel 的 CUDA 主版本仍必须被驱动支持。
- 风险推断：当前 `uv.lock` 锁定 CUDA 13 相关 wheel，而服务器驱动 570.124.06 最高报告 CUDA 12.8，存在 CUDA 13 runtime wheel 无法加载或运行的高风险。该风险需要在服务器上先用只读导入命令验证；若失败，应切换项目级 CUDA 12.x wheel，而不是改驱动。

## 当前项目环境事实
- `pyproject.toml`：
  - Python `>=3.10,<3.11`
  - `mjlab==1.2.0`
  - `mujoco-warp>=3.8.0,<3.9`
  - `trl==0.28.0`
  - `accelerate>=1.3.0`
  - uv source 中 `mujoco` 使用 `https://py.mujoco.org`。
- `uv.lock`：
  - 锁定 `torch-2.11.0`
  - 包含 `cuda-toolkit 13.0.2`
  - 包含 `nvidia-cuda-runtime 13.0.96`、`nvidia-cudnn-cu13`、`nvidia-nccl-cu13`、`nvidia-cusparselt-cu13` 等 CUDA 13 wheel。
- 当前本地 `.venv` 只作为证据，不代表服务器：
  - `uv run python --version` 输出 `Python 3.10.20`
  - `torch 2.11.0+cu130`
  - `torch.version.cuda=13.0`
  - `torch.cuda.is_available=True`
  - `mujoco 3.8.1`
  - `mjlab 1.2.0`
- 当前 shell 有 conda `twist2` 的 `libtinfo` warning；执行阶段必须显式使用 `uv run`，不得向 base 安装依赖。

## 数据事实
- 当前仓库存在：
  - `data/motion_lib_bones_seed/robot_filtered`
  - `data/motion_lib_bones_seed/robot_medium`
  - `data/motion_lib_bones_seed/robot_smoke`
  - `data/motion_lib_bones_seed/soma_uniform_medium`
  - `data/motion_lib_bones_seed/soma_uniform_smoke`
  - `data/smpl_filtered`
  - `data/bones_seed_smpl`
- 当前未发现：
  - `data/motion_lib_bones_seed/soma_filtered`
- 结论：`sonic_release` 默认 robot/smpl 数据路径当前具备；`sonic_bones_seed` 默认 SOMA 路径仍可能需要 override 到 `soma_uniform_*` 或补数据。

## 历史验证证据
- `PROGRESS.md` 记录 2026-05-08 已在本机完成：
  - `compileall sonic_mj gear_sonic/train_agent_trl.py` 通过。
  - `sonic_release` mjlab env cfg 构造通过。
  - G1 XML actuator order 与 `sonic_mj/assets.py` canonical joint order 一致，action dim 29。
  - CPU reset/step 和 20 步 rollout 通过。
  - `sonic_release` 10 iteration PPO smoke 通过。
  - `sonic_release` medium 数据 100 iteration PPO 训练通过。
  - `sonic_bones_seed` SOMA 20 iteration 短训练通过。
- 这些结果证明代码路径曾可训练，但不能替代 H20 服务器依赖兼容验证。

## 关键风险
- 当前 CUDA 13 依赖锁与服务器 Driver 570 / CUDA 12.8 兼容性存在高风险；这是本轮计划的第一优先级。
- `mujoco-warp` 可能依赖 CUDA runtime / JIT / 编译能力；服务器 `nvcc 11.6` 与项目 wheel 的 CUDA runtime 不一致时必须区分“系统 nvcc 不匹配”和“Python wheel 不兼容”。
- 如果需要把 `torch` 切到 CUDA 12.x，必须确认 `mjlab==1.2.0`、`mujoco-warp>=3.8.0,<3.9`、`trl==0.28.0` 的可用组合，并用 uv 锁定，不得 ad hoc pip 安装到 base。
- H20 GPU 的显存通常足够 smoke，但默认 `num_envs=4096`、rough/trimesh terrain、完整数据训练仍可能因时间或资源受限；本轮验收只要求服务器训练 smoke。
- Universal Token 对 obs key/shape/order 敏感；环境修复不应通过修改 trainer、网络结构或 checkpoint key 来绕过。

## 结论
- 当前没有阻塞生成计划的问题；服务器信息足够制定可执行计划。
- `PLAN.md` 应先验证服务器上现有锁文件是否可运行；若 CUDA 13 wheel 不兼容，再在项目 uv 环境内切换到 CUDA 12.x 兼容依赖，并复跑 compile、compose、reset/step、order diagnostics 和 PPO smoke。
- 按 workflow 规则，本阶段完成 grounding/plan/review 后停止，等待用户确认 `PLAN.md`。
