# AutoDL H20 测试服务器验证记录

日期：2026-05-11  
项目：SONICMJ / `GR00T-WholeBodyControl`  
远端项目目录：`/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`  
登录入口：`ssh -p 37716 root@region-42.seetacloud.com`  

> 说明：本文不记录服务器密码。测试过程中未修改 CUDA、显卡驱动或系统级 GPU 组件。

## 1. 服务器环境

### GPU / 驱动

GPU 开启后的测试服务器环境：

- GPU：2 x `NVIDIA H20`
- 单卡显存：约 `97871 MiB`
- `nvidia-smi` Driver：`580.105.08`
- `nvidia-smi` reported CUDA：`13.0`
- `nvcc`：该 AutoDL 服务器未找到 `nvcc`

早期无 GPU 阶段同一 AutoDL 环境曾显示：

- 系统：Ubuntu 20.04.4
- host：`autodl-container-e9b742b627-c588c445`
- GPU 未开启时：`nvidia-smi` 返回 `No devices were found`
- `nvcc --version`：CUDA `11.3, V11.3.109`

### 项目 Python / 依赖

远端项目环境使用项目内 `.venv`：

- Python：`3.10.20`
- PyTorch：`torch 2.7.1+cu126`
- `torch.version.cuda`：`12.6`
- `torch.cuda.is_available()`：`True`
- `torch.cuda.device_count()`：`2`
- `mujoco`：`3.8.1`
- `mujoco_warp`：可导入
- `mjlab`：可导入

为兼容服务器和 H20，项目依赖锁定到了 CUDA 12.6 wheel 组合：

- `pyproject.toml` 中显式加入 `torch==2.7.1`
- `uv.lock` 解析到 `torch 2.7.1`、`torchvision 0.22.1`、`triton 3.3.1` 及 NVIDIA cu12 wheel

## 2. 数据放置与同步

所有测试数据均放在 AutoDL 数据盘 `/root/autodl-tmp` 下，不放系统盘。

远端项目目录：

```bash
/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl
```

### 已同步数据

- `gear_sonic/data/assets/`
  - 用于补齐 G1 MuJoCo XML 等资产。
- `data/motion_lib_bones_seed/robot_smoke`
- `data/motion_lib_bones_seed/robot_medium`
- `data/motion_lib_bones_seed/robot_filtered`
  - 大小约 `7.9G`
  - `.pkl` 数量：`129785`
- `data/bones_seed_smpl`
  - 从本机 `data/smpl_filtered/` 同步到远端默认路径。
  - 大小约 `31G`
  - `.pkl` 数量：`131455`
- `data/smpl_filtered`
  - 远端保留为符号链接：`data/smpl_filtered -> bones_seed_smpl`

同步 SMPL 默认数据的命令：

```bash
rsync -az --info=progress2 --partial --inplace \
  -e 'ssh -o UserKnownHostsFile=/tmp/sonicmj_known_hosts -o StrictHostKeyChecking=no -p 37716' \
  data/smpl_filtered/ \
  root@region-42.seetacloud.com:/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl/data/bones_seed_smpl/
```

### 数据注意事项

`robot_filtered` 中有 7 个 motion stem 在本机和远端 `data/bones_seed_smpl` 中都没有同名 SMPL `.pkl`：

- `kneeling_loop_002__A098_M`
- `neutral_button press_001__A543`
- `neutral_button press_001__A543_M`
- `neutral_button press_001__A544`
- `neutral_button press_001__A544_M`
- `neutral_button press_001__A545`
- `neutral_button press_001__A545_M`

在 `/home/ykj/Downloads/dataset/bones-seed` 中也未找到这 7 个同名 `.pkl`，因此不是同步遗漏。当前 `MotionLibBase` 对缺失 SMPL 文件会将对应条目置为 `None`。

补齐完整 SMPL 后，AutoDL 数据盘状态：

- `/root/autodl-tmp` 总量：`50G`
- 剩余空间：约 `2.0G`
- 使用率：约 `97%`

正式训练前建议清理数据盘或换更大数据盘，避免 checkpoint / 日志写满磁盘。

## 3. GPU 导入检查

命令：

```bash
cd /root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl
.venv/bin/python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory // 1024**2)
PY
```

结果：

- Python `3.10.20`
- `torch 2.7.1+cu126`
- CUDA runtime `12.6`
- `torch.cuda.is_available()` 为 `True`
- `torch.cuda.device_count()` 为 `2`
- 两张 GPU 均为 `NVIDIA H20`

## 4. mjlab reset / step 与顺序诊断

测试目标：

- 创建 mjlab env
- reset 成功
- step 成功
- 检查 observation / action shape
- 检查 SONIC MuJoCo joint/body/action/order

配置要点：

```bash
+exp=manager/universal_token/all_modes/sonic_release
use_mjlab=True
sim_type=mjlab
checkpoint=null
num_envs=2
headless=True
++algo.config.num_learning_iterations=1
++algo.config.num_steps_per_env=2
++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke
++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
manager_env.config.terrain_type=plane
```

结果：

- 加载 `210` 个 `robot_smoke` motions
- `actor_obs` shape：`(2, 930)`
- `critic_obs` shape：`(2, 1789)`
- action dim：`29`
- reward / done shape：`(2,)`
- reward finite
- mean reward：`-0.47829321026802063`
- order diagnostics 全部通过：
  - robot joints match SONIC MuJoCo
  - robot bodies match
  - motion bodies match
  - action joints match
  - policy `joint_pos` / `joint_vel` / `actions` order match
  - motion dof mapping identity

## 5. 单卡 PPO smoke

### 首次过小 batch 测试

命令：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m accelerate.commands.launch --num_processes=1 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=2 headless=True \
  ++algo.config.num_learning_iterations=1 \
  ++algo.config.num_steps_per_env=2 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

结果：

- 训练入口能启动并完成 1 iteration
- 但配置过小，出现 batch 配置错误：

```text
batch_size must be a multiple of num_mini_batches, inexact division: 2 / 4 = 0.5
```

判定：不是 GPU / 环境不可用，而是 smoke 参数太小。

### 修正后单卡 smoke

命令：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m accelerate.commands.launch --num_processes=1 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=4 headless=True \
  ++algo.config.num_learning_iterations=2 \
  ++algo.config.num_steps_per_env=4 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

结果：

- 退出码 `0`
- 完成 learning iteration `1` 和 `2`
- 初始化 `g1` / `teleop` / `smpl` encoders
- 初始化 `g1_dyn` / `g1_kin` decoders
- total episodes：`8`
- total timesteps：`32`
- 无 OOM / NaN

## 6. 双卡 PPO smoke

命令：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -m accelerate.commands.launch --num_processes=2 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=4 headless=True \
  ++algo.config.num_learning_iterations=2 \
  ++algo.config.num_steps_per_env=4 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

结果：

- 退出码 `0`
- Accelerate 进入 multi-GPU 模式
- rank 0 使用 `cuda:0`
- rank 1 使用 `cuda:1`
- 完成 learning iteration `1` 和 `2`
- total episodes：`16`
- total timesteps：`64`
- mean rewards：`-6.04001`
- 无 OOM / NaN
- 退出时有非致命 PyTorch distributed 清理警告：

```text
destroy_process_group() was not called before program exit
```

## 7. 默认 robot 数据双卡短训练

命令：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -m accelerate.commands.launch --num_processes=2 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=4 headless=True \
  ++algo.config.num_learning_iterations=2 \
  ++algo.config.num_steps_per_env=4 \
  manager_env.config.terrain_type=plane
```

说明：

- 未覆盖 `motion_file`
- 使用 `sonic_release.yaml` 默认 robot motion 路径：
  - `data/motion_lib_bones_seed/robot_filtered`
- 当时默认 `data/bones_seed_smpl` 还未补齐，loader 对缺失 SMPL 路径容忍并使用 `None` 数据项

结果：

- 退出码 `0`
- 两个 rank 均加载 `129785` 个 robot motion 文件
- 完成 learning iteration `1` 和 `2`
- total episodes：`16`
- total timesteps：`64`
- mean reward：
  - iteration 1：`-2.69212`
  - iteration 2：`-12.68186`
- 无 OOM / NaN

## 8. 完整默认路径双卡训练

补齐 `data/bones_seed_smpl` 后，运行完整默认数据路径测试。

命令：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -m accelerate.commands.launch --num_processes=2 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=16 headless=True \
  ++algo.config.num_learning_iterations=10 \
  manager_env.config.terrain_type=plane
```

关键点：

- 未覆盖 `motion_file`
- 未覆盖 `smpl_motion_file`
- 使用 `sonic_release.yaml` 默认：
  - `data/motion_lib_bones_seed/robot_filtered`
  - `data/bones_seed_smpl`
- 双卡多进程
- 每个 rank 创建 `16` 个 mjlab env

结果：

- 退出码 `0`
- 完成 learning iteration `1` 到 `10`
- 两个 rank 均加载 `129785` 个 robot motion 文件
- 从默认完整 SMPL 路径读取对应 SMPL 数据
- 初始化 `g1`、`teleop`、`smpl` 三个 encoder
- tokenizer 观测包含：
  - `smpl_joints_multi_future_local_nonflat (10, 72)`
  - `smpl_root_ori_b_multi_future (10, 6)`
  - `joint_pos_multi_future_wrist_for_smpl (10, 6)`
- policy obs shape：`(930,)`
- critic obs shape：`(1789,)`
- action dim：`29`
- 第 10 iteration：
  - computation：`434 steps/s`
  - total episodes：`320`
  - total timesteps：`7680`
  - mean rewards：`-24.92989`
  - mean length：`10.49500`
- 无 OOM、NaN 或 CUDA 初始化错误

日志目录：

```bash
logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260511_152916
```

## 9. 每卡 env 容量测试

### 测试说明

当前 `accelerate --num_processes=2` 下，`num_envs=N` 会让每个 rank / 每张卡各自创建 `N` 个 env。因此这里记录的是每卡 env 数。

共同设置：

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python -m accelerate.commands.launch --num_processes=2 \
gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True sim_type=mjlab \
  num_envs=<N> headless=True \
  ++algo.config.num_learning_iterations=1 \
  manager_env.config.terrain_type=plane
```

数据路径未覆盖，使用默认真实数据：

- `data/motion_lib_bones_seed/robot_filtered`
- `data/bones_seed_smpl`

### 4096 env / GPU

命令中的关键参数：

```bash
num_envs=4096
++algo.config.num_learning_iterations=1
manager_env.config.terrain_type=plane
```

结果：

- 失败，退出码 `1`
- 两个 rank 均创建了 `4096` env
- 两个 rank 均加载 `129785` 个 motion 文件
- 失败点：Warp CUDA graph 创建阶段 OOM

关键错误：

```text
Warp CUDA error 2: out of memory
RuntimeError: Graph creation error: Warp CUDA error 2: out of memory
```

### 2048 env / GPU

命令中的关键参数：

```bash
num_envs=2048
++algo.config.num_learning_iterations=1
manager_env.config.terrain_type=plane
```

结果：

- 通过，退出码 `0`
- 完成 learning iteration `1`
- 每卡 `2048` env
- 双卡总 env 数 `4096`
- computation：`12761 steps/s`
- collection：`6.250s`
- total episodes：`4096`
- total timesteps：`98304`
- mean rewards：`-22.35138`

### 3072 env / GPU

命令中的关键参数：

```bash
num_envs=3072
++algo.config.num_learning_iterations=1
manager_env.config.terrain_type=plane
```

结果：

- 通过，退出码 `0`
- 完成 learning iteration `1`
- 每卡 `3072` env
- 双卡总 env 数 `6144`
- computation：`11792 steps/s`
- collection：`10.570s`
- total episodes：`6144`
- total timesteps：`147456`
- mean rewards：`-18.76445`

### 3584 env / GPU

按用户要求停止继续测试，手动 SIGTERM 中断，不计入容量结论。

### 当前建议

- 保守建议先使用 `num_envs=2048`
  - 每卡 `2048`
  - 双卡总 env 数 `4096`
- 已知 `num_envs=3072` 也能完成 1 iteration
- `num_envs=4096` 每卡不可用，会在 Warp CUDA graph 创建阶段 OOM

## 10. 清理状态

多次测试后均检查：

```bash
pgrep -af "gear_sonic/train_agent_trl.py|accelerate.commands.launch|accelerate launch|rsync|uv sync" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

最终状态：

- 无遗留训练进程
- 无遗留 rsync / uv sync 进程
- GPU 0 显存占用：`0 MiB`
- GPU 1 显存占用：`0 MiB`

## 11. 已验证与未验证

### 已验证

- 服务器项目 uv 环境可用
- `torch 2.7.1+cu126` 可在 H20 上初始化 CUDA
- `mujoco` / `mujoco_warp` / `mjlab` 可导入
- mjlab reset / step
- SONIC MuJoCo joint/body/action/order 诊断
- 单卡 PPO smoke
- 双卡 PPO smoke
- 默认 `robot_filtered` 数据加载
- 默认 `data/bones_seed_smpl` 数据加载
- 完整默认路径双卡训练闭环
- 每卡 env 容量基础测试

### 未验证

- 从官方 Bones-SEED 原始 CSV 重新运行数据转换：
  - `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`
  - `gear_sonic/data_process/filter_and_copy_bones_data.py`
- SOMA full 数据处理：
  - `gear_sonic/data_process/extract_soma_joints_from_bvh.py`
  - `data/motion_lib_bones_seed/soma_filtered`
- 官方正式规模长训
- checkpoint resume / finetune：
  - `+checkpoint=sonic_release/last.pt`
- rough / trimesh terrain 正式训练
- render / eval
- ONNX export
- W&B 在线记录

## 12. 总结

当前 AutoDL 双 H20 测试服务器已经证明：

- SONICMJ 的 mjlab backend 可以在服务器 GPU 上运行。
- 默认真实数据路径 `robot_filtered + bones_seed_smpl` 可以完成双卡训练。
- 当前依赖组合 `torch 2.7.1+cu126` 不需要修改系统 CUDA / 驱动即可运行。
- 保守训练规模建议先用 `num_envs=2048`，即每卡 2048、双卡总 4096。

主要风险：

- AutoDL 数据盘剩余约 `2G`，正式训练前需要清理或扩容。
- 每卡 `4096` env 已确认 OOM。
- 还没有测试 checkpoint finetune、SOMA full 数据和长时间训练。
