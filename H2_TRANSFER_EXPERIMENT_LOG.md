# SONIC G1 -> H2 迁移实验记录

更新时间：2026-06-18  
分支：`cross_humanoid`  
项目路径：`/home/ykj/project/SONICMJ/GR00T-WholeBodyControl`  
PKU 路径：`/home/nvme02/GR00T/GR00T`

## 目标

在 SONIC/SONICMJ 现有 G1 whole-body motion tracking 基础上迁移到 Unitree H2，优先关注 tracker，而不是 planner。整体策略是尽量复用官方 G1 权重，通过 H2 重定向动作数据和少量配置/训练阶段调整完成 H2 适配。

主要训练入口与配置：

- 训练入口：`gear_sonic/train_agent_trl.py`
- 评估入口：`gear_sonic/eval_agent_trl.py`
- H2 配置：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
- eval callback：`gear_sonic/trl/callbacks/im_eval_callback.py`
- checkpoint 加载相关：`gear_sonic/trl/trainer/ppo_trainer.py`

## 数据与评估集

### H2 训练数据

使用已经重定向好的 H2 数据：

```text
/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered
```

全量数据量：

```text
129785 motions
```

### Simple curriculum 子集

为早期 curriculum 训练创建过一个 symlink 子集：

```text
/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_curriculum_simple
```

数量：

```text
27515 motions
```

筛选思路：保留 `idle / walk / reach / turn_walk / idle_turn` 等较简单动作，排除 `jog / jump / run / dance / crouch / crawl / kick / punch / flip / climb / fall / roll` 等较难动作。

### 固定 8 motion 评估集

用于跨阶段比较：

```text
/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_eval_compare8
```

包含：

```text
jog_ff_start_180_R_003__A234_M
walk_ff_loop_315_R_001__A237_M
checking_time_R_001__A235
itching_left_body_side_R_001__A238_M
brush_off_dust_001__A235
body_stretch_V004_001__A238_M
jump_ff_180_R_003__A238_M
sneeze_R_001__A235
```

主要指标：

- `success`: 完整跑完并通过 eval termination 的比例，越高越好。
- `progress`: motion 完成比例，越高越好。
- `mpjpe_g`: global MPJPE，越低越好。
- `mpjpe_l`: local MPJPE，越低越好。
- `foot_g`: foot global MPJPE，越低越好。
- `vr_g`: VR/关键点 global MPJPE，越低越好。

## 实验总览

| 阶段 | 方案 | 初始化 | 数据 | 主要目的 | 当前结论 |
| --- | --- | --- | --- | --- | --- |
| 0 | G1 官方权重 shape-aware warm-start 到 H2 | 官方 G1 checkpoint | H2 全量 | 验证 G1 权重能否部分加载到 H2 | 可启动，可训练，但 H2 tracking 不稳 |
| 1 | full H2 baseline | warm-start 后继续训 | H2 全量 | 建立 H2 baseline | progress 会升，但 MPJPE 变差 |
| 2 | strict reward trick | `orig2000` | H2 全量 | 尝试更强 tracking reward | 无明显改善，不建议继续 |
| 3 | simple curriculum | `orig2000` | H2 simple 子集 | 降低动作难度，先学简单动作 | progress 略升，MPJPE 仍变差 |
| 4 | tracking_first_full | `orig2000` | H2 全量 | 先提高 survival/progress | success/progress 大幅提高，但 global tracking 漂移 |
| 5 | precision_refine | `tf2000` | H2 全量 | 从可存活策略精修 MPJPE | MPJPE 降低，但 success/progress 下降 |
| 6 | hybrid_balance | `pr12000` | H2 全量 | 在低 MPJPE 基础上拉回 progress | 有折中，但仍未达到理想平衡 |

## 关键评估结果

固定 8 motion eval 对比：

| checkpoint | success | progress | mpjpe_g | mpjpe_l | foot_g | vr_g | terminated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `orig2000` | 0.000 | 0.1379 | 0.0720 | 0.0526 | 0.0876 | 0.0605 | 8/8 |
| `orig4000` | 0.000 | 0.1776 | 0.0948 | 0.0671 | 0.1013 | 0.0888 | 8/8 |
| `orig5750` | 0.125 | 0.4229 | 0.1299 | 0.0822 | 0.1459 | 0.1274 | 7/8 |
| `strict100` | 0.000 | 0.1500 | 0.0760 | 0.0592 | 0.0865 | 0.0608 | 8/8 |
| `curr4000` | 0.000 | 0.2120 | 0.1089 | 0.0705 | 0.1479 | 0.1002 | 8/8 |
| `tf2000` | 0.625 | 0.6891 | 0.1547 | 0.0889 | 0.1727 | 0.1668 | 3/8 |
| `pr2000` | 0.500 | 0.6273 | 0.1570 | 0.0662 | 0.1591 | 0.1705 | 4/8 |
| `pr4000` | 0.625 | 0.7096 | 0.3898 | 0.0628 | 0.3898 | 0.3929 | 3/8 |
| `pr8000` | 0.000 | 0.0962 | 0.0690 | 0.0532 | 0.1001 | 0.0705 | 8/8 |
| `pr12000` | 0.000 | 0.1918 | 0.0703 | 0.0538 | 0.0921 | 0.0668 | 8/8 |
| `pr14000` | 0.250 | 0.3909 | 0.1166 | 0.0607 | 0.1356 | 0.1281 | 6/8 |
| `hy2000` | 0.250 | 0.4005 | 0.1338 | 0.0670 | 0.1600 | 0.1477 | 6/8 |
| `hy4000` | 0.500 | 0.5948 | 0.1823 | 0.0617 | 0.2002 | 0.1959 | 4/8 |

## 各阶段细节

### 0. 官方 G1 权重 warm-start 到 H2

目标：尽量复用 Sonic 官方 G1 权重，而不是从零训练 H2。

实际行为：

```text
shape-aware loaded: 47
skipped shape: 8
missing/unexpected: 0
```

说明：结构兼容部分能正常加载，H2 特有或 shape 不一致的部分跳过/重新初始化。

结论：可以作为 H2 训练起点，但不能假设 G1 latent/action 表征能直接迁移到 H2。后续实验也证明，单靠 warm-start 会出现 survival 与 tracking 精度冲突。

### 1. Full H2 baseline

原始 run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260613_171155
```

关键 checkpoint：

```text
model_step_002000.pt
model_step_004000.pt
last.pt around 5750
```

结果：

- `orig2000` MPJPE 低，但 progress 很低。
- `orig5750` progress/success 提高，但 MPJPE 明显变差。

结论：baseline 会学到一些 survival/progress，但不是稳定提升 tracking precision。

### 2. Strict reward trick

run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_strict_anchor_from2000-20260614_130241
```

设置：从 `orig2000` 初始化，重置 optimizer，增强 tracking reward。

结果：

```text
strict100: success=0.0 progress=0.1500 mpjpe_g=0.0760 mpjpe_l=0.0592
```

结论：没有明显优于 `orig2000`，不建议继续这个方向。

### 3. Simple curriculum

run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_curriculum_simple_from2000-20260614_233737
```

数据：

```text
/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_curriculum_simple
```

目标：通过简单动作子集降低初期难度，减少直接全量 H2 训练的崩溃。

结果：

```text
curr4000: success=0.0 progress=0.2120 mpjpe_g=0.1089
```

结论：progress 比 `orig4000` 略好，但 MPJPE 更差，说明 curriculum 仍然主要学到“撑久一点”，不是精确 tracking。

### 4. Tracking-first full

run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_tracking_first_full_from2000-20260615_131142
```

初始化：

```text
orig2000
```

核心思路：

- 使用全量 H2 数据。
- 增强 tracking reward，但也放松一部分 termination。
- 目标是先获得能跑完整段的策略。

代表评估：

```text
tf2000: success=0.625 progress=0.6891 mpjpe_g=0.1547
```

结论：

- survival/progress 大幅提高。
- 但 global MPJPE、foot、VR 点显著变差。
- 该阶段可作为“可存活初始化”，但不是最终 tracker。

### 5. Precision refine

run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_precision_refine_from_tf2000-20260615_220728
```

初始化：

```text
tf2000
```

核心思路：

- 收紧 tracking reward 和 termination。
- 从可存活策略回压 MPJPE。
- 牺牲部分 progress/success，换 tracking precision。

关键结果：

```text
pr8000:  success=0.0 progress=0.0962 mpjpe_g=0.0690
pr12000: success=0.0 progress=0.1918 mpjpe_g=0.0703
pr14000: success=0.25 progress=0.3909 mpjpe_g=0.1166
```

结论：

- `pr8000/pr12000` 的 global MPJPE 最好，接近或略优于 `orig2000`。
- 但 success/progress 很差。
- `pr14000` 稍微拉回 progress，但 MPJPE 变差。
- 说明该方案能压精度，但容易回到“短时间精确、跑不完整”的状态。

### 6. Hybrid balance

run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_hybrid_balance_from_pr12000-20260617_225628
```

初始化：

```text
pr12000
```

核心思路：

- 从低 MPJPE checkpoint 出发。
- 比 precision 稍微放松 termination。
- reward 介于 tracking_first 和 precision 之间。
- 目标是得到 `mpjpe_g < 0.10` 且 `progress > 0.4` 的折中策略。

结果：

```text
hy2000: success=0.25 progress=0.4005 mpjpe_g=0.1338
hy4000: success=0.50 progress=0.5948 mpjpe_g=0.1823
```

结论：

- progress/success 被拉回来了。
- 但 MPJPE 明显变差，尤其继续训练到 `hy4000` 后向“能跑但漂移”方向发展。
- `hy2000` 是目前 hybrid 中较好的折中点，但仍未达到目标。
- 因此已经停止该 run，避免继续烧卡。

## 当前保留 checkpoint 价值

### 精度优先

```text
pr8000 / pr12000
```

优点：

- `mpjpe_g` 最低，约 `0.069-0.070`。

缺点：

- `success=0`
- `progress` 很低。

用途：

- 适合作为 imitation/auxiliary refinement 或 anchor correction 的起点。

### 折中候选

```text
hy2000
```

优点：

- `progress≈0.40`
- 比 precision 高。

缺点：

- `mpjpe_g≈0.134`，仍偏高。

用途：

- 可用于可视化，判断是否虽然 MPJPE 高但视觉上可接受。
- 可作为下一步结构改造的对照。

### 存活优先

```text
tf2000 / hy4000
```

优点：

- success/progress 高。

缺点：

- global tracking 漂移明显。

用途：

- 证明当前 reward/termination 能让 H2 活下来。
- 不适合作为最终 tracker。

## 当前判断

只靠 reward/termination 继续调，会在两个极端之间摆动：

1. 精度好但跑不完整：`pr8000/pr12000`
2. 能跑但 global drift 明显：`tf2000/hy4000`

这说明当前问题不只是训练时长或 reward 权重，而是 H2 策略表征/动作空间/全局 anchor 对齐仍然没有被充分约束。

## 建议下一步

### 方案 1：可视化关键 checkpoint

优先可视化：

```text
pr12000
hy2000
tf2000
```

目标：确认 MPJPE 与视觉质量是否一致，尤其检查：

- root/global drift
- foot sliding
- upper-body tracking
- early termination 的具体原因

### 方案 2：显式 anchor/global correction

从现有结果看，global MPJPE 与 progress 的冲突明显。下一步可以考虑加入或加强显式 root/anchor correction，例如：

- 额外 loss/reward 直接约束 anchor trajectory。
- 将 anchor tracking 的尺度与 termination 解耦。
- 在 eval/training 中记录 per-motion root drift，用于自适应采样。

### 方案 3：supervised / imitation auxiliary 阶段

从 `pr12000` 或 `hy2000` 出发，加入小规模 imitation auxiliary，使 policy 不只依赖 RL reward 学 tracking：

- 目标：压低 MPJPE 同时不让 progress 崩掉。
- 可先只做 decoder/action head 相关模块的辅助训练。
- 后续再恢复 PPO 微调。

### 方案 4：结构方案

如果继续只调 reward 无法突破，应该进入真正的跨人形结构改造：

- robot-conditioned token/decoder
- H2 morphology embedding
- H2-specific decoder adapter
- G1 encoder 保留，H2 decoder/adapter 重新训练或部分微调

这更接近最初的跨人形设想，而不是继续假设 G1 latent/action 表征能自然迁移到 H2。

## 重要路径

训练日志：

```text
/home/nvme02/GR00T/GR00T/logs_pku/
```

RL run：

```text
/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/
```

评估结果：

```text
/home/nvme02/GR00T/GR00T/logs_eval/20260614_compare_h2/
/home/nvme02/GR00T/GR00T/logs_eval/20260615_curriculum_compare_h2/
/home/nvme02/GR00T/GR00T/logs_eval/20260615_tracking_first_compare_h2/
/home/nvme02/GR00T/GR00T/logs_eval/20260617_precision_compare_h2/
/home/nvme02/GR00T/GR00T/logs_eval/20260618_hybrid_compare_h2/
```

## 备注

- 所有训练均未修改 CUDA/driver。
- 训练命令均使用项目内 `./.tools/uv/uv run ...`。
- 目前 reward/termination 方案均通过 Hydra override 进行，没有为这些阶段额外修改项目代码。
- 本文记录的是截至 2026-06-18 的实验状态。
