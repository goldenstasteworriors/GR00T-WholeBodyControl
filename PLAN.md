# PLAN: H2 迁移诊断与 cross-humanoid 结构方案

状态：等待用户确认。确认前不启动训练、不改训练代码。

## 总体策略
- 先解释现有 `pr12000 / hy2000 / tf2000` 的行为差异，再决定是否继续训练。
- 用最小 global-anchor 实验验证 root XY / heading drift 是否是 reward / termination 盲区。
- H2 后续训练优先走 decoder / adapter 分阶段路线，冻结或保护 G1 encoder / token 表征，避免继续全网络 PPO 乱改。
- cross-humanoid 版本单独规划为结构升级，不把它混进短期 reward trick。

## M0 - 基线恢复与证据固化
任务：
- 读取 `H2_TRANSFER_EXPERIMENT_LOG.md`、`TASK.md`、`AGENTS.md`、`PROGRESS.md`、`task.json`。
- 确认 `pr12000`、`hy2000`、`tf2000` 的 checkpoint/run/eval 路径和固定 8 motion 评估集。
- 收集已有 `metrics_eval.json`、训练日志、W&B/本地曲线、render 输出路径；若路径缺失，只记录缺失，不补跑训练。

完成标准：
- `PROGRESS.md` 写入三个 checkpoint 的 run/eval 路径、已有文件、缺失文件。
- 形成 checkpoint 对比表：success、progress、mpjpe_g、mpjpe_l、foot_g、vr_g、terminated、主要失败 motion。

测试/命令：
```bash
uv run python --version
uv run python -m compileall -q sonic_mj gear_sonic/eval_agent_trl.py gear_sonic/trl/callbacks/im_eval_callback.py
```

## M1 - pr12000 / hy2000 / tf2000 诊断与可视化拆解
任务：
- 对三个 checkpoint 做同一评估集的离线 metrics 拆解：per-motion progress、terminated、mpjpe_g/mpjpe_l、foot/VR error。
- 优先复用 `gear_sonic/eval_agent_trl.py` 与 `gear_sonic/trl/callbacks/im_eval_callback.py` 已有输出；必要时增加只读分析脚本/小工具，测试后删除临时脚本。
- 可视化至少覆盖 root/global drift、heading drift、foot sliding、upper-body tracking、termination 时刻。

完成标准：
- 输出一份诊断记录，说明 `pr12000` 为什么早死、`tf2000` 为什么 global 漂、`hy2000` 是否只是折中。
- 不得仅以平均 MPJPE 排序；必须包含 per-motion 和时间序列判断。

测试/命令：
```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/eval_agent_trl.py \
  checkpoint=<CHECKPOINT> use_mjlab=True sim_type=mjlab headless=True \
  ++num_envs=8 ++output_dir=<EVAL_OUT>
```

## M2 - global-anchor 最小实验
任务：
- 设计最小配置，不改变主干网络，只增加或 override global anchor 诊断项。
- 第一层只做 logging：记录 per-step anchor XY error、anchor Z error、heading error、root yaw drift、termination reason。
- 第二层做 ablation：加入 global XY / heading reward 或 termination 的最小版本，与原配置在固定 8 motion 上对比。
- 最小实验只跑短程，不继续长训练。

候选代码位置：
- reward/termination：`sonic_mj/mdp/rewards.py`、`sonic_mj/mdp/terminations.py`
- 配置装配：`sonic_mj/env_cfg.py`
- eval 记录：`gear_sonic/trl/callbacks/im_eval_callback.py` 或 wrapper extras

完成标准：
- 如果开启 global-anchor 约束后 `tf2000/hy2000` 的 global drift 下降但 progress 可控，说明 drift 盲区假设成立。
- 如果约束只导致更早 termination 而不改善 MPJPE，需要回到表征/decoder 方案，不继续加权重。

测试/命令：
```bash
uv run python -m compileall -q sonic_mj gear_sonic/trl/callbacks/im_eval_callback.py
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/eval_agent_trl.py \
  checkpoint=<CHECKPOINT> use_mjlab=True sim_type=mjlab headless=True \
  ++num_envs=8 ++manager_env.config.terrain_type=plane ++output_dir=<ANCHOR_EVAL_OUT>
```

## M3 - H2 decoder / adapter 方案设计
任务：
- 保留 G1/teleop/SMPL encoders、FSQ token 和大部分 backbone，优先新增 H2-specific action decoder 或 residual adapter。
- 设计三种候选并排序：
  1. `h2_dyn` decoder：token + proprioception -> H2 31 DOF action，冻结 encoders/quantizer，先训练 decoder。
  2. action residual adapter：在 G1-compatible latent/action 上输出 H2 residual，限制幅度并加 regularization。
  3. morphology-conditioned decoder：token + proprioception + robot embedding -> action，作为 cross-humanoid 过渡。
- 明确 checkpoint 加载策略：G1 权重只初始化共享模块，H2 decoder/adapter 新初始化；不破坏原 checkpoint key，必要时用 shape-aware load。

候选代码位置：
- decoder 配置：`gear_sonic/config/actor_critic/decoders/`
- actor config：`gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml`
- H2 exp config：`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
- module 支持：`gear_sonic/trl/modules/universal_token_modules.py`
- checkpoint shape-aware：`gear_sonic/trl/utils/checkpoint.py`

完成标准：
- 写出 adapter 配置草案、冻结策略、可加载 checkpoint key 预期、训练阶段和失败判据。
- 用户确认前不改 `UniversalTokenModule` 主逻辑；如必须改 checkpoint key，先暂停询问。

测试/命令：
```bash
uv run python -m compileall -q gear_sonic/trl/modules gear_sonic/trl/utils/checkpoint.py
```

## M4 - H2 adapter 小规模训练计划
任务：
- 制定 decoder-only / adapter-only 训练流程：短 rollout smoke -> 固定 8 motion eval -> 小数据 overfit -> H2 full data 短训。
- 对比 `pr12000/hy2000/tf2000`，目标不是单一 MPJPE，而是同时看 `progress > 0.4` 和 `mpjpe_g < 0.10` 的可行性。
- 训练命令必须显式使用项目 uv 环境；远端 GPU 限制如需使用 PKU，延续用户给定 GPU 规则并写入命令。

完成标准：
- 给出阶段训练命令模板和停止条件：NaN、OOM、global drift 未改善、G1 shared 表征大幅漂移、checkpoint 不可加载。
- 明确不会把 `sonic_release/last.pt` warm-start 当 full resume。

## M5 - cross-humanoid 版本规划
任务：
- 设计真正跨机器人架构，不只服务 H2：
  - robot/morphology embedding：DOF、body graph、link length、mass/inertia、actuator limits、canonical part ids。
  - part-level encoder/decoder：lower body、torso、left/right arm、hands 分部 token。
  - robot-specific adapter：G1/H2 共享 token，机器人专属 action heads。
  - 多机器人联合训练：G1 保持原性能，H2 学习新 embodiment，采样比例和 aux loss 防遗忘。
- 明确数据需求：G1 motion_lib、H2 retargeted motion_lib、统一 body/part mapping、per-robot eval set。

完成标准：
- 产出结构路线图：短期 H2 adapter、中期 morphology-conditioned decoder、长期 multi-robot joint training。
- 列出必须先确认的问题和不可跨越的边界。

## M6 - 文档、review 与交付
任务：
- 更新 `TASK.md`、`.workflow/artifacts/grounding.md`、`.workflow/artifacts/paper_project_map.json`、`.workflow/artifacts/open_questions.json`、review 文件和 `PROGRESS.md`。
- 确认没有 blocking 问题后，停在等待用户确认本 `PLAN.md`。

完成标准：
- `open_questions.json` 没有 `blocking=true`。
- `PROGRESS.md` 记录已完成 grounding/plan/review 和下一步。

## Review 点
- scope：本阶段以诊断和设计为主，不继续长训练，不盲目调 reward。
- milestones：先拆解 checkpoint，再做最小 global-anchor 实验，再设计 adapter，最后规划 cross-humanoid。
- architecture：保护 G1 Universal Token 表征；H2 差异先落在 decoder/adapter，不默认全网络 PPO。
- testing：所有新诊断项必须有 compile/eval smoke；临时测试脚本用完删除。
- executor：执行前先确认 PLAN；不动 CUDA/驱动，不安装到 base，不修改 workflow 源码。
