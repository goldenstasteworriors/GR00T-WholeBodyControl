# Grounding: H2 迁移诊断与结构方案

时间：2026-06-18T21:30:01+08:00

## 已读资料
- `AGENTS.md`：确认 SonicMJ 使用 uv，不使用 conda；不得修改 CUDA/显卡驱动/系统 GPU 组件；外部仓库只读；代码相关回答要注明位置。
- `PERSON.md`：没有实际 open 条目。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：上一轮还是 PKU H2 训练/验证计划，本轮需要改成诊断和结构设计。
- `README.md`、`docs/README.md`：上游训练文档仍偏 IsaacLab；本项目以 `AGENTS.md` 的 SonicMJ/mjlab 约束为准。
- `H2_TRANSFER_EXPERIMENT_LOG.md`：本轮最关键证据，记录 G1->H2 迁移实验、固定 8 motion eval、checkpoint 价值和下一步建议。
- 目标代码：
  - `gear_sonic/train_agent_trl.py`
  - `gear_sonic/eval_agent_trl.py`
  - `gear_sonic/trl/callbacks/im_eval_callback.py`
  - `gear_sonic/trl/modules/universal_token_modules.py`
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
  - `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml`
  - `sonic_mj/mdp/rewards.py`
  - `sonic_mj/mdp/terminations.py`
  - `sonic_mj/env_cfg.py`

## 关键实验事实
- `orig2000/orig4000/orig5750` 说明 H2 warm-start 可启动，但 progress 提升伴随 MPJPE 变差。
- `strict100` 和 `curr4000` 没有突破，说明单纯 reward trick / 简单 curriculum 不够。
- `tf2000`：`success=0.625`、`progress=0.6891`、`mpjpe_g=0.1547`，代表能跑但 global tracking 漂移。
- `pr12000`：`success=0`、`progress=0.1918`、`mpjpe_g=0.0703`，代表短时间精确但跑不完整。
- `hy2000`：`success=0.25`、`progress=0.4005`、`mpjpe_g=0.1338`，代表折中但没有达到 `mpjpe_g < 0.10 && progress > 0.4`。
- 日志结论已经明确：继续只靠 reward / termination 会在“精度好但早死”和“能跑但 drift”之间摆动。

## 代码证据
- `sonic_mj/mdp/terminations.py`：
  - `anchor_pos` 只比较 `abs(anchor_z - robot_anchor_z)`。
  - `anchor_ori_full` 比较 quaternion error。
  - `foot_pos_xyz` 对脚部做 xyz norm，但没有全局 root XY / heading drift 的独立终止。
- `sonic_mj/mdp/rewards.py`：
  - `tracking_anchor_pos` 有全局 anchor position reward。
  - `tracking_vr_5point_local` 在 ref/robot anchor-local frame 比较，可能让局部 tracking 看起来合理但掩盖 global drift。
- `gear_sonic/trl/callbacks/im_eval_callback.py`：
  - 已收集 `ref_body_pos_extend` / `rigid_body_pos_extend`，计算 `mpjpe_g/mpjpe_l/progress/terminated`。
  - 适合扩展 root drift、heading drift、termination reason 的 logging 或离线拆解。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`：
  - H2 action dim 31，仍沿用 Universal Token 的 G1/teleop/SMPL encoder 和 `g1_kin/g1_dyn` decoder 结构。
  - `robot_conditioning.enabled=false` 是 no-op placeholder。
- `gear_sonic/trl/modules/universal_token_modules.py`：
  - 支持多 encoder/decoder、`active_decoders`、`freeze_encoders`、`freeze_decoders`。
  - 是 H2-specific decoder / adapter 的自然落点，但改 checkpoint key 前必须先询问用户。

## 参考资料映射
- SONIC 论文与官方训练代码提供 Universal Token、motion tracking 和 released checkpoint 的语义边界。
- 当前任务没有新增论文搜索需求；重点是基于已有项目证据做工程诊断和结构方案。
- 如果后续需要 cross-humanoid 学术依据，应优先在 Zotero 搜索 embodiment-conditioned policy、morphology-conditioned control、multi-robot imitation/RL 等主题。

## 风险
- 若不先拆解 `pr12000/hy2000/tf2000`，继续训练可能只是重复同一 trade-off，浪费 GPU。
- global-anchor 实验若直接上强 termination，可能把策略推回 `pr12000` 的早死模式；必须先 logging，再最小 ablation。
- H2 adapter 如果改动 `UniversalTokenModule` 或 checkpoint key 过大，会破坏 G1 checkpoint 可加载性；应优先用新增 decoder/config 和 shape-aware load。
- cross-humanoid 版本需要 body/part mapping、robot metadata 和多机器人数据配比，不能只靠 H2 单机器人训练自然得到。

## 当前判断
没有 blocking 问题阻止生成计划。本轮应停在等待用户确认 `PLAN.md`，确认后先执行 M0/M1 诊断，而不是启动新的长期训练。
