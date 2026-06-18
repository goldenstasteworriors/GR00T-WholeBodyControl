# TASK: H2 迁移诊断、global-anchor 最小实验与 cross-humanoid 结构计划

## 项目元信息
- 项目名：`GR00T-WholeBodyControl / SonicMJ`
- 目标代码库：`/home/ykj/project/SONICMJ/GR00T-WholeBodyControl`
- 当前分支：`cross_humanoid`
- 主要实验记录：`H2_TRANSFER_EXPERIMENT_LOG.md`
- workflow 请求：完成 grounding、计划和 Codex 多角色 review，停在等待用户确认 `PLAN.md`。

## 本轮目标
- [requirement] 基于 `H2_TRANSFER_EXPERIMENT_LOG.md` 和 SONIC G1->H2 迁移诊断建议，后续工作先做诊断和结构设计，不盲目继续训练。
- [requirement] 优先对 `pr12000`、`hy2000`、`tf2000` 做日志、metrics、termination、可视化拆解，解释“低 MPJPE 但跑不完整”和“能跑但 global drift”的差异。
- [requirement] 做一个 global-anchor 最小实验，验证 global root / heading drift 是否是当前 reward / termination 盲区。
- [requirement] 设计 H2 decoder / adapter 方案，避免继续全网络 PPO 破坏 G1 Universal Token 表征。
- [requirement] 规划真正 cross-humanoid 版本，包括 robot / morphology embedding、part-level encoder / decoder、H2-specific adapter、多机器人联合训练。
- [requirement] 当前阶段只完成 grounding、计划和 Codex 多角色 review，等待用户确认 `PLAN.md` 后再进入代码或训练执行。

## 环境与硬约束
- [constraint] 中文回答和记录。
- [constraint] 本项目 SonicMJ 的 mjlab 迁移使用 uv 管理环境，不使用 conda。
- [constraint] 不修改 CUDA、显卡驱动、Isaac Sim 底层安装或系统级 GPU 组件。
- [constraint] 不安装依赖到 base；依赖只能在项目 uv 环境中处理。
- [constraint] 不修改 workflow 源码仓库 `/home/ykj/tool/WORKFLOW`。
- [constraint] 不修改外部 `mjlab`、`InstinctMJ` 或原始参考仓库，除非用户明确要求。
- [constraint] 不回退或删除用户/历史未提交文件；当前已有 `H2_TRANSFER_EXPERIMENT_LOG.md`、`wandb/`、训练曲线图片等未跟踪内容，只读对待。
- [constraint] 临时测试脚本如需创建，测试完成后必须删除；计划阶段不创建临时测试脚本。
- [constraint] 代码相关回答必须注明代码位置；修改或添加文件后必须列出文件清单。

## 当前证据
- [fact] `H2_TRANSFER_EXPERIMENT_LOG.md` 显示继续调 reward / termination 会在两个极端间摆动：`pr8000/pr12000` 精度好但 progress / success 差，`tf2000/hy4000` survival / progress 好但 global MPJPE 和漂移明显，`hy2000` 只是折中但未达目标。
- [fact] 固定 8 motion eval 中：`tf2000 success=0.625 progress=0.6891 mpjpe_g=0.1547`，`pr12000 success=0 progress=0.1918 mpjpe_g=0.0703`，`hy2000 success=0.25 progress=0.4005 mpjpe_g=0.1338`。
- [fact] `sonic_mj/mdp/terminations.py` 的 `anchor_pos` 当前只检查 anchor z 高度差；`foot_pos_xyz` 检查脚部 xyz，`anchor_ori_full` 检查全姿态角度，但没有独立 global XY drift / heading drift termination。
- [fact] `sonic_mj/mdp/rewards.py` 有 `tracking_anchor_pos` 和 `tracking_anchor_ori`，但 `tracking_vr_5point_local` 在 anchor-local frame 比较关键点，相对 tracking 可能掩盖 global XY / heading 漂移。
- [fact] `gear_sonic/trl/callbacks/im_eval_callback.py` 已保存 global body position traces 并计算 `mpjpe_g/mpjpe_l/progress/terminated`，适合扩展或离线分析 per-motion root drift、heading drift、termination reason。
- [fact] `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml` 仍使用 G1/teleop/SMPL encoders 与 `g1_dyn/g1_kin` decoder 结构，且 `robot_conditioning.enabled=false` 只是占位。
- [fact] `gear_sonic/trl/modules/universal_token_modules.py` 已有 `active_encoders`、`active_decoders`、`freeze_encoders`、`freeze_decoders`、token cache 等机制，可作为 H2 adapter / decoder 分阶段训练的落点。

## 计划产物
- [artifact] `.workflow/artifacts/grounding.md`：项目现状、关键证据、参考资料、风险。
- [artifact] `.workflow/artifacts/paper_project_map.json`：论文/参考实现/目标代码映射。
- [artifact] `.workflow/artifacts/open_questions.json`：开放问题，无 blocking 时允许生成计划。
- [artifact] `.workflow/artifacts/reviews/*.md`：scope、milestones、architecture、testing、executor review。
- [artifact] `PLAN.md`：等待用户确认的可执行计划。

## 验收条件
- [acceptance] `TASK.md` 保留硬需求、环境约束、验收条件，并明确不盲目继续训练。
- [acceptance] `grounding.md` 记录 H2 三个关键 checkpoint、global drift 假设、代码位置和风险。
- [acceptance] `paper_project_map.json` 结构化记录 SONIC 论文/参考实现/目标代码之间的映射；若未新增论文检索，说明原因。
- [acceptance] `open_questions.json` 使用 `{ "questions": [...] }` 格式；没有 blocking 问题时可生成 `PLAN.md`。
- [acceptance] `PLAN.md` 包含诊断、global-anchor 最小实验、H2 decoder/adapter、cross-humanoid 规划的里程碑、完成标准、测试命令和 review 点。
- [acceptance] `PROGRESS.md` 记录本轮 grounding、plan、review、阻塞/非阻塞问题和下一步。
