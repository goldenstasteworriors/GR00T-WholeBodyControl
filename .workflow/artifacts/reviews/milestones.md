# Milestones Review

时间：2026-06-18T21:30:01+08:00

## 结论
里程碑顺序合理：先证据固化，再 checkpoint 拆解，再 global-anchor 最小实验，再 adapter 设计，最后 cross-humanoid 规划。

## 检查
- M0/M1 能解释当前失败模式，避免无效训练。
- M2 把 global drift 假设拆成 logging 和 ablation 两层，降低误判风险。
- M3/M4 把 H2 差异限制在 decoder/adapter 和短程训练计划中。
- M5 把真正 cross-humanoid 放到独立架构阶段，不污染短期诊断。

## 风险
- 如果已有 eval 输出缺少 per-step trace，M1 需要补跑 eval 或增加 logging；这属于计划确认后的执行内容。
- global-anchor ablation 如果过强，可能只复现 `pr12000` 早死，需要以小权重和日志为先。
