# Scope Review

时间：2026-06-18T21:30:01+08:00

## 结论
范围应从“继续 H2 训练”收缩为“先诊断再设计”。本阶段只完成 grounding、计划和 review，等待用户确认 `PLAN.md`。

## 覆盖项
- `pr12000 / hy2000 / tf2000` 的日志、metrics、可视化拆解。
- global root / heading drift 是否为 reward / termination 盲区的最小实验。
- H2 decoder / adapter 方案，保护 G1 Universal Token 表征。
- cross-humanoid 长期结构规划：morphology embedding、part-level encoder/decoder、多机器人联合训练。

## 不包含
- 不启动新的长期训练。
- 不把 reward/termination 权重继续盲目扫参。
- 不修改 CUDA、驱动、base 环境或 workflow 源码。
- 不在用户确认前改 `UniversalTokenModule`、checkpoint key 或训练器。

## 主要风险
如果跳过 checkpoint 诊断直接开训，大概率重复已有 `precision <-> survival` 摆动。
