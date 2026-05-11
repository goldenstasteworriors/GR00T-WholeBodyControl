# Scope Review

时间：2026-05-11T12:01:50+08:00

## 结论
范围清晰：本轮不是继续扩大 SonicMJ 功能迁移，而是让当前项目在指定 H20 服务器上用 uv 环境完成训练 smoke test。

## 覆盖项
- 服务器事实纳入任务：Driver `570.124.06`、CUDA runtime `12.8`、`nvcc 11.6`、GPU `NVIDIA H20`。
- 明确禁止修改 CUDA、显卡驱动和系统级 GPU 组件。
- 明确依赖修复边界：只允许改项目级 `pyproject.toml`、`uv.lock`、`.venv` 或文档。
- smoke test 范围合理：导入、GPU 初始化、compile、compose、reset/step、order diagnostics、PPO 10 iteration。

## 风险
- 当前 `uv.lock` 包含 CUDA 13 wheel，可能不适配 Driver 570；计划已将其作为 M1/M2 核心风险。
- 默认完整训练、大规模 `num_envs=4096`、rough terrain、多 GPU不是本轮验收；如果用户期望“完整训练”，需另开阶段。

## 建议
确认计划后先在服务器运行 M0/M1，不要先改依赖。只有出现明确兼容错误时再重锁 uv。
