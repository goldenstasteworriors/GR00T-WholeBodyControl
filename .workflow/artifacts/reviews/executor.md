# Executor Review

时间：2026-05-11T12:01:50+08:00

## 执行原则
- 先运行只读诊断，再决定是否改依赖。
- 所有 Python 命令使用 `uv run`。
- 依赖安装或重锁只作用于项目 uv 环境，不进入 base。
- 不修改 CUDA、驱动、系统 GPU 组件。
- 临时测试脚本尽量不用；必须创建时测试后删除。
- 每个阶段结果写回 `PROGRESS.md`。

## 可能分支
- 分支 A：现有 CUDA 13 lock 在 H20 服务器可用。直接进入 M3-M6，记录“无需依赖修改”。
- 分支 B：CUDA 13 wheel 失败。修改项目依赖到 CUDA 12.x 兼容组合，更新 lock，重建 `.venv`，再复跑 M1-M6。
- 分支 C：依赖可导入但 `mujoco-warp` 或 mjlab runtime 失败。先定位 Python package 组合，再考虑代码兼容修复；不触碰外部 mjlab 仓库。
- 分支 D：训练 smoke OOM 或 NaN。先降低 `num_envs` 到 8 或 4 定位；若仍失败，按代码/依赖/数据分类继续修复。

## 停止条件
- `PLAN.md` 未确认前不进入执行。
- 需要修改外部仓库、系统 CUDA/驱动、trainer/network/checkpoint key 时停止询问用户。
- 服务器缺少数据且无法用现有 smoke 数据替代时，记录 blocking open question。
