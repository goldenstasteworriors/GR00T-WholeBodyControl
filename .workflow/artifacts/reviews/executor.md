# Executor Review

时间：2026-06-18T21:30:01+08:00

## 执行原则
- 用户确认 `PLAN.md` 前不改训练代码、不启动训练。
- 使用项目 uv 环境，不安装到 base。
- 不修改 CUDA、显卡驱动、Isaac Sim 底层或系统 GPU 组件。
- 不修改 workflow 源码仓库 `/home/ykj/tool/WORKFLOW`。
- 不删除非本轮创建内容；已有未跟踪训练日志/图片只读对待。
- 临时测试脚本用完删除。

## 分支处理
- 如果 eval metrics 缺少 per-step trace：先补跑 eval/logging，不直接训练。
- 如果 global-anchor logging 显示 drift 明显：再做最小 reward/termination ablation。
- 如果 global-anchor 只导致早死：停止 reward trick，转入 H2 decoder/adapter。
- 如果 adapter 需要改 checkpoint key 或 trainer：暂停并询问用户。

## 停止条件
- 需要修改 CUDA/驱动/base 环境。
- 需要修改外部仓库或 workflow 源码。
- 需要在用户确认前启动长期训练。
- 计划执行中出现 blocking open question。
