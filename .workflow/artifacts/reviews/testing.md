# Testing Review

时间：2026-05-11T12:01:50+08:00

## 结论
测试链覆盖服务器训练 smoke 的关键失败模式：依赖导入、GPU runtime、env 构造、仿真 step、obs/order、PPO 更新。

## 必跑测试
- `uv run python --version`
- `uv run python` 导入 `torch`、`mujoco`、`mjlab`、`mujoco_warp` 并打印 CUDA/GPU 信息。
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`
- Hydra compose `sonic_release` + `use_mjlab=True sim_type=mjlab`。
- GPU env reset/step + order diagnostics。
- `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 ... num_envs=16 ... num_learning_iterations=10`。

## 验收口径
- smoke 通过的最低标准是 10 iteration PPO 完成、reward finite、无 NaN/OOM、日志目录存在。
- 如果当前 CUDA 13 lock 失败，不能把失败归因于服务器驱动问题并要求改驱动；必须先尝试项目级 CUDA 12.x 依赖修复。
- 如果默认数据检查失败，需要区分数据缺失、权限问题、路径配置问题和代码问题。

## 未覆盖项
- 多 GPU、多进程大规模训练。
- 默认 `num_envs=4096` 完整训练。
- rough/trimesh terrain 性能验证。
- checkpoint resume、导出、render/eval、W&B 在线曲线。
