# Architecture Review

时间：2026-05-11T12:01:50+08:00

## 结论
计划不改变 SonicMJ 架构，只处理项目环境兼容和服务器验证，符合迁移约束。

## 检查
- 训练入口仍是 `gear_sonic/train_agent_trl.py`，通过 `use_mjlab=True` / `sim_type=mjlab` 进入 `sonic_mj.train.create_mjlab_manager_env`。
- SONIC 语义由 `sonic_mj/assets.py`、`sonic_mj/env_cfg.py`、`sonic_mj/mdp/commands.py`、`sonic_mj/mdp/observations.py` 和 `sonic_mj/wrapper.py` 保持。
- 依赖兼容只应影响 `pyproject.toml` / `uv.lock` / `.venv`，不应触及 `UniversalTokenModule`、PPO trainer、checkpoint key 或网络结构。
- order diagnostics 被纳入服务器验证，能防止环境修复后误改 29-DOF action/order。

## 风险
- `mujoco-warp` 与 `torch` / CUDA wheel 的组合可能有限；如果必须降级或 pin 版本，应优先保持 `mjlab==1.2.0` 可用并记录原因。
- 不应引入外部仓库 shim 或修改 `/home/ykj/project/SONICMJ/mjlab`。
