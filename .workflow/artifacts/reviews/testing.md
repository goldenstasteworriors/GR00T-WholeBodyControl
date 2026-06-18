# Testing Review

时间：2026-06-18T21:30:01+08:00

## 结论
测试应围绕“诊断指标可信”和“最小改动可回滚”。不要用长期训练替代诊断。

## 必跑测试
- `uv run python --version`
- `uv run python -m compileall -q sonic_mj gear_sonic/eval_agent_trl.py gear_sonic/trl/callbacks/im_eval_callback.py`
- 对 `pr12000 / hy2000 / tf2000` 用同一 eval set 复算或读取 metrics。
- global-anchor logging 版本 eval smoke。
- 如新增 reward/termination term，至少跑一次小 `num_envs` eval smoke。

## 验证记录
每次执行后写入 `PROGRESS.md`：
- 命令和退出码。
- checkpoint 与 eval 输出目录。
- per-motion progress / terminated / mpjpe_g / mpjpe_l。
- root XY drift、heading drift、foot/VR error。
- 是否创建临时脚本；若创建，确认已删除。

## 未覆盖项
- 长期收敛质量。
- 多机器人联合训练效果。
- H2 adapter 的最终网络结构性能。
这些必须等用户确认计划后分阶段执行。
