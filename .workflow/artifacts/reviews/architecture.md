# Architecture Review

时间：2026-06-18T21:30:01+08:00

## 结论
当前 H2 迁移不应继续默认全网络 PPO。架构上应保护 G1 encoders、FSQ token 和 shared representation，把 H2 embodiment 差异优先放到 decoder/adapter。

## 关键边界
- `gear_sonic/trl/modules/universal_token_modules.py` 已支持多 decoder、active decoder 和 freeze 选项，是 adapter 方案的自然落点。
- `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml` 是当前 action decoder 模板；新增 `h2_dyn` 比改名覆盖 `g1_dyn` 更利于 checkpoint 兼容和 ablation。
- `sonic_mj/mdp/terminations.py` 当前缺少 root XY / heading drift 独立 termination；这是 global-anchor 最小实验的主要验证点。
- `sonic_mj/mdp/rewards.py` 的 local 关键点 reward 可能掩盖 global drift；需要 per-step drift logging 验证。

## 风险
- 修改 checkpoint key 或 `UniversalTokenModule` forward 语义会扩大 blast radius，必须先询问用户。
- morphology embedding 若只做一个 learned id，没有 body graph / part mapping，很难称为真正 cross-humanoid。
