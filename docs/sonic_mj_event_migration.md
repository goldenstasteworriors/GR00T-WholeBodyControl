# SonicMJ Event Migration Matrix

更新时间：2026-05-03

本文记录 SONIC IsaacLab event 到 SonicMJ/mjlab event 的当前迁移状态。状态含义：

- `等价`：字段语义和触发时机在 mjlab 中有直接对应实现。
- `近似`：训练链路可用，但 MuJoCo/mjlab 物理接口与 IsaacLab/PhysX 不完全同构。
- `未映射`：当前没有等价实现，保留为已知限制。

## Event 对照

| SONIC event | IsaacLab source | SonicMJ/mjlab target | 状态 | 说明 |
|---|---|---|---|---|
| `reset_scene_to_default` | IsaacLab scene reset | `mjlab.envs.mdp.reset_scene_to_default` | 等价 | 作为 reset event 固定注入，先恢复默认 scene 状态。 |
| `add_joint_default_pos` | `gear_sonic.envs.manager_env.mdp.randomize_joint_default_pos` | `sonic_mj.mdp.events.randomize_joint_default_pos` | 等价 | 透传 `asset_cfg`、`pos_distribution_params`、`operation`、`distribution`；同步更新 `JointPositionAction` offset。 |
| `base_com` | `gear_sonic.envs.manager_env.mdp.randomize_rigid_body_com` | `mjlab.envs.mdp.dr.body_com_offset` | 等价 | 透传 body 选择、xyz range、`operation`、`distribution`、`axes`、`shared_random`。 |
| `randomize_rigid_body_mass` | `gear_sonic.envs.manager_env.mdp.randomize_rigid_body_mass` | `mjlab.envs.mdp.dr.pseudo_inertia` for `scale` | 近似 | `scale` 映射为 pseudo-inertia log alpha，使 mass/inertia 一起缩放；比只改 mass 更物理一致，但与 IsaacLab 原事件不完全同构。非 `scale` 操作回退到 `dr.body_mass`。 |
| `physics_material` | `gear_sonic.envs.manager_env.mdp.randomize_rigid_body_material` | `sonic_mj.mdp.events.randomize_physics_material` | 近似 | 保留 bucket 采样和 geom/body 选择。static friction 映射到 MuJoCo sliding friction；dynamic friction best-effort 映射到 torsional friction；restitution 暂不映射。 |
| `push_robot` | `gear_sonic.envs.manager_env.mdp.push_by_setting_velocity` | `mjlab.envs.mdp.push_by_setting_velocity` | 等价 | 透传 interval、`velocity_range` 和 `asset_cfg`；默认 evaluation 中通过 `train_only_events` 临时禁用。 |

## 仍需复查的差异

- `physics_material.restitution_range` 没有直接 MuJoCo solver 参数映射；暂未实现，避免引入未经验证的接触稳定性变化。
- `physics_material.dynamic_friction_range` 与 PhysX dynamic friction 语义不同，目前只写入 MuJoCo friction 第二维。
- `randomize_rigid_body_mass` 的 `scale` 迁移为 pseudo-inertia 后，长训练分布可能与 IsaacLab 略有差异，需要继续观察 reward/termination 和 NaN。
- 如果后续加入 hand/object/table 相关 event，需要按 mjlab asset/body/geom 命名重新逐项核对。

## 当前验证入口

- 配置构建：`sonic_mj/env_cfg.py`
- event 实现：`sonic_mj/mdp/events.py`
- eval 中禁用训练专用 event：`sonic_mj/wrapper.py`
- 原始 SONIC event 配置：`gear_sonic/config/manager_env/events/`
