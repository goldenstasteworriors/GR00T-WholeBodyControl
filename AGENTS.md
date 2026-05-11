# Agent 指南 - GR00T-WholeBodyControl/gear_sonic 到 mjlab 迁移（SonicMJ）

## 作用范围

- 本指南适用于 SonicMJ 分支/项目下的全部内容。
- 除非用户明确要求跨项目修改，否则改动应限制在当前项目内。
- `GR00T-WholeBodyControl/gear_sonic`、`gear_sonic_deploy`、`mjlab` 和 `InstinctMJ` 仓库默认仅用于参考。
- 未经用户明确要求，不得修改原始 `mjlab`、`InstinctMJ` 或其它外部仓库。
- 不得修改 CUDA、显卡驱动、Isaac Sim 底层安装或系统级 GPU 环境。
- 本项目 SonicMJ 的 mjlab 迁移使用 uv 管理环境，不使用 conda。
- SONIC 官方 Bones-SEED 数据集本机路径：`/home/ykj/Downloads/dataset/bones-seed`。
- 人体模型/SMPL/SMPLX 相关资源本机路径：`/home/ykj/commonly_used/body_models`。
- 如果遇到命名、资产选择、顺序映射、训练入口、依赖环境等不确定问题，必须先停止并询问用户，不要自行猜测。

## 核心目标

- 在 `SonicMJ` 内以 mjlab 原生方式复现 `gear_sonic` 的 SONIC 训练环境。
- 优先迁移训练闭环：motion command、环境 reset/step、观测、动作、奖励、终止、事件随机化、课程学习。
- 保留 SONIC 的训练语义：G1/teleop/SMPL/SOMA 多编码器、motion_lib PKL 数据、future reference frames、Universal Token policy 输入输出约定。
- 文件/模块命名在可行范围内贴近原 `gear_sonic` 结构；Python 包名建议使用小写，例如 `sonic_mj`。
- 迁移时优先参考 `mjlab` 中相似实现，采用 mjlab 原生接口与组织方式。

## 源训练链路

- 训练入口来自 `gear_sonic/train_agent_trl.py`。
- 默认实验配置来自 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`。
- 环境配置来自 `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`。
- MDP term 来自 `gear_sonic/envs/manager_env/mdp/`：
  - `commands.py`：`TrackingCommand`，负责 motion_lib 加载、参考帧推进、编码器模式采样。
  - `observations.py`：policy/critic/tokenizer/teacher 等观测组。
  - `rewards.py`：motion tracking 奖励。
  - `terminations.py`：tracking 失败和 timeout 终止。
  - `events.py`：reset、randomization、push、domain randomization。
  - `actions.py`：关节位置动作。
- 机器人配置来自 `gear_sonic/envs/manager_env/robots/g1.py`，其中包含 G1 资产、PD 参数、action scale、IsaacLab 与 MuJoCo 顺序映射。
- 策略与训练器来自 `gear_sonic/trl/`。第一阶段不要重写 `UniversalTokenModule`、PPO trainer 或辅助损失。

## 迁移规则

- 先迁移环境与 manager terms，再接入现有 TRL 训练；不要一开始重写策略网络。
- 优先复用 `gear_sonic` 的函数职责和类层级；原实现有明确类/函数边界时，迁移后保持对应结构。
- 不得无依据新增兼容层、桥接层、adapter shim 或过渡包装。
- 不得以“更简洁”为由合并或隐藏原版关键逻辑，尤其是 motion command、坐标变换、关节顺序映射、采样策略。
- reward 必须改为 mjlab reward-term 表达；其他 manager 若存在表达差异，也按 mjlab manager/config 模式迁移。
- 地形、reset、command、观测、动作、终止、事件随机化、课程学习必须保持行为一致。
- 迁移后清理 Isaac Lab 接口残留，包括 import、类型、字段、注释和路径。
- SONIC 特有命名保持稳定，仅在 mjlab 必要字段映射处改名。

## 旧框架到 mjlab 对齐要点

- Isaac Lab 的 `ManagerBasedRLEnvCfg` 对齐到 mjlab 的 RL env config。
- Isaac Lab 的 `InteractiveSceneCfg` 对齐到 mjlab `SceneCfg` 和 MuJoCo 资产配置。
- Isaac Lab 的 `ArticulationCfg`、`UrdfFileCfg`、`ContactSensorCfg`、`TerrainImporterCfg` 不应原样保留。
- 原 `prim_path="{ENV_REGEX_NS}/Robot"`、USD/URDF spawn 逻辑迁移为 mjlab/MuJoCo 资产加载方式。
- `asset_name`、`sensor_name` 等旧字段迁移时按 mjlab 要求改为 `entity_name` / `SceneEntityCfg` 等原生表达。
- manager 均采用 mjlab 字典式配置：`actions`、`observations`、`commands`、`rewards`、`terminations`、`events`、`curriculum`。
- `TrackingCommand` 必须保留以下能力：
  - 加载 robot/SMPL/SOMA motion PKL。
  - 每个 env 采样 motion id 和 start time。
  - 维护当前帧与 multi-future reference。
  - 支持 G1/teleop/SMPL/SOMA encoder sampling。
  - 支持 adaptive sampling。
  - 输出 reward/observation/termination 所需的 reference tensors。
- observation group 需要保持名称和顺序稳定，尤其是 `policy`、`critic`、tokenizer 输入项和 `actor_obs` 维度。
- reward term 名称尽量保持原名，例如 `tracking_anchor_pos`、`tracking_relative_body_pos`、`tracking_vr_5point_local`、`feet_acc`。
- termination term 名称尽量保持原名，例如 `time_out`、`anchor_pos`、`anchor_ori_full`、`ee_body_pos`、`foot_pos_xyz`。
- events/domain randomization 迁移时先实现训练必要项：joint default offset、mass randomization、physics material、push robot。
- terrain 先支持 `plane`，再迁移 rough terrain/trimesh；不要在基础训练未通之前扩展复杂地形。
- camera、render、teleop、object/table/HOI 逻辑不作为第一阶段训练迁移阻塞项，除非目标实验明确依赖。

## SONIC 机器人资产与顺序约束

- 不得直接复用 InstinctMJ 的 `g1_29dof_torsobase_popsicle.xml` 作为 SONIC 默认资产。
- SONIC 默认 MuJoCo 资产应优先使用：
  `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`。
- SONIC motion_lib PKL 的 29-DOF joint order 以 `g1_29dof_rev_1_0.xml` actuator order 为准：
  left leg 6、right leg 6、waist 3、left arm 7、right arm 7。
- InstinctMJ 的 G1 XML 使用 torso-base/popsicle 结构，joint order 从 waist 开始；该顺序不得作为 SONIC motion_lib、policy action 或 checkpoint 的默认顺序。
- 迁移到 mjlab 后必须明确一个 canonical order：
  - 第一阶段建议使用 SONIC MuJoCo order 作为 canonical order。
  - 只有在兼容旧 IsaacLab checkpoint/config 时，才保留 IsaacLab 与 MuJoCo 映射。
- 不得删除或弱化以下映射语义：
  `isaaclab_to_mujoco_dof`、`mujoco_to_isaaclab_dof`、`isaaclab_to_mujoco_body`、`mujoco_to_isaaclab_body`。
  若 mjlab 环境内部已使用 SONIC MuJoCo order，这些映射可以退化为兼容检查/加载旧配置用，但必须显式验证。
- root/anchor 语义以 SONIC 为准：
  - `anchor_body` 默认保持 `pelvis`。
  - `TrackingCommand`、reward、termination、reset 中的 root pose 不得改成 InstinctMJ 的 torso-root 语义，除非同步重做 motion_lib 和 checkpoint 兼容策略。
- 初始姿态、默认关节位置、action scale、PD/armature/effort/velocity limit 优先从 `gear_sonic/envs/manager_env/robots/g1.py` 迁移。
- `JointPositionAction` 必须保持 SONIC 语义：
  `target_joint_pos = default_joint_pos + action * action_scale`。
- 资产迁移完成后必须打印并核对：
  - mjlab robot joint names/order
  - mjlab robot body names/order
  - motion_lib joint/body order
  - action term joint order
  - policy observation 中 `joint_pos`、`joint_vel`、`actions` 的顺序

## 兼容性约束

- `SonicMJ` 应首先兼容 `sonic_release` 训练配置。
- `sonic_bones_seed` 的 SOMA 编码器作为第二阶段兼容目标。
- 不破坏 motion_lib PKL 数据格式，不重命名数据字段，不改变 fps/未来帧语义。
- 不破坏 checkpoint 可加载性；网络结构、obs key、obs dim、action dim 改动必须有明确迁移理由。
- 默认 G1 29-DOF body policy action dim 应保持 29。
- 若引入 43-DOF hand/primitive 支持，必须兼容原 `ManagerEnvWrapper` 中 body/hand action 分离语义。
- W&B、Accelerate、TRL 配置接口尽量保持原命令行 override 方式。
- 如果必须修改 TRL trainer、`UniversalTokenModule` 或 checkpoint key，先停止并询问用户。
- 如果必须修改外部 `mjlab`、`InstinctMJ` 或原始 `GR00T-WholeBodyControl` 仓库，先停止并询问用户。
- 调试和依赖安装必须在项目对应 conda 环境中进行；不确定环境名时先询问用户。
- 不得安装依赖到 base 环境。
- 不得修改 CUDA、显卡驱动或 Isaac Sim 底层版本。

## 建议迁移顺序

1. 建立 SonicMJ 包结构和最小 mjlab 训练入口。
2. 迁移 G1 MuJoCo 资产配置、默认关节位置、action scale、关节/body 名称表。
3. 迁移 `TrackingCommand` 的 motion_lib 加载、reset 采样、step 更新时间、当前帧和未来帧输出。
4. 迁移 policy/critic/tokenizer 观测项，先跑通 obs dim 对齐。
5. 迁移动作项，确认 action dim、scale、default offset 与旧训练一致。
6. 迁移核心 rewards 和 terminations。
7. 接入现有 TRL PPO 训练器，先用小 `num_envs` 和 sample_data 做 smoke test。
8. 再迁移 events/randomization、terrain、adaptive sampling、SOMA、render/eval/ONNX export。

## 验证要求

- 第一阶段至少验证：
  - 环境可 reset。
  - 单步 step 不报错。
  - `policy`、`critic`、tokenizer 观测维度与旧配置一致。
  - action dim 与 G1 29-DOF 配置一致。
  - motion command 能加载 sample_data。
- 小规模训练命令优先使用：
  `num_envs=16 headless=True ++algo.config.num_learning_iterations=10`。
- 测试脚本如临时创建，测试完成后必须删除。
- 交付前列出运行过的命令、结果和未验证项。

## 代码质量

- 遵循当前项目风格，保持代码清晰、模块化，避免无关重构。
- 迁移过程中清理无用 import、死代码、旧路径和过期注释。
- 涉及项目代码的回答必须注明代码位置。
- 修改或添加文件/代码后，回答最后必须列出修改/添加了哪些文件/代码。
- 不得删除非自己创建或明确属于本次任务的文件内容。
- 每一步代码修改操作后都需要关注 git 状态；是否提交、推送或上传到远程必须按用户明确指令执行。

<!-- AGENTIC-WORKFLOW:BEGIN -->
## WORKFLOW Runtime Rules

以下规则只适用于当前目标项目被 `workflow` skill 接管后的 agent 会话；维护 workflow skill 本身时不适用。

1. 启动后先读 `PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json` 和当前项目的 `AGENTS.md`。
2. 先处理 `PERSON.md` 中仍为 `open` 的条目；问题要回答，新增要求或修正要同步进 `TASK.md` 或 `PLAN.md`。
3. 不依赖历史对话上下文，必须通过项目文档和 git 状态恢复进度。
4. `task.json` 只是轻量运行摘要，不是状态机；不要靠旧 gate/milestone 字段推进。
5. 代码修改后必须运行与风险匹配的测试，并把命令和结果写入 `PROGRESS.md`。
6. 不得修改 CUDA、显卡驱动、系统级 GPU 组件；依赖只能安装在项目指定 conda/uv 环境中，不能装到 base。
7. workflow 源码仓库 `/home/ykj/tool/WORKFLOW` 对目标项目 agent 只读，除非用户明确要求维护 workflow 工具本身。
<!-- AGENTIC-WORKFLOW:END -->
