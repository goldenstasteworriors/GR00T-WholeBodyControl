# SonicMJ mjlab 官方文档命令对照目录

这个文件是官方 SONIC / GR00T-WholeBodyControl 文档的命令替换索引。用法是：先在下面目录里找到你正在看的官方 `README` 或 `docs/source/...` 页面，再进入对应小节，把官方命令替换成 SonicMJ/mjlab 版本。

本项目训练迁移的核心事实：

- 训练环境使用项目根 uv 环境，不使用 Isaac Lab conda/python 环境。
- `gear_sonic/train_agent_trl.py` 和 `gear_sonic/eval_agent_trl.py` 仍是入口，但要加 mjlab override。
- 训练入口位置：[gear_sonic/train_agent_trl.py](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/gear_sonic/train_agent_trl.py)
- eval 入口位置：[gear_sonic/eval_agent_trl.py](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/gear_sonic/eval_agent_trl.py)
- mjlab backend 位置：[sonic_mj/train.py](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/sonic_mj/train.py)、[sonic_mj/env_cfg.py](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/sonic_mj/env_cfg.py)
- uv/mjlab 依赖位置：[pyproject.toml](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/pyproject.toml)

## 快速规则

| 官方命令形态 | SonicMJ/mjlab 替换形态 |
|---|---|
| `python gear_sonic/train_agent_trl.py ...` | `uv run python gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...` |
| `accelerate launch ... gear_sonic/train_agent_trl.py ...` | `uv run accelerate launch ... gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...` |
| `python gear_sonic/eval_agent_trl.py ...` | `uv run python gear_sonic/eval_agent_trl.py ... ++use_mjlab=True ++sim_type=mjlab ...` |
| `accelerate launch ... gear_sonic/eval_agent_trl.py ...` | `uv run accelerate launch ... gear_sonic/eval_agent_trl.py ... ++use_mjlab=True ++sim_type=mjlab ...` |
| `python gear_sonic/data_process/*.py ...` | `uv run python gear_sonic/data_process/*.py ...` |
| `python download_from_hf.py ...` | `uv run python download_from_hf.py ...` |
| `pip install ...` | 优先用 `uv sync` 或 `uv pip install ...`，不要装到 base。 |

快速 smoke test 推荐先用平地：

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=10 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

## 官方文档索引

### 根目录和 docs

- [README.md](#readmemd)
- [docs/README.md](#docsreadmemd)

### Getting Started

- [docs/source/getting_started/installation_training.md](#docssourcegetting_startedinstallation_trainingmd)
- [docs/source/getting_started/download_models.md](#docssourcegetting_starteddownload_modelsmd)
- [docs/source/getting_started/quickstart.md](#docssourcegetting_startedquickstartmd)
- [docs/source/getting_started/installation_deploy.md](#docssourcegetting_startedinstallation_deploymd)
- [docs/source/getting_started/vr_teleop_setup.md](#docssourcegetting_startedvr_teleop_setupmd)

### User Guide

- [docs/source/user_guide/training.md](#docssourceuser_guidetrainingmd)
- [docs/source/user_guide/training_data.md](#docssourceuser_guidetraining_datamd)
- [docs/source/user_guide/configuration.md](#docssourceuser_guideconfigurationmd)
- [docs/source/user_guide/new_embodiments.md](#docssourceuser_guidenew_embodimentsmd)
- [docs/source/user_guide/troubleshooting.md](#docssourceuser_guidetroubleshootingmd)
- [docs/source/user_guide/teleoperation.md](#docssourceuser_guideteleoperationmd)

### Tutorials

- [docs/source/tutorials/keyboard.md](#docssourcetutorialskeyboardmd)
- [docs/source/tutorials/gamepad.md](#docssourcetutorialsgamepadmd)
- [docs/source/tutorials/zmq.md](#docssourcetutorialszmqmd)
- [docs/source/tutorials/manager.md](#docssourcetutorialsmanagermd)
- [docs/source/tutorials/vr_wholebody_teleop.md](#docssourcetutorialsvr_wholebody_teleopmd)
- [docs/source/tutorials/data_collection.md](#docssourcetutorialsdata_collectionmd)

### References

- [docs/source/references/training_code.md](#docssourcereferencestraining_codemd)
- [docs/source/references/deployment_code.md](#docssourcereferencesdeployment_codemd)
- [docs/source/references/motion_reference.md](#docssourcereferencesmotion_referencemd)
- [docs/source/references/decoupled_wbc.md](#docssourcereferencesdecoupled_wbcmd)
- [docs/source/references/planner_onnx.md](#docssourcereferencesplanner_onnxmd)
- [docs/source/references/observation_config.md](#docssourcereferencesobservation_configmd)
- [docs/source/references/conventions.md](#docssourcereferencesconventionsmd)
- [docs/source/references/index.md](#docssourcereferencesindexmd)
- [docs/source/references/jetpack6.md](#docssourcereferencesjetpack6md)

### API / Resources / External READMEs

- [docs/source/api/index.md](#docssourceapiindexmd)
- [docs/source/api/teleop.md](#docssourceapiteleopmd)
- [docs/source/resources/*.md](#docssourceresourcesmd)
- [decoupled_wbc/sim2mujoco/README.md](#decoupled_wbcsim2mujocoreadmemd)
- [external_dependencies/unitree_sdk2_python/README.md](#external_dependenciesunitree_sdk2_pythonreadmemd)

<a id="readmemd"></a>

## README.md

状态：有训练命令需要替换。

| 官方位置 | 官方命令 | 替换为 |
|---|---|---|
| Quick start 安装训练依赖 | `pip install -e "gear_sonic/[training]"` | `uv sync`。如果只缺少单个包，用 `uv pip install <pkg>`，不要装到 base。 |
| Quick start 下载 HF 依赖 | `pip install huggingface_hub` | `uv pip install huggingface_hub` |
| Quick start 下载训练 checkpoint/data | `python download_from_hf.py --training` | `uv run python download_from_hf.py --training` |
| Convert Bones-SEED | `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` | `uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` |
| Filter Bones-SEED | `python gear_sonic/data_process/filter_and_copy_bones_data.py ...` | `uv run python gear_sonic/data_process/filter_and_copy_bones_data.py ...` |
| Finetune | `accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py ...` | `uv run accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...` |
| Verify environment | `python check_environment.py` | `uv run python check_environment.py`；注意它仍可能检查官方 Isaac 环境，mjlab 以本文 smoke test 为准。 |
| Environment table | `Train / finetune SONIC = Isaac Lab's Python env` | SonicMJ 中改为项目根 `.venv` / uv 环境。 |

README 中 finetune 完整替换命令：

```bash
uv run accelerate launch --num_processes=8 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=4096 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

`bash install_scripts/install_mujoco_sim.sh`、`install_pico.sh`、`install_data_collection.sh`、部署文档命令不属于训练迁移，仍按官方环境使用。

<a id="docsreadmemd"></a>

## docs/README.md

状态：文档站构建说明，无 mjlab 训练命令。

| 官方命令 | 处理方式 |
|---|---|
| `pip install sphinx ...` | 如要在 SonicMJ uv 环境里构建 docs，用 `uv pip install sphinx ...`；否则可按文档独立环境执行。 |
| `make html` | 不需要追加 `use_mjlab=True`。 |
| `python -m http.server 8000` | 可写成 `uv run python -m http.server 8000`，但和 mjlab 迁移无关。 |

<a id="docssourcegetting_startedinstallation_trainingmd"></a>

## docs/source/getting_started/installation_training.md

状态：整页是官方 Isaac Lab 训练安装，需要按 SonicMJ 替换。

| 官方位置 | 官方命令 | 替换为 |
|---|---|---|
| Install Isaac Lab | `python -c "import isaaclab; ..."` | SonicMJ 不需要 Isaac Lab；改查 `uv run python -c "import torch, mujoco, mjlab; print(torch.__version__)"`。 |
| Install gear_sonic | `pip install -e "gear_sonic/[training]"` | `uv sync` |
| Download HF | `pip install huggingface_hub` | `uv pip install huggingface_hub` |
| Download training data | `python download_from_hf.py --training` | `uv run python download_from_hf.py --training` |
| Convert motion | `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` | `uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` |
| Filter motion | `python gear_sonic/data_process/filter_and_copy_bones_data.py ...` | `uv run python gear_sonic/data_process/filter_and_copy_bones_data.py ...` |
| Pre-flight | `python check_environment.py --training` | `uv run python check_environment.py --training`，但以 mjlab import/smoke 为准。 |
| Interactive/headless smoke | `python gear_sonic/train_agent_trl.py ...` | 加 `uv run`、`use_mjlab=True`、`sim_type=mjlab`，并显式传 motion path。 |

替换后的 headless smoke：

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=5 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

<a id="docssourcegetting_starteddownload_modelsmd"></a>

## docs/source/getting_started/download_models.md

状态：下载命令多数只需进入 uv；其中 eval/train 命令需要 mjlab override。

| 官方位置 | 官方命令 | 替换为 |
|---|---|---|
| Install dependency | `pip install huggingface_hub` / `pip install huggingface_hub[cli]` | `uv pip install huggingface_hub` / `uv pip install "huggingface_hub[cli]"` |
| Download script | `python download_from_hf.py` | `uv run python download_from_hf.py` |
| Training download | `python download_from_hf.py --training` | `uv run python download_from_hf.py --training` |
| Sample download | `python download_from_hf.py --sample` | `uv run python download_from_hf.py --sample` |
| Eval checkpoint | `python gear_sonic/eval_agent_trl.py ...` | `uv run python gear_sonic/eval_agent_trl.py ... ++use_mjlab=True ++sim_type=mjlab ...` |
| Test training sample data | `python gear_sonic/train_agent_trl.py ...` | `uv run python gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...` |
| SMPL data train | `python gear_sonic/train_agent_trl.py ...` | 同上，并补齐 robot motion path。 |

`hf download ...` 和 `cat bones_seed_smpl/... | tar ...` 不需要 mjlab override。

Sample data 训练替换命令：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=16 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

Eval checkpoint 替换命令：

```bash
uv run python gear_sonic/eval_agent_trl.py \
  +checkpoint=models/sonic_release/last.pt \
  +num_envs=1 \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
```

<a id="docssourcegetting_startedquickstartmd"></a>

## docs/source/getting_started/quickstart.md

状态：主要是 MuJoCo sim2sim / C++ deployment，不是训练迁移。

| 官方命令 | 处理方式 |
|---|---|
| `python download_from_hf.py` | 可替换为 `uv run python download_from_hf.py`。 |
| `bash install_scripts/install_mujoco_sim.sh` | 保持官方命令；它使用 `.venv_sim`，不是 SonicMJ 训练 `.venv`。 |
| `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py` | 保持官方命令；这是部署仿真循环，不是 mjlab 训练 backend。 |
| `bash deploy.sh sim` / `./deploy.sh real` | 保持官方命令；C++ deployment 不追加 `use_mjlab=True`。 |
| `python visualize_motion.py ...` | 和训练迁移无关；按 deployment 环境执行。 |

<a id="docssourcegetting_startedinstallation_deploymd"></a>

## docs/source/getting_started/installation_deploy.md

状态：部署安装说明，无训练迁移命令。

这里的 `TensorRT`、`gear_sonic_deploy/scripts/install_deps.sh`、`source scripts/setup_env.sh`、`just build`、Docker 命令都属于 C++ deployment。不要追加 `use_mjlab=True`，也不要为了训练迁移去改 CUDA/驱动/TensorRT。

<a id="docssourcegetting_startedvr_teleop_setupmd"></a>

## docs/source/getting_started/vr_teleop_setup.md

状态：VR teleop 环境说明，无训练迁移命令。

`bash install_scripts/install_pico.sh` 仍使用官方 `.venv_teleop` 路径，不属于 SonicMJ 训练 uv 环境。

<a id="docssourceuser_guidetrainingmd"></a>

## docs/source/user_guide/training.md

状态：主要训练页面，几乎所有训练/eval/export 命令都需要替换。

### Data Processing

| 官方命令 | 替换为 |
|---|---|
| `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` | `uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` |
| `python gear_sonic/data_process/filter_and_copy_bones_data.py ...` | `uv run python gear_sonic/data_process/filter_and_copy_bones_data.py ...` |
| `python gear_sonic/data_process/extract_soma_joints_from_bvh.py ...` | `uv run python gear_sonic/data_process/extract_soma_joints_from_bvh.py ...` |

### Basic / sample / full training

官方所有 `python gear_sonic/train_agent_trl.py ... +exp=manager/universal_token/all_modes/sonic_release` 替换为：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=4096 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

本地快速测试建议：

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=16 \
  headless=True \
  ++algo.config.num_learning_iterations=10 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

### Finetuning

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=4096 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

### Single-node / multi-node

把官方 `accelerate launch` 前加 `uv run`，并在脚本参数里加 `use_mjlab=True sim_type=mjlab`：

```bash
uv run accelerate launch --num_processes=8 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=4096 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

多机命令保留官方 `--multi_gpu --num_machines --machine_rank --main_process_ip --main_process_port`，同样加 `uv run` 和 mjlab overrides。

### W&B

```bash
WANDB_MODE=offline uv run python gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...
```

### Replay

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  ++replay=True \
  num_envs=4 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

### Eval metrics

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/eval_agent_trl.py \
  +checkpoint=<path_to_checkpoint.pt> \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  ++eval_callbacks=im_eval \
  ++run_eval_loop=False \
  ++num_envs=128 \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered>" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>"
```

### Render eval

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/eval_agent_trl.py \
  +checkpoint=<path_to_checkpoint.pt> \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  ++eval_callbacks=im_eval \
  ++run_eval_loop=False \
  ++num_envs=8 \
  ++manager_env.config.render_results=True \
  "++manager_env.config.save_rendering_dir=/tmp/renders" \
  ++manager_env.config.env_spacing=10.0 \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered>" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>" \
  "~manager_env/recorders=empty" "+manager_env/recorders=render"
```

说明：mjlab eval callback 已接入；视频渲染和 Isaac recorder 的完全等价性仍需按机器验证。

### ONNX export

```bash
uv run python gear_sonic/eval_agent_trl.py \
  +checkpoint=<path_to_checkpoint.pt> \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  ++num_envs=1 \
  +export_onnx_only=true \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered>" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>"
```

### SOMA / `sonic_bones_seed`

```bash
uv run accelerate launch \
  --multi_gpu --num_machines=8 --num_processes=64 \
  --machine_rank=$MACHINE_RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bones_seed \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=4096 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_filtered
```

已验证过的短训形态：

```bash
WANDB_MODE=disabled uv run accelerate launch --num_processes=1 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_bones_seed \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=32 \
  headless=True \
  ++algo.config.num_learning_iterations=20 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_medium \
  ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_uniform_medium \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  manager_env.config.terrain_type=plane
```

<a id="docssourceuser_guidetraining_datamd"></a>

## docs/source/user_guide/training_data.md

状态：数据下载说明，无训练 backend；只需要按 uv 环境安装依赖。

| 官方命令 | 替换为 |
|---|---|
| `pip install huggingface_hub` | `uv pip install huggingface_hub` |
| `huggingface-cli download bones-studio/seed ...` | 可保持不变；如果 CLI 只在 uv 环境中安装，则先 `uv run huggingface-cli ...`。 |

Python API 示例不需要改 `use_mjlab=True`。

<a id="docssourceuser_guideconfigurationmd"></a>

## docs/source/user_guide/configuration.md

状态：所有训练 recipe 都需要加 mjlab overrides。

| 官方写法 | SonicMJ 写法 |
|---|---|
| `python gear_sonic/train_agent_trl.py +exp=... num_envs=16` | `uv run python gear_sonic/train_agent_trl.py +exp=... use_mjlab=True sim_type=mjlab num_envs=16` |
| `python gear_sonic/train_agent_trl.py +exp=... ++manager_env...=...` | `uv run python gear_sonic/train_agent_trl.py +exp=... use_mjlab=True sim_type=mjlab ++manager_env...=...` |

示例，官方 flat ground recipe：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  ++manager_env.config.terrain_type=plane
```

示例，修改 reward：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  ++manager_env.rewards.tracking_anchor_pos.weight=1.0
```

<a id="docssourceuser_guidenew_embodimentsmd"></a>

## docs/source/user_guide/new_embodiments.md

状态：数据处理命令按 uv 替换；训练命令还需要 mjlab overrides，但当前 G1 以外机器人未完成同等验证。

| 官方命令 | 替换为 |
|---|---|
| `python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` | `uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py ...` |
| `python gear_sonic/data_process/filter_and_copy_bones_data.py ...` | `uv run python gear_sonic/data_process/filter_and_copy_bones_data.py ...` |
| `python download_from_hf.py --training` | `uv run python download_from_hf.py --training` |
| `python gear_sonic/data_process/extract_soma_joints_from_bvh.py ...` | `uv run python gear_sonic/data_process/extract_soma_joints_from_bvh.py ...` |
| `python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_h2 ...` | 见下方。 |

H2 示例替换形态：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_h2 \
  use_mjlab=True \
  sim_type=mjlab \
  num_envs=16 \
  headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/h2_motions> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy
```

注意：当前 SonicMJ 迁移主验证目标是 G1 `sonic_release` / `sonic_bones_seed`。H2 或新机器人训练前必须先确认 mjlab 资产、joint/body order、action dim 和 motion_lib 顺序。

<a id="docssourceuser_guidetroubleshootingmd"></a>

## docs/source/user_guide/troubleshooting.md

状态：训练相关 troubleshooting 需要改命令；部署/sim2sim troubleshooting 不需要。

| 官方位置 | 官方命令 | 替换为 |
|---|---|---|
| pip conflict | `pip install -e "gear_sonic/[training]" --upgrade` | 不建议在 SonicMJ 中使用；改用 `uv sync` 或项目内 `uv pip install ...`。 |
| Motion file path fix | `python gear_sonic/train_agent_trl.py ...` | `uv run python gear_sonic/train_agent_trl.py ... use_mjlab=True sim_type=mjlab ...` |
| Sample data | `hf download ...` | 可保持不变；如果 CLI 在 uv 环境中，使用 `uv run hf download ...`。 |

Motion path fix 替换命令：

```bash
uv run python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  use_mjlab=True \
  sim_type=mjlab \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

`run_sim_loop.py` 相关报错属于 `.venv_sim` / deployment，不属于 mjlab 训练 backend。

<a id="docssourceuser_guideteleoperationmd"></a>

## docs/source/user_guide/teleoperation.md

状态：teleop / deployment 流程，不属于 mjlab 训练迁移。

| 官方命令 | 处理方式 |
|---|---|
| `source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py` | 保持官方命令。 |
| `bash deploy.sh --input-type zmq_manager sim/real` | 保持官方命令。 |
| `source .venv_teleop/bin/activate && python gear_sonic/scripts/pico_manager_thread_server.py --manager` | 保持官方命令。 |

<a id="docssourcetutorialskeyboardmd"></a>

## docs/source/tutorials/keyboard.md

状态：C++ deployment / MuJoCo sim2sim，不属于 mjlab 训练迁移。

`run_sim_loop.py` 使用 `.venv_sim`；`bash deploy.sh --input-type keyboard ...` 使用 `gear_sonic_deploy`。都不追加 `use_mjlab=True`。

<a id="docssourcetutorialsgamepadmd"></a>

## docs/source/tutorials/gamepad.md

状态：real robot deployment，不属于 mjlab 训练迁移。

`bash deploy.sh --input-type gamepad real` 保持官方命令。

<a id="docssourcetutorialszmqmd"></a>

## docs/source/tutorials/zmq.md

状态：ZMQ streaming deployment，不属于 mjlab 训练迁移。

`run_sim_loop.py`、`deploy.sh --input-type zmq ...`、`pico_manager_thread_server.py` 都保持官方对应 `.venv_sim` / `.venv_teleop` / deployment 环境，不追加 mjlab overrides。

<a id="docssourcetutorialsmanagermd"></a>

## docs/source/tutorials/manager.md

状态：manager input deployment，不属于 mjlab 训练迁移。

`python gear_sonic/scripts/run_sim_loop.py` 和 `bash deploy.sh --input-type manager ...` 保持官方命令。

<a id="docssourcetutorialsvr_wholebody_teleopmd"></a>

## docs/source/tutorials/vr_wholebody_teleop.md

状态：VR teleop deployment，不属于 mjlab 训练迁移。

`install_pico.sh`、`run_sim_loop.py`、`./deploy.sh --input-type zmq_manager ...`、`pico_manager_thread_server.py` 都保持官方命令。

<a id="docssourcetutorialsdata_collectionmd"></a>

## docs/source/tutorials/data_collection.md

状态：数据采集 pipeline，不是 mjlab 训练 backend。

| 官方命令 | 处理方式 |
|---|---|
| `bash install_scripts/install_data_collection.sh` | 保持官方命令，使用 `.venv_data_collection`。 |
| `bash install_scripts/install_camera_server.sh` | 保持官方命令。 |
| `python -m gear_sonic.camera.composed_camera ...` | 在 camera/data collection 对应 venv 中运行，不追加 mjlab overrides。 |
| `python gear_sonic/scripts/launch_data_collection.py ...` | 保持官方命令，它会启动 deploy/sim/data exporter。 |
| `python gear_sonic/scripts/run_data_exporter.py ...` | 保持官方命令。 |
| `python gear_sonic/scripts/process_dataset.py ...` | 数据后处理，不需要 `use_mjlab=True`；若在项目 uv 环境运行，可写成 `uv run python ...`。 |

<a id="docssourcereferencestraining_codemd"></a>

## docs/source/references/training_code.md

状态：训练代码说明页，需要把示例入口理解为 mjlab backend。

| 官方描述/命令 | SonicMJ 对照 |
|---|---|
| `python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release` | `uv run python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab` |
| “Launches IsaacLab AppLauncher” | mjlab 路径不会启动 IsaacLab；`create_manager_env()` 会进入 `sonic_mj.train.create_mjlab_manager_env`。 |
| “ManagerBasedRLEnv (IsaacLab) -> ManagerEnvWrapper” | mjlab 路径为 `mjlab.envs.ManagerBasedRlEnv -> SonicMjEnvWrapper`。 |
| `python gear_sonic/eval_agent_trl.py ...` | `uv run python gear_sonic/eval_agent_trl.py ... ++use_mjlab=True ++sim_type=mjlab ...` |

Eval 示例替换：

```bash
uv run python gear_sonic/eval_agent_trl.py \
  +checkpoint=path/to/model.pt \
  +headless=True \
  ++use_mjlab=True \
  ++sim_type=mjlab \
  ++num_envs=1
```

<a id="docssourcereferencesdeployment_codemd"></a>

## docs/source/references/deployment_code.md

状态：C++ deployment 参考，不属于 mjlab 训练迁移。

`python ../gear_sonic/scripts/run_sim_loop.py` 保持官方 `.venv_sim` 用法；deployment binary 不追加 `use_mjlab=True`。

<a id="docssourcereferencesmotion_referencemd"></a>

## docs/source/references/motion_reference.md

状态：deployment motion reference，不属于 mjlab 训练迁移。

`./deploy.sh --motion-data ... sim` 和 `./deploy.sh sim` 保持官方命令。

<a id="docssourcereferencesdecoupled_wbcmd"></a>

## docs/source/references/decoupled_wbc.md

状态：Decoupled WBC 参考页，无 SonicMJ 训练入口替换。

<a id="docssourcereferencesplanner_onnxmd"></a>

## docs/source/references/planner_onnx.md

状态：ONNX planner 参考页，无 SonicMJ 训练入口替换。

<a id="docssourcereferencesobservation_configmd"></a>

## docs/source/references/observation_config.md

状态：部署 observation config 参考页，无训练命令替换。

<a id="docssourcereferencesconventionsmd"></a>

## docs/source/references/conventions.md

状态：约定说明页，无训练命令替换。

<a id="docssourcereferencesindexmd"></a>

## docs/source/references/index.md

状态：索引页，无训练命令替换。

<a id="docssourcereferencesjetpack6md"></a>

## docs/source/references/jetpack6.md

状态：Jetson/JetPack 部署说明，不属于 mjlab 训练迁移；不要因为训练迁移修改 CUDA、驱动或 JetPack。

<a id="docssourceapiindexmd"></a>

## docs/source/api/index.md

状态：API 索引页，无训练命令替换。

<a id="docssourceapiteleopmd"></a>

## docs/source/api/teleop.md

状态：teleop API 说明，无 mjlab 训练入口替换。

<a id="docssourceresourcesmd"></a>

## docs/source/resources/*.md

状态：citation / support / contributing / license 资源页，无训练命令替换。

<a id="decoupled_wbcsim2mujocoreadmemd"></a>

## decoupled_wbc/sim2mujoco/README.md

状态：Decoupled WBC sim2mujoco 工具说明，不属于 SonicMJ 训练迁移。

`pip install urdf2mjcf` 如需在项目环境内执行，可改为 `uv pip install urdf2mjcf`，但和 `use_mjlab=True` 无关。

<a id="external_dependenciesunitree_sdk2_pythonreadmemd"></a>

## external_dependencies/unitree_sdk2_python/README.md

状态：Unitree SDK 外部依赖说明，不属于 SonicMJ 训练迁移。

`pip install unitree_sdk2py` 按其外部依赖环境处理；不要装到 base。若必须装入某个项目环境，用对应环境的 pip/uv。

## 未在上面展开的 md/rst 页面

以下页面没有发现需要迁移的 SONIC 训练/eval 命令；按官方说明阅读即可：

- `docs/source/index.rst`
- `docs/source/resources/citations.md`
- `docs/source/resources/contributing.md`
- `docs/source/resources/license.md`
- `docs/source/resources/support.md`
- `external_dependencies/unitree_sdk2_python/README zh.md`
- `external_dependencies/unitree_sdk2_python/example/g1/readme.md`
- `decoupled_wbc/sim2mujoco/resources/robots/g1/README.md`

## 最终检查清单

看到官方训练命令时，至少确认这几项：

1. 命令前缀是否从 `python` / `accelerate` 改成 `uv run python` / `uv run accelerate`。
2. `train_agent_trl.py` 参数里是否有 `use_mjlab=True sim_type=mjlab`。
3. `eval_agent_trl.py` 参数里是否有 `++use_mjlab=True ++sim_type=mjlab`。
4. 是否显式传了 `motion_file` 和 `smpl_motion_file`。
5. 快速 smoke 是否加了 `manager_env.config.terrain_type=plane`。
6. 是否没有安装依赖到 base，也没有修改 CUDA、显卡驱动或系统 GPU 组件。
