# PKU H2 SONIC Training and Validation Plan

## 0. Scope and GPU allocation

- PKU project path: `/home/nvme02/GR00T/GR00T`
- PKU H2 retargeted CSV data: `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/csv`
- PKU H2 converted motion-lib data: `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib`
- PKU H2 filtered training data: `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered`
- Environment entry: `./.tools/uv/uv run ...`
- Maximum GPU usage: only GPUs `0,1,2,3,4,5`
- Main training: GPUs `0,1,2,3` with 4 accelerate processes
- Validation/eval: GPUs `4,5` with 2 accelerate processes
- Do not modify CUDA, GPU driver, or system GPU stack.
- Do not overwrite unrelated remote worktree changes.

## 1. Pre-flight checks

Run on PKU:

```bash
cd /home/nvme02/GR00T/GR00T

./.tools/uv/uv --version
./.tools/uv/uv run python - <<'PY'
import sys, torch, mjlab
print(sys.executable)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(mjlab.__file__)
PY

find /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/csv -name '*.csv' | wc -l
find /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered -name '*.pkl' | wc -l || true
du -sh /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '1,6p'
```

Expected:

- H2 source CSV count is non-zero.
- H2 filtered PKL count is non-zero after the data preparation step.
- GPUs `0-5` are available or lightly used.
- `torch.cuda.is_available()` is `True`.

## 2. Data preparation

The H2 retargeted source data is the Bones-SEED-style CSV tree under
`/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/csv`. Convert it with the
SONIC motion-lib converter in H2 mode, then apply the same keyword-based filter
used for the original Bones data.

```bash
cd /home/nvme02/GR00T/GR00T

./.tools/uv/uv run python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/csv \
  --output /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib \
  --fps 30 \
  --fps_source 120 \
  --individual \
  --num_workers 16 \
  --robot h2

./.tools/uv/uv run python gear_sonic/data_process/filter_and_copy_bones_data.py \
  --source /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib \
  --dest /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered \
  --workers 16

find /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib -name '*.pkl' | wc -l
find /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered -name '*.pkl' | wc -l
du -sh /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib \
       /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered
```

## 3. Main 4-GPU warm-start training

This starts H2 training from the official G1 `sonic_release/last.pt` by shape-aware policy initialization. This is not full resume; optimizer, scheduler, env state, and global step are not loaded.

```bash
cd /home/nvme02/GR00T/GR00T

mkdir -p logs_pku

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_MODE=offline \
nohup ./.tools/uv/uv run python -m accelerate.commands.launch \
  --multi_gpu --num_processes=4 --num_machines=1 --main_process_port=29540 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_h2 \
  use_mjlab=True sim_type=mjlab num_envs=128 headless=True \
  ++algo.trl.bf16=False ++algo.trl.fp16=False \
  ++algo.config.num_learning_iterations=1000 \
  ++algo.config.num_mini_batches=1 \
  ++algo.config.num_learning_epochs=1 \
  ++algo.config.save_interval=50 \
  ++algo.config.pretrained_model.path=sonic_release/last.pt \
  ++algo.config.pretrained_model.state_dict_key=policy_state_dict \
  ++algo.config.pretrained_model.shape_aware=True \
  ++algo.config.pretrained_model.source_robot=g1 \
  ++algo.config.pretrained_model.target_robot=h2 \
  ++algo.config.pretrained_model.source_action_dim=29 \
  ++algo.config.pretrained_model.target_action_dim=31 \
  ++manager_env.config.terrain_type=plane \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  > logs_pku/h2_4gpu_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs_pku/h2_4gpu_train.pid
cat logs_pku/h2_4gpu_train.pid
```

Monitor:

```bash
cd /home/nvme02/GR00T/GR00T
tail -f logs_pku/h2_4gpu_train_*.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '1,6p'
```

Find latest run directory:

```bash
cd /home/nvme02/GR00T/GR00T
ls -td logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-* | head -5
```

Expected early checks:

- motion files are loaded from `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered`
- `Active Action Terms (shape: 31)`
- policy obs shape `(990,)`
- critic obs shape `(1907,)`
- shape-aware report: loaded `47`, skipped shape `8`, missing `0`, unexpected `0`
- no `non_finite_*` termination in training logs

## 4. 2-GPU validation/evaluation

Use GPUs `4,5`. Run validation after the 4-GPU training job writes its first checkpoint.
With `++algo.config.save_interval=50`, the first usable checkpoint should be:

```bash
CKPT=/home/nvme02/GR00T/GR00T/<NEW_RUN_DIR>/last.pt
```

Before running validation, confirm it exists:

```bash
test -f "${CKPT}" && ls -lh "${CKPT}"
```

2-GPU eval command:

```bash
cd /home/nvme02/GR00T/GR00T

CUDA_VISIBLE_DEVICES=4,5 \
WANDB_MODE=offline \
./.tools/uv/uv run python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --num_machines=1 --main_process_port=29545 \
  gear_sonic/eval_agent_trl.py \
  +checkpoint=${CKPT} \
  +use_mjlab=True +sim_type=mjlab +num_envs=64 +headless=True \
  +max_render_steps=500 \
  ++manager_env.config.terrain_type=plane \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

Expected:

- checkpoint loads without shape mismatch
- H2 env builds with action dim `31`
- eval loop exits after `max_render_steps=500`
- no non-finite state crash

## 5. 2-GPU subset training validation

If the full 129785-motion training run spends too long in first-time motion preprocessing,
keep the 4-GPU full run alive and use GPUs `4,5` to validate the actual training loop on a
small symlink subset built from the same correct filtered H2 data.

```bash
cd /home/nvme02/GR00T/GR00T

SUBSET=/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered_4096
rm -rf "${SUBSET}"
mkdir -p "${SUBSET}/subset"
find /home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered \
  -name '*.pkl' | sort | head -4096 | while read -r f; do
    ln -s "$f" "${SUBSET}/subset/$(basename "$f")"
  done
find "${SUBSET}" -name '*.pkl' | wc -l

CUDA_VISIBLE_DEVICES=4,5 \
WANDB_MODE=offline \
nohup ./.tools/uv/uv run python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --num_machines=1 --main_process_port=29545 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_h2 \
  use_mjlab=True sim_type=mjlab num_envs=64 headless=True \
  ++algo.trl.bf16=False ++algo.trl.fp16=False \
  ++algo.config.num_learning_iterations=20 \
  ++algo.config.num_mini_batches=1 \
  ++algo.config.num_learning_epochs=1 \
  ++algo.config.save_interval=10 \
  ++algo.config.pretrained_model.path=sonic_release/last.pt \
  ++algo.config.pretrained_model.state_dict_key=policy_state_dict \
  ++algo.config.pretrained_model.shape_aware=True \
  ++algo.config.pretrained_model.source_robot=g1 \
  ++algo.config.pretrained_model.target_robot=h2 \
  ++algo.config.pretrained_model.source_action_dim=29 \
  ++algo.config.pretrained_model.target_action_dim=31 \
  ++manager_env.config.terrain_type=plane \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="${SUBSET}" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  > logs_pku/h2_2gpu_subset_train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Expected:

- `Loaded 4096 motion files`
- H2 env action dim is `31`
- official G1 checkpoint is loaded through shape-aware initialization
- training reaches at least several iterations without traceback or non-finite termination

## 6. Resume main training from checkpoint

Use this only for continuing an H2 run. Do not include `pretrained_model.*` overrides when doing true resume.

```bash
cd /home/nvme02/GR00T/GR00T

RUN_DIR=/home/nvme02/GR00T/GR00T/logs_rl/TRL_H2_Track/manager/universal_token/all_modes/<RUN_NAME>
CKPT=${RUN_DIR}/last.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_MODE=offline \
nohup ./.tools/uv/uv run python -m accelerate.commands.launch \
  --multi_gpu --num_processes=4 --num_machines=1 --main_process_port=29540 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_h2 \
  resume=True \
  +checkpoint=${CKPT} \
  use_mjlab=True sim_type=mjlab num_envs=128 headless=True \
  ++algo.trl.bf16=False ++algo.trl.fp16=False \
  ++algo.config.num_learning_iterations=3000 \
  ++algo.config.num_mini_batches=1 \
  ++algo.config.num_learning_epochs=1 \
  ++algo.config.save_interval=50 \
  ++manager_env.config.terrain_type=plane \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  > logs_pku/h2_4gpu_resume_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs_pku/h2_4gpu_resume.pid
cat logs_pku/h2_4gpu_resume.pid
```

## 7. Stop commands

Stop main training if needed:

```bash
cd /home/nvme02/GR00T/GR00T
kill "$(cat logs_pku/h2_4gpu_train.pid)"
```

If accelerate child processes remain:

```bash
ps -eo pid,cmd | grep -E 'train_agent_trl|accelerate.commands.launch' | grep -v grep
```

Kill only the PIDs belonging to this H2 run. Do not kill unrelated user processes.

## 8. Reporting checklist

Record these after each run:

- run directory
- command log path
- checkpoint path
- number of iterations completed
- total timesteps
- shape-aware loading report
- mean reward trend
- `Env/Metrics/motion/error_anchor_pos`
- all `non_finite_*` termination metrics
- GPU allocation actually used
