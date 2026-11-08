#!/usr/bin/env bash
# NaVILA 8B, frozen vision/projector, language-model BF16 LoRA, one visible GPU.
set -euo pipefail
NAVILA_JOB_ROOT="${NAVILA_JOB_ROOT:-/nix/ykj/navila_two_tasks_20261109}"
NAVILA_TRAIN_GPU="${NAVILA_TRAIN_GPU:-2}"
NAVILA_RUN_NAME="${NAVILA_RUN_NAME:-$(date +%Y%m%d)_navila_two_tasks_bf16_lora_gpu${NAVILA_TRAIN_GPU}}"
TOOLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$NAVILA_TRAIN_GPU"
export HF_HOME="$NAVILA_JOB_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$NAVILA_JOB_ROOT/cache/xdg"
export TMPDIR="$NAVILA_JOB_ROOT/cache/tmp"
export PIP_CACHE_DIR="$NAVILA_JOB_ROOT/cache/pip"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export RUN_NAME="$NAVILA_RUN_NAME"
export OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$NAVILA_JOB_ROOT/runs"
cd "$NAVILA_JOB_ROOT/NaVILA"
exec "$NAVILA_JOB_ROOT/env/bin/torchrun" --standalone --nnodes=1 --nproc_per_node=1 \
  "$TOOLS_ROOT/gear_sonic/scripts/train_navila_two_tasks.py" \
  --navila-repo "$NAVILA_JOB_ROOT/NaVILA" \
  --lab-dataset "$NAVILA_JOB_ROOT/navila_two_tasks_weak_20261109" \
  --model_name_or_path "$NAVILA_JOB_ROOT/models/navila-llama3-8b-8f" \
  --version llama_3 --seed 20261109 --data_seed 20261109 \
  --data_mixture navila_lab_train --eval_data_mixture navila_lab_validation \
  --vision_tower google/siglip-so400m-patch14-384 --mm_projector mlp_downsample \
  --mm_vision_select_feature cls_patch --mm_vision_select_layer -2 \
  --num_video_frames 8 --image_aspect_ratio resize \
  --mm_use_im_start_end False --mm_use_im_patch_token False \
  --lora_enable True --lora_llm True --lora_vt False --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --tune_vision_tower False --tune_mm_projector False --tune_language_model True \
  --bf16 True --tf32 True --bits 16 --model_dtype torch.bfloat16 \
  --output_dir "$NAVILA_JOB_ROOT/runs/$NAVILA_RUN_NAME" --run_name "$NAVILA_RUN_NAME" \
  --num_train_epochs 3 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 --do_eval True --evaluation_strategy epoch --save_strategy epoch \
  --save_total_limit 3 --load_best_model_at_end True --metric_for_best_model loss --greater_is_better False \
  --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.05 --lr_scheduler_type cosine \
  --model_max_length 4096 --gradient_checkpointing True --longvila_sampler False \
  --dataloader_num_workers 2 --lazy_preprocess True --logging_steps 1 --report_to none "$@"
