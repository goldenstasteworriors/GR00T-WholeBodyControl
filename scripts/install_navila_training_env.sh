#!/usr/bin/env bash
# Official NaVILA recipe, scoped to the new project environment:
# https://github.com/AnjieCheng/NaVILA/blob/main/environment_setup.sh
# The optional system/Conda CUDA-toolkit install is intentionally omitted.
set -euo pipefail
export NAVILA_JOB_ROOT="${NAVILA_JOB_ROOT:-/nix/ykj/navila_two_tasks_20261109}"
NAVILA_TOOLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$NAVILA_JOB_ROOT/cache/huggingface"
export XDG_CACHE_HOME="$NAVILA_JOB_ROOT/cache/xdg"
export TMPDIR="$NAVILA_JOB_ROOT/cache/tmp"
export PIP_CACHE_DIR="$NAVILA_JOB_ROOT/cache/pip"
export DS_BUILD_OPS=0
export CUDA_VISIBLE_DEVICES="${NAVILA_TRAIN_GPU:-2}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
NAVILA_PYTHON="$NAVILA_JOB_ROOT/env/bin/python"
if [[ ! -d "$NAVILA_JOB_ROOT/NaVILA/.git" ]]; then
  git clone --depth 1 https://github.com/AnjieCheng/NaVILA.git "$NAVILA_JOB_ROOT/NaVILA"
  git -C "$NAVILA_JOB_ROOT/NaVILA" checkout 76b98f233dd0fff05dfcd69435eec6740febff9d
fi
cd "$NAVILA_JOB_ROOT/NaVILA"
"$NAVILA_PYTHON" -m pip install --upgrade pip
"$NAVILA_PYTHON" -m pip install -e '.[train]'
"$NAVILA_PYTHON" -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
NAVILA_SITE_PACKAGES="$("$NAVILA_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
cp -r ./llava/train/transformers_replace/. "$NAVILA_SITE_PACKAGES/transformers/"
cp -r ./llava/train/deepspeed_replace/. "$NAVILA_SITE_PACKAGES/deepspeed/"
"$NAVILA_PYTHON" -c 'import torch,flash_attn,transformers,peft,deepspeed; print(torch.__version__,flash_attn.__version__,transformers.__version__,peft.__version__,deepspeed.__version__)'
mkdir -p "$NAVILA_JOB_ROOT/models"
"$NAVILA_PYTHON" "$NAVILA_TOOLS_ROOT/gear_sonic/scripts/download_navila_model.py" \
  --destination "$NAVILA_JOB_ROOT/models/navila-llama3-8b-8f"
