#!/usr/bin/env bash
# Source the collection environment from any working directory.
COLLECTION_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COLLECTION_REPO"
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate decoupled_vla_collection
source "$CONDA_PREFIX/setup.bash"
set -u
export PYTHONPATH="$COLLECTION_REPO"

