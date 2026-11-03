#!/usr/bin/env bash
# Read-only output viewer; no --execute argument exists in preview.py.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/thinkpad_collection_env.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python "$REPO/deploy/navila/preview.py" "$@"
