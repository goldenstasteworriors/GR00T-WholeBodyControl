#!/usr/bin/env bash
# Navigation-only collection; keep the normal manipulation entry unchanged.
set -euo pipefail
[[ "$#" == 2 ]] || { echo "Usage: bash $0 NAVIGATION_INSTRUCTION PROJECT_NAME" >&2; exit 2; }
export NAVILA_COLLECTION=1
exec bash "$(dirname "${BASH_SOURCE[0]}")/collect_on_thinkpad.sh" default "$1" "navila_$2"
