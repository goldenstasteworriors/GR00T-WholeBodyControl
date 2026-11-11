#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
if [[ -f "$root/.runtime/brainco-web-native/install/lib/libddsc.so" ]]; then
    export CYCLONEDDS_HOME="$root/.runtime/brainco-web-native/install"
else
    export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/usr/local}"
fi
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$root"
exec "$root/.venv_brainco_hand/bin/python" -m gear_sonic.scripts.brainco_hand_web "$@"
