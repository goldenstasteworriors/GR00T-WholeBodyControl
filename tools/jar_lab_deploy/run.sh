#!/usr/bin/env bash
set -euo pipefail
deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deploy_root="$(cd -- "$deploy_dir/../.." && pwd)"
export TMPDIR="$deploy_root/runtime/tmp"
export XDG_CACHE_HOME="$deploy_root/runtime/cache"
export PYTHONDONTWRITEBYTECODE=1
export CYCLONEDDS_HOME="$deploy_root/runtime/cyclonedds"
command_name="${1:-doctor}"
if [ "$#" -gt 0 ]; then shift; fi
case "$command_name" in
    install) exec bash "$deploy_dir/install.sh" "$@" ;;
    setup-operator) exec "$deploy_root/.venv-download/bin/python" "$deploy_dir/setup_operator.py" --root "$deploy_root" "$@" ;;
    operator) exec bash "$deploy_dir/start_operator.sh" "$@" ;;
    doctor|benchmark)
        exec "$deploy_root/.venv/bin/python" "$deploy_dir/$command_name.py" --root "$deploy_root" "$@" ;;
    start-policy|stop-policy|start-sync|stop-sync|status)
        exec "$deploy_root/.venv/bin/python" "$deploy_dir/manage.py" "$command_name" --root "$deploy_root" "$@" ;;
    read-only)
        exec "$deploy_root/.venv/bin/python" "$deploy_dir/run_robot.py" "$@" ;;
    live)
        exec "$deploy_root/.venv/bin/python" "$deploy_dir/run_robot.py" --live "$@" ;;
    *) echo "Use setup-operator, operator, doctor, benchmark, start-policy, stop-policy, start-sync, stop-sync, status, read-only or live" >&2; exit 2 ;;
esac
