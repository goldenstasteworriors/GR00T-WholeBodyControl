#!/usr/bin/env bash
set -euo pipefail
deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deploy_root="$(cd -- "$deploy_dir/../.." && pwd)"
export TMPDIR="$deploy_root/runtime/tmp"
export UV_CACHE_DIR="$deploy_root/runtime/uv-cache"
export XDG_CACHE_HOME="$deploy_root/runtime/cache"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$deploy_root/reports"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
if [ ! -x "$deploy_root/.venv/bin/python" ]; then
    "$uv_bin" venv --python /usr/bin/python3 "$deploy_root/.venv"
fi
python_bin="$deploy_root/.venv/bin/python"
# Both source archives arrive through ModelScope, with their original licenses.
tar -xzf "$deploy_dir/vendor/cyclonedds-0.10.5.tar.gz" -C "$deploy_root/runtime"
cmake -S "$deploy_root/runtime/cyclonedds-0.10.5" -B "$deploy_root/runtime/cyclonedds-build" \
    -DCMAKE_INSTALL_PREFIX="$deploy_root/runtime/cyclonedds" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build "$deploy_root/runtime/cyclonedds-build" --target install -j 4
export CYCLONEDDS_HOME="$deploy_root/runtime/cyclonedds"
"$uv_bin" pip install --python "$python_bin" -r "$deploy_dir/requirements.txt"
tar -xzf "$deploy_dir/vendor/unitree_sdk2_python.tar.gz" -C "$deploy_root/runtime"
"$uv_bin" pip install --python "$python_bin" --no-deps "$deploy_root/runtime/external_dependencies/unitree_sdk2_python"
"$python_bin" "$deploy_dir/doctor.py" --root "$deploy_root"
"$uv_bin" pip freeze --python "$python_bin" > "$deploy_root/reports/installed-requirements.txt"
