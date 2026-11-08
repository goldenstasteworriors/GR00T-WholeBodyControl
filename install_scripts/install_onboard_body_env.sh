#!/usr/bin/env bash
# Install only project-local Python packages; reuse the robot's CycloneDDS.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export TMPDIR="$REPO/.runtime/tmp"
export UV_CACHE_DIR="$REPO/.runtime/uv-cache"
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
UV=/home/unitree/GR00T-WholeBodyControl/.runtime/bin/uv
PYTHON=/home/unitree/GR00T-WholeBodyControl/.runtime/bin/python3.10
if [[ ! -x .venv_body/bin/python ]]; then
    "$UV" venv --python "$PYTHON" .venv_body
fi
offline=()
if [[ -d .runtime/onboard_body_wheels ]]; then
    offline=(--offline --find-links "$REPO/.runtime/onboard_body_wheels")
fi
"$UV" pip install --python .venv_body/bin/python "${offline[@]}" setuptools wheel
"$UV" pip install --python .venv_body/bin/python "${offline[@]}" --no-build-isolation \
    numpy==1.26.4 PyYAML msgpack onnxruntime opencv-python==4.11.0.86 cyclonedds==0.10.2
# The upstream SDK uses editable installation so its bundled CRC .so files
# remain available. An ordinary wheel omits those non-Python package files.
"$UV" pip install --python .venv_body/bin/python "${offline[@]}" --no-build-isolation \
    --no-deps --editable external_dependencies/unitree_sdk2_python
"$UV" pip freeze --python .venv_body/bin/python > .runtime/onboard_body_requirements.txt
.venv_body/bin/python -c 'import onnxruntime, numpy, yaml, msgpack; from unitree_sdk2py.utils.crc import CRC; CRC(); print("Body runtime and ARM64 CRC ready")'
