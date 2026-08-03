#!/usr/bin/env bash
# install_decoupled_vla_collection.sh
#
# Non-Docker environment for decoupled WBC + Sonic VLA data collection.
# Creates a conda env with:
#   - RoboStack ROS 2 Humble / rclpy
#   - decoupled_wbc[full]
#   - gear_sonic[data_collection,teleop,camera,sim]
#   - unitree_sdk2_python
#   - XRoboToolkit SDK + isaacteleop[cloudxr]
#
# Usage:
#   bash install_scripts/install_decoupled_vla_collection.sh
#   bash install_scripts/install_decoupled_vla_collection.sh my_env_name
#
# This script does not modify CUDA, GPU drivers, or install Python packages into
# the base conda environment.

set -euo pipefail

ENV_NAME="${1:-decoupled_vla_collection}"
PY_VERSION="3.10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARCH="$(uname -m)"
TMPDIR="$REPO_ROOT/.tmp/decoupled_vla_collection"
mkdir -p "$TMPDIR"
export TMPDIR
# Editable dependencies may contain large Git LFS datasets that are not needed
# for collection runtime. Avoid downloading them while pip clones pinned repos.
export GIT_LFS_SKIP_SMUDGE=1
PIP_CONSTRAINT_FILE="$TMPDIR/pip_constraints.txt"
cat > "$PIP_CONSTRAINT_FILE" <<'EOF'
torch==2.6.0
torchvision==0.21.0
pin==2.7.0
rerun-sdk==0.21.0
EOF

pip_install() {
    PIP_CONFIG_FILE=/dev/null python -m pip install -c "$PIP_CONSTRAINT_FILE" "$@"
}

echo "[OK] Repository: $REPO_ROOT"
echo "[OK] Architecture: $ARCH"
echo "[OK] Conda env: $ENV_NAME"
echo "[OK] TMPDIR: $TMPDIR"

if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda is not on PATH."
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[OK] Reusing existing conda env: $ENV_NAME"
else
    echo "[INFO] Creating conda env '$ENV_NAME' with Python $PY_VERSION..."
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

set +u
conda activate "$ENV_NAME"
set -u

echo "[INFO] Configuring conda channels in env..."
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults || true

echo "[INFO] Checking ROS 2 Humble / rclpy..."
if python - <<'PY' >/dev/null 2>&1
import rclpy
PY
then
    echo "[OK] rclpy already imports; skipping ROS install"
else
    echo "[INFO] Installing ROS 2 Humble Desktop from RoboStack..."
    # RoboStack activation/deactivation hooks reference variables that may not
    # exist yet, so conda operations must run with nounset temporarily relaxed.
    set +u
    if command -v mamba &>/dev/null; then
        mamba install -y ros-humble-desktop
    else
        conda install -y ros-humble-desktop
    fi
    set -u
fi

# RoboStack's setup.bash may read unset AMENT_* variables. Temporarily relax
# nounset so this script can keep using set -u for the rest of the install.
set +u
# shellcheck disable=SC1091
source "$CONDA_PREFIX/setup.bash"
set -u

cd "$REPO_ROOT"

echo "[INFO] Upgrading pip tooling inside $ENV_NAME..."
PIP_CONFIG_FILE=/dev/null python -m pip install -U pip setuptools wheel

echo "[INFO] Installing editable project packages..."
GIT_LFS_SKIP_SMUDGE=1 pip_install \
    -e "gear_sonic[data_collection,teleop,camera,sim]"

echo "[INFO] Installing decoupled_wbc[full] dependencies..."
pip_install tomli
DECOUPLED_REQ_FILE="$(mktemp)"
python - <<'PY' > "$DECOUPLED_REQ_FILE"
from pathlib import Path
import tomli

config = tomli.loads(Path("decoupled_wbc/pyproject.toml").read_text())
for requirement in config["project"]["dependencies"]:
    print(requirement)
for requirement in config["project"]["optional-dependencies"]["full"]:
    print(requirement)
PY
GIT_LFS_SKIP_SMUDGE=1 pip_install -r "$DECOUPLED_REQ_FILE"
rm -f "$DECOUPLED_REQ_FILE"

echo "[INFO] Adding repository root to Python path for decoupled_wbc imports..."
python - <<PY
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
pth_path = site_packages / "gr00t_wholebodycontrol_repo.pth"
pth_path.write_text("$REPO_ROOT\n")
print(f"[OK] Wrote {pth_path}")
PY

echo "[INFO] Installing Unitree Python SDK..."
if [ "$ARCH" = "aarch64" ]; then
    CDDS_DIR="$HOME/cyclonedds"
    CDDS_PREFIX="${CYCLONEDDS_HOME:-$CDDS_DIR/install}"
    if [ ! -f "$CDDS_PREFIX/lib/libddsc.so" ]; then
        echo "[INFO] Building CycloneDDS releases/0.10.x -> $CDDS_PREFIX ..."
        if [ ! -d "$CDDS_DIR/.git" ]; then
            git clone -b releases/0.10.x --depth 1 \
                https://github.com/eclipse-cyclonedds/cyclonedds.git "$CDDS_DIR"
        fi
        cmake -S "$CDDS_DIR" -B "$CDDS_DIR/build" \
            -DCMAKE_INSTALL_PREFIX="$CDDS_PREFIX" \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_TESTING=OFF
        cmake --build "$CDDS_DIR/build" -j"$(nproc)"
        cmake --install "$CDDS_DIR/build"
    else
        echo "[OK] CycloneDDS already present at $CDDS_PREFIX"
    fi
    export CYCLONEDDS_HOME="$CDDS_PREFIX"
fi
pip_install -e external_dependencies/unitree_sdk2_python

echo "[INFO] Installing XRoboToolkit SDK..."
pip_install cmake pybind11
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
XRT_DIR="$REPO_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
if [ "$ARCH" = "aarch64" ] && [ ! -f "$XRT_DIR/lib/aarch64/libPXREARobotSDK.so" ]; then
    echo "[INFO] Building PXREARobotSDK for aarch64..."
    XRT_TMP="$XRT_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]; then
        git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
            "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XRT_DIR/lib/aarch64" "$XRT_DIR/include/aarch64"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
        "$XRT_DIR/include/aarch64/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
        "$XRT_DIR/include/aarch64/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
        "$XRT_DIR/lib/aarch64/"
    rm -rf "$XRT_TMP"
fi
pip_install --no-build-isolation \
    -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/

echo "[INFO] Installing isaacteleop[cloudxr] from pypi.nvidia.com..."
PIP_CONFIG_FILE=/dev/null python -m pip install \
    -c "$PIP_CONSTRAINT_FILE" \
    'isaacteleop[cloudxr]~=1.3.0' \
    --pre \
    --extra-index-url https://pypi.nvidia.com

if [ ! -f "$HOME/cloudxr.env" ]; then
    echo "NV_DEVICE_PROFILE=Quest3" > "$HOME/cloudxr.env"
    echo "[OK] Seeded $HOME/cloudxr.env with NV_DEVICE_PROFILE=Quest3"
else
    echo "[OK] $HOME/cloudxr.env already exists"
fi

echo "[INFO] Verifying imports..."
python - <<'PY'
import rclpy
import decoupled_wbc
import gear_sonic
import zmq
print("[OK] rclpy, decoupled_wbc, gear_sonic, zmq imported")
PY

cat <<EOF

Setup complete.

Activate the environment with:

  conda activate $ENV_NAME
  source "\$CONDA_PREFIX/setup.bash"

Launch decoupled VLA collection with:

  python gear_sonic/scripts/launch_decoupled_vla_collection.py --help

If you want voice feedback, install the system espeak package separately:

  sudo apt-get install espeak
EOF
