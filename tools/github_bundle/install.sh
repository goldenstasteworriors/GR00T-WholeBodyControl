#!/usr/bin/env bash
# GitHub-free installation using a prepared bundle. APT/PyPI still need network.
set -euo pipefail
bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
action=${1:-help}
if [[ "$action" == help || "$action" == --help ]]; then
    echo 'Usage: bash tools/github_bundle/install.sh check|camera|robot-build|hand|pc /absolute/project/path'
    echo 'No action starts robot control or hand motion. Keep the extracted bundle in place.'
    exit 0
fi
project=${2:?Specify the existing project directory}
project=$(cd "$project" && pwd)
test -f "$project/gear_sonic/scripts/launch_data_collection.py"
test -f "$bundle/downloads/UPSTREAM.json"
arch=$(uname -m)
case "$arch" in aarch64|x86_64) ;; *) echo "Unsupported architecture: $arch"; exit 1;; esac
mkdir -p "$project/.runtime/github-bundle" "$project/.runtime/tmp"
export TMPDIR="$project/.runtime/tmp"
export UV_PYTHON_DOWNLOADS=never
export UV_CACHE_DIR="$project/.runtime/uv-cache"
runtime="$project/.runtime/github-bundle"
if [[ "$action" == check ]]; then
    (cd "$bundle" && sha256sum --quiet -c SHA256SUMS)
    echo "Bundle checksum OK; host architecture: $arch"
    exit 0
fi
if [[ ! -x "$runtime/uv/uv" ]]; then
    mkdir -p "$runtime/uv"
    tar -xzf "$bundle/downloads/uv-$arch.tar.gz" -C "$runtime/uv" --strip-components=1
fi
if [[ ! -x "$runtime/python/bin/python3.10" ]]; then
    tar -xzf "$bundle/downloads/python-$arch.tar.gz" -C "$runtime"
fi
export PATH="$runtime/uv:$PATH"
python="$runtime/python/bin/python3.10"
"$python" -c 'import sys; print(sys.version)'
venv() {
    local target="$project/$1"
    if [[ ! -e "$target" ]]; then
        uv venv "$target" --python "$python"
    fi
    "$target/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
}
case "$action" in
camera)
    sudo apt-get install -y build-essential cmake pkg-config libusb-1.0-0-dev libudev-dev libssl-dev v4l-utils
    venv .venv_camera
    uv pip install --python "$project/.venv_camera/bin/python" 'numpy==1.26.4' 'opencv-python<4.12' pyzmq msgpack msgpack-numpy tyro requests
    uv pip install --python "$project/.venv_camera/bin/python" --no-deps -e "$project/gear_sonic"
    cmake -S "$bundle/sources/librealsense" -B "$runtime/realsense-build" \
        -DSONIC_OFFLINE_ARCHIVES="$bundle/downloads" \
        -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON \
        -DBUILD_PYTHON_BINDINGS=ON -DBUILD_LEGACY_PYBACKEND=OFF \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
        -DBUILD_GLSL_EXTENSIONS=OFF -DBUILD_TOOLS=OFF -DBUILD_WITH_CUDA=OFF \
        -DIMPORT_DEPTH_CAM_FW=OFF -DCHECK_FOR_UPDATES=OFF \
        -DPYTHON_EXECUTABLE="$project/.venv_camera/bin/python" \
        -DPYTHON_INSTALL_DIR="$project/.venv_camera/lib/python3.10/site-packages" \
        -DCMAKE_INSTALL_PREFIX="$runtime/realsense-install"
    cmake --build "$runtime/realsense-build" -j "${BUILD_JOBS:-2}"
    uv pip uninstall --python "$project/.venv_camera/bin/python" pyrealsense2
    cmake --install "$runtime/realsense-build"
    "$project/.venv_camera/bin/python" -c 'import pyrealsense2 as r; print(r.__file__); devices=[d.get_info(r.camera_info.serial_number) for d in r.context().devices]; print(devices); assert devices, "SDK installed, but no camera detected; check USB permissions and device ownership"'
    ;;
robot-build)
    [[ "$arch" == aarch64 ]] || { echo 'robot-build requires ARM64 Jetson'; exit 1; }
    sudo apt-get install -y build-essential cmake pkg-config zlib1g-dev libeigen3-dev libzmq3-dev libmsgpack-dev libyaml-cpp-dev libfmt-dev libspdlog-dev libboost-program-options-dev
    if [[ ! -d "$runtime/onnxruntime-linux-aarch64-1.16.3" ]]; then
        tar -xzf "$bundle/downloads/onnxruntime-aarch64.tgz" -C "$runtime"
    fi
    export onnxruntime_ROOT="$runtime/onnxruntime-linux-aarch64-1.16.3"
    cmake -S "$project/gear_sonic_deploy" -B "$runtime/sonic-build" \
        -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$bundle/sources/googletest" \
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
        -DCMAKE_CXX_FLAGS="-I$bundle/sources/cppzmq -I$bundle/sources/json/single_include"
    cmake --build "$runtime/sonic-build" --target g1_deploy_onnx_ref -j "${BUILD_JOBS:-2}"
    ldd "$project/gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    ;;
hand)
    sudo apt-get install -y build-essential cmake libspdlog-dev libfmt-dev libboost-program-options-dev libyaml-cpp-dev
    cmake -S "$project/gear_sonic_deploy/thirdparty/unitree_sdk2" -B "$runtime/unitree-build" -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$runtime/unitree-sdk"
    cmake --build "$runtime/unitree-build" -j "${BUILD_JOBS:-2}"
    cmake --install "$runtime/unitree-build"
    cmake -S "$bundle/sources/brainco_hand_service" -B "$runtime/brainco-build" \
        -DCMAKE_PREFIX_PATH="$runtime/unitree-sdk" \
        -DCMAKE_CXX_FLAGS="-I$runtime/unitree-sdk/include -I$runtime/unitree-sdk/include/ddscxx" \
        -DCMAKE_EXE_LINKER_FLAGS="-L$runtime/unitree-sdk/lib -Wl,-rpath,$runtime/unitree-sdk/lib"
    cmake --build "$runtime/brainco-build" --target brainco_hand_server -j "${BUILD_JOBS:-2}"
    venv .venv_brainco_hand
    export CYCLONEDDS_HOME="$runtime/unitree-sdk"
    export LD_LIBRARY_PATH="$runtime/unitree-sdk/lib:${LD_LIBRARY_PATH:-}"
    uv pip install --python "$project/.venv_brainco_hand/bin/python" -r "$project/tools/brainco_hand_web/requirements.txt" -e "$project/external_dependencies/unitree_sdk2_python"
    ldd "$bundle/sources/brainco_hand_service/bin/brainco_hand_server"
    ;;
pc)
    [[ "$arch" == x86_64 ]] || { echo 'PC installation requires x86_64'; exit 1; }
    sudo apt-get install -y build-essential cmake espeak libgl1 libglib2.0-0
    printf 'lerobot @ file://%s/sources/lerobot\n' "$bundle" > "$runtime/overrides.txt"
    venv .venv_teleop
    uv pip install --python "$project/.venv_teleop/bin/python" -e "$project/gear_sonic[teleop]" 'cmake>=3.22,<4' 'pybind11==2.10.3' 'setuptools<81'
    export PATH="$project/.venv_teleop/bin:$PATH"
    export CMAKE_PREFIX_PATH
    CMAKE_PREFIX_PATH=$("$project/.venv_teleop/bin/python" -m pybind11 --cmakedir)
    uv pip install --python "$project/.venv_teleop/bin/python" --no-build-isolation -e "$project/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    venv .venv_data_collection
    uv pip install --python "$project/.venv_data_collection/bin/python" --overrides "$runtime/overrides.txt" -e "$project/gear_sonic[data_collection]"
    "$project/.venv_data_collection/bin/python" "$project/gear_sonic/scripts/launch_data_collection.py" --help
    ;;
*) echo "Unknown action: $action"; exit 2;;
esac
