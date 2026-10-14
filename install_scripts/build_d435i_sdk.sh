#!/usr/bin/env bash
# Optional SDK build, NOT needed by sonic_d435i_camera.service (RGB over UVC).
# SDK import was verified; RSUSB capture timed out on the onboard D435i.
# Build the official SDK against this robot's existing system libraries.
# Requires the project's .venv_camera (Python 3.10), uv, CMake, C++ and libusb-dev.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TMPDIR="$ROOT/.runtime/tmp" UV_CACHE_DIR="$ROOT/.runtime/uv-cache"
export UV_NO_PROGRESS=1
mkdir -p "$TMPDIR" .runtime/src .runtime/wheels
PYTHON="$ROOT/.venv_camera/bin/python"
UV="$ROOT/.runtime/bin/uv"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYTHON_LIBRARY="$("$PYTHON" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY"))')"
SDK="$ROOT/.runtime/src/librealsense"
REVISION=e196cefa896e312d79c2df400c7623aa1e9c62ac
if [[ ! -d "$SDK" ]]; then
    git clone --depth 1 --branch v2.55.1 git@github.com:realsenseai/librealsense.git "$SDK"
fi
[[ "$(git -C "$SDK" rev-parse HEAD)" == "$REVISION" ]] || {
    echo "Unexpected SDK revision; refusing to overwrite existing source." >&2
    exit 1
}
[[ -z "$(git -C "$SDK" status --porcelain --untracked-files=no)" ]] || {
    echo "SDK source has tracked modifications; refusing to build." >&2
    exit 1
}
"$UV" pip install --python "$PYTHON" setuptools wheel
BUILD="$ROOT/.runtime/build-realsense"
cmake -S "$SDK" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/.venv_camera" \
    -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE="$PYTHON" -DPython_EXECUTABLE="$PYTHON" \
    -DPython_INCLUDE_DIR="$PYTHON_INCLUDE" -DPython_LIBRARY="$PYTHON_LIBRARY" \
    -DPYTHON_INSTALL_DIR="$ROOT/.venv_camera/lib/python3.10/site-packages" \
    -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_DDS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_GLSL_EXTENSIONS=OFF \
    -DBUILD_TOOLS=OFF -DBUILD_LEGACY_PYBACKEND=OFF -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_UNIT_TESTS=OFF -DIMPORT_DEPTH_CAM_FW=OFF -DCHECK_FOR_UPDATES=OFF
nice -n 10 cmake --build "$BUILD" --target pyrealsense2 --parallel "${BUILD_JOBS:-4}"

# Use the upstream wheel packaging, keeping generated files out of its Git source.
STAGE="$ROOT/.runtime/realsense-wheel"
mkdir -p "$STAGE/pyrealsense2"
cp "$SDK/wrappers/python/setup.py" "$SDK/wrappers/python/README.rst" "$STAGE/"
cp -a "$SDK/wrappers/python/examples" "$STAGE/"
cp "$SDK/wrappers/python/pyrealsense2/__init__.py" "$STAGE/pyrealsense2/"
cp -L "$BUILD/Release/pyrealsense2.cpython-310-"*.so "$STAGE/pyrealsense2/"
"$PYTHON" "$SDK/wrappers/python/find_librs_version.py" "$SDK" "$STAGE/pyrealsense2"
(cd "$STAGE" && "$PYTHON" setup.py bdist_wheel --dist-dir "$ROOT/.runtime/wheels")
"$UV" pip install --python "$PYTHON" --reinstall \
    "$ROOT"/.runtime/wheels/pyrealsense2-2.55.1-cp310-cp310-*.whl
"$PYTHON" -c 'import pyrealsense2 as rs; print("RealSense SDK import OK:", rs.__file__)'
sha256sum "$ROOT"/.runtime/wheels/pyrealsense2-2.55.1-cp310-cp310-*.whl
