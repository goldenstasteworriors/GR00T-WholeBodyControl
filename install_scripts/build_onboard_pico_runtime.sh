#!/usr/bin/env bash
# Build the official orin source against the robot's compatible build environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-${REPO_ROOT}/.runtime/onboard_deployment/pico-source/RoboticsService}"
BUILD_ENV="${REPO_ROOT}/.build_envs/xrobotoolkit-ubuntu20-arm64"
CONDA="${REPO_ROOT}/.runtime/miniforge3/bin/conda"
RUNTIME="${REPO_ROOT}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/runtime/ubuntu20-aarch64"
PYBIND_STAGE="${REPO_ROOT}/.runtime/onboard_deployment/xrt-pybind"
COLLECTION_ENV="${REPO_ROOT}/.runtime/miniforge3/envs/decoupled_vla_collection"

[[ "$(uname -m)" == aarch64 ]] || { echo "Run this build on the robot" >&2; exit 1; }
[[ -f "${SOURCE_DIR}/CMakeLists.txt" && -f "${BUILD_ENV}/lib/libQt6Core.so.6.7.3" ]] || {
    echo "Missing official source or Qt 6.7.3 build environment" >&2; exit 1;
}
export TMPDIR="${REPO_ROOT}/.runtime/tmp"
mkdir -p "$TMPDIR"
"$CONDA" run --no-capture-output -p "$BUILD_ENV" cmake \
    -S "$SOURCE_DIR" -B "${SOURCE_DIR}/build-ubuntu20" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$BUILD_ENV" -DBUILD_LIB_PATH="$BUILD_ENV" \
    -DQT_HOST_PATH="$BUILD_ENV"
"$CONDA" run --no-capture-output -p "$BUILD_ENV" cmake --build "${SOURCE_DIR}/build-ubuntu20" --parallel 2
"$CONDA" run --no-capture-output -p "$BUILD_ENV" cmake \
    -S "${SOURCE_DIR}/PXREARobotSDK" -B "${SOURCE_DIR}/PXREARobotSDK/build-ubuntu20" -DCMAKE_BUILD_TYPE=Release
"$CONDA" run --no-capture-output -p "$BUILD_ENV" cmake --build "${SOURCE_DIR}/PXREARobotSDK/build-ubuntu20" --parallel 2

mkdir -p "$RUNTIME" "$PYBIND_STAGE"
cp -a "${SOURCE_DIR}/bin/." "$RUNTIME/"
# Stage the Python binding separately; leave the bundled reference SDK unchanged.
for name in setup.py CMakeLists.txt bindings include; do
    cp -a "${REPO_ROOT}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/${name}" "$PYBIND_STAGE/"
done
mkdir -p "${PYBIND_STAGE}/lib/aarch64"
cp "${SOURCE_DIR}/PXREARobotSDK/build-ubuntu20/libPXREARobotSDK.so" "${PYBIND_STAGE}/lib/aarch64/"
mkdir -p "${COLLECTION_ENV}/etc/conda/activate.d" "${COLLECTION_ENV}/etc/conda/deactivate.d"
cat > "${COLLECTION_ENV}/etc/conda/activate.d/onboard_xrt.sh" <<EOF
export _ONBOARD_XRT_OLD_LD_LIBRARY_PATH=\${LD_LIBRARY_PATH-}
export LD_LIBRARY_PATH='${PYBIND_STAGE}/lib/aarch64:${BUILD_ENV}/lib':\${LD_LIBRARY_PATH-}
EOF
cat > "${COLLECTION_ENV}/etc/conda/deactivate.d/onboard_xrt.sh" <<'EOF'
export LD_LIBRARY_PATH="${_ONBOARD_XRT_OLD_LD_LIBRARY_PATH-}"
unset _ONBOARD_XRT_OLD_LD_LIBRARY_PATH
EOF
"$CONDA" run --no-capture-output -p "$COLLECTION_ENV" python -m pip install \
    --no-build-isolation --no-deps "$PYBIND_STAGE"
echo "Built robot PICO service and Python binding; no system libraries changed"
