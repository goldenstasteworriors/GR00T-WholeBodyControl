#!/usr/bin/env bash
# Project-local CycloneDDS development files and the BrainCo web environment.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
native="$root/.runtime/brainco-web-native"
archive=${1:-"$root/.runtime/brainco-diagnostics/cyclonedds-0.10.2.tar.gz"}
index=${BRAINCO_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
mkdir -p "$native" "$root/.runtime/tmp" "$root/.runtime/uv-cache" "$(dirname "$archive")"
export TMPDIR="$root/.runtime/tmp"
export UV_CACHE_DIR="$root/.runtime/uv-cache"
if [[ ! -f "$native/install/include/idl/retcode.h" || ! -f "$native/install/lib/libddsc.so" ]]; then
    if [[ ! -f "$archive" ]]; then
        curl -fLsS --retry 3 https://codeload.github.com/eclipse-cyclonedds/cyclonedds/tar.gz/refs/tags/0.10.2 -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    printf '%s  %s\n' bc84e137e0c8a055b8cd97fbeafec94e36de1b0c2e88800896a82384fd867ae5 "$archive" | sha256sum -c -
    tar -xzf "$archive" -C "$native"
    cmake -S "$native/cyclonedds-0.10.2" -B "$native/build" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$native/install" \
        -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DBUILD_DDSPERF=OFF -DBUILD_IDLC=ON -DENABLE_ICEORYX=OFF \
        -DENABLE_SECURITY=OFF -DENABLE_SSL=OFF
    cmake --build "$native/build" -j "${BUILD_JOBS:-2}"
    cmake --install "$native/build"
fi
export CYCLONEDDS_HOME="$native/install"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -e "$root/.venv_brainco_hand" ]]; then
    uv venv "$root/.venv_brainco_hand" --python 3.10
fi
uv --no-config pip install --python "$root/.venv_brainco_hand/bin/python" --default-index "$index" setuptools wheel
uv --no-config pip install --python "$root/.venv_brainco_hand/bin/python" \
    --default-index "$index" --no-build-isolation \
    -r "$here/requirements.txt" -e "$root/external_dependencies/unitree_sdk2_python"
"$root/.venv_brainco_hand/bin/python" -c 'import flask, cyclonedds, unitree_sdk2py; print("BrainCo web environment imports OK")'
