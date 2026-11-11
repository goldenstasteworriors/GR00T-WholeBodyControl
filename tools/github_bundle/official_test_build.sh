#!/usr/bin/env bash
# Build the unmodified Unitree explicit-port reference service in its own bundle.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash build.sh  (from the extracted official test bundle)'
    echo 'Builds only; does not open serial ports or send commands.'
    exit 0
fi
case "$(uname -m)" in
    aarch64) sdk_zip=linux-arm64.zip;;
    x86_64) sdk_zip=linux.zip;;
    *) echo 'Only aarch64 and x86_64 are supported'; exit 1;;
esac
cd "$root"
sha256sum --quiet -c SHA256SUMS
sudo apt-get install -y build-essential cmake unzip libspdlog-dev libfmt-dev libboost-program-options-dev libyaml-cpp-dev
mkdir -p "$root/work/tmp"
export TMPDIR="$root/work/tmp"
cmake -S "$root/unitree_sdk2" -B "$root/work/unitree-build" -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$root/work/unitree-install"
cmake --build "$root/work/unitree-build" -j "${BUILD_JOBS:-2}"
cmake --install "$root/work/unitree-build"
unzip -oq "$root/downloads/$sdk_zip" -d "$root/brainco_hand_service/thirdparty/stark-serialport-example"
prefix="$root/work/unitree-install"
cmake -S "$root/brainco_hand_service" -B "$root/brainco_hand_service/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$prefix" \
    -DCMAKE_CXX_FLAGS="-I$prefix/include -I$prefix/include/ddscxx" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib"
cmake --build "$root/brainco_hand_service/build" -j "${BUILD_JOBS:-2}"
ldd "$root/brainco_hand_service/build/brainco_hand"
"$root/brainco_hand_service/build/brainco_hand" --help
