#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
build="$root/.runtime/brainco-both-test"
mkdir -p "$build/tmp"
export TMPDIR="$build/tmp"
sdk=${UNITREE_SDK_PREFIX:-/usr/local}
if [[ -f "$sdk/include/unitree/robot/channel/channel_subscriber.hpp" && -f "$sdk/lib/libunitree_sdk2.a" ]]; then
    includes=(-I"$sdk/include" -I"$sdk/include/ddscxx")
    libs=(-L"$sdk/lib" -Wl,-rpath,"$sdk/lib")
else
    sdk="$root/gear_sonic_deploy/thirdparty/unitree_sdk2"
    includes=(-I"$sdk/include" -I"$sdk/thirdparty/include" -I"$sdk/thirdparty/include/ddscxx")
    libs=(-L"$sdk/lib/$(uname -m)" -L"$sdk/thirdparty/lib/$(uname -m)" -Wl,-rpath,"$sdk/thirdparty/lib/$(uname -m)")
fi
if [[ ! -x "$build/both_hands" || "$here/both_hands.cpp" -nt "$build/both_hands" ]]; then
    c++ -std=c++17 -O2 -pthread "${includes[@]}" "$here/both_hands.cpp" "${libs[@]}" -lunitree_sdk2 -lddsc -lddscxx -lrt -o "$build/both_hands"
fi
exec "$build/both_hands" "$@"
