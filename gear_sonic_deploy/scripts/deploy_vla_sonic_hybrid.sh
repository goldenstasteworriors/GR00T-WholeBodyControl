#!/usr/bin/env bash
set -euo pipefail

DATA_COLLECTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_PROJECT_ROOT="${VLA_PROJECT_ROOT:-/home/unitree/VLA/GR00T-WholeBodyControl}"
VLA_DEPLOY_ROOT="$VLA_PROJECT_ROOT/gear_sonic_deploy"
PATCH_FILE="$DATA_COLLECTION_ROOT/gear_sonic_deploy/vla_patches/sonic_hybrid_arm_override.patch"
ORIGINAL_BINARY="$VLA_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref"
HYBRID_BINARY="$VLA_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref_sonic_hybrid"
BACKUP_ROOT="$VLA_DEPLOY_ROOT/.sonic_hybrid_backup/$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$VLA_DEPLOY_ROOT/src/g1/g1_deploy_onnx_ref" ]]; then
  echo "VLA deployment source missing: $VLA_DEPLOY_ROOT" >&2
  exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Tracked VLA patch missing: $PATCH_FILE" >&2
  exit 1
fi
if [[ ! -x "$ORIGINAL_BINARY" ]]; then
  echo "Existing production SONIC binary missing: $ORIGINAL_BINARY" >&2
  exit 1
fi

mkdir -p "$BACKUP_ROOT"
cp -a "$ORIGINAL_BINARY" "$BACKUP_ROOT/g1_deploy_onnx_ref.before_hybrid"

restore_original_binary() {
  if [[ -f "$BACKUP_ROOT/g1_deploy_onnx_ref.before_hybrid" ]]; then
    cp -a "$BACKUP_ROOT/g1_deploy_onnx_ref.before_hybrid" "$ORIGINAL_BINARY"
  fi
}
trap restore_original_binary EXIT

cd "$VLA_DEPLOY_ROOT"
if patch -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
  while IFS= read -r relative_path; do
    mkdir -p "$BACKUP_ROOT/$(dirname "$relative_path")"
    cp -a "$relative_path" "$BACKUP_ROOT/$relative_path"
  done < <(sed -n 's|^--- a/||p' "$PATCH_FILE")
  patch -p1 < "$PATCH_FILE"
elif patch -R -p1 --dry-run < "$PATCH_FILE" >/dev/null 2>&1; then
  echo "VLA arm-override patch is already applied; rebuilding the dedicated binary."
else
  echo "VLA source does not match either side of the tracked patch; refusing to modify it." >&2
  exit 1
fi

cmake --build build --target g1_deploy_onnx_ref --parallel 2
install -m 755 "$ORIGINAL_BINARY" "$HYBRID_BINARY"
restore_original_binary
trap - EXIT

echo "Dedicated SONIC hybrid binary installed:"
echo "  $HYBRID_BINARY"
echo "Production binary restored unchanged:"
echo "  $ORIGINAL_BINARY"
echo "Source/binary backup:"
echo "  $BACKUP_ROOT"
