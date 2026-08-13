#!/usr/bin/env bash
# 在机器人端启动：DDS -> Modbus bridge 与仅本机可访问的网页控制服务。
# Conda 的机器人环境激活钩子会读取未定义的 CONDA_BUILD；不能启用
# nounset（-u），否则 conda activate 会在钩子执行前失败。
set -eo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-decoupled_vla_collection}"
DDS_NETWORK="${DDS_NETWORK:-eth0}"
HAND_TASK="${HAND_TASK:-open_door}"
WEB_PORT="${WEB_PORT:-5000}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "未找到 conda 初始化脚本：$CONDA_SH" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/decoupled_wbc/scripts/inspire_hand_web.py" ]]; then
  echo "未找到 inspire_hand_web.py；请先将包含本脚本的项目版本同步到机器人。" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO_ROOT"

cleanup() {
  [[ -n "${BRIDGE_PID:-}" ]] && kill "$BRIDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if pgrep -af 'decoupled_wbc/scripts/inspire_modbus_hand.py.*--mode dds' >/dev/null; then
  echo "已有 DDS -> Modbus bridge 正在运行。为避免两个 bridge 同时写手，停止本脚本并复用已有 bridge。" >&2
  echo "请确认已有 bridge 使用 --side both；若不是，请先停止它再重试。" >&2
  exit 1
fi

python decoupled_wbc/scripts/inspire_modbus_hand.py \
  --mode dds --network "$DDS_NETWORK" \
  --left-ip 192.168.123.210 --right-ip 192.168.123.211 \
  --hand-task "$HAND_TASK" --side both --publish-state &
BRIDGE_PID=$!
sleep 1
if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
  echo "DDS -> Modbus bridge 启动失败。" >&2
  exit 1
fi

echo "网页服务已启动；请在本机运行 start_inspire_hand_web_local.sh。"
python decoupled_wbc/scripts/inspire_hand_web.py \
  --network "$DDS_NETWORK" --host 127.0.0.1 --port "$WEB_PORT"
