#!/usr/bin/env bash
# 在机器人端启动：DDS -> Modbus bridge 与仅本机可访问的网页控制服务。
# Conda 的机器人环境激活钩子会读取未定义的 CONDA_BUILD；不能启用
# nounset（-u），否则 conda activate 会在钩子执行前失败。
set -eo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-decoupled_vla_collection}"
HAND_SIDE="${HAND_SIDE:-left}"
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
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$HAND_SIDE" != "left" && "$HAND_SIDE" != "right" && "$HAND_SIDE" != "both" ]]; then
  echo "HAND_SIDE 只能是 left、right 或 both，当前值：$HAND_SIDE" >&2
  exit 1
fi

if pgrep -af 'decoupled_wbc/scripts/inspire_modbus_hand.py.*--mode dds' >/dev/null; then
  echo "检测到已有 DDS -> Modbus bridge；仅启动网页并复用该 bridge。"
  pgrep -af 'decoupled_wbc/scripts/inspire_modbus_hand.py.*--mode dds'
  exec python decoupled_wbc/scripts/inspire_hand_web.py \
    --network "$DDS_NETWORK" --host 127.0.0.1 --port "$WEB_PORT"
fi

echo "未检测到 bridge；启动 $HAND_SIDE 手 DDS -> Modbus bridge。"
python decoupled_wbc/scripts/inspire_hand_web.py \
  --network "$DDS_NETWORK" --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!
sleep 1
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  echo "网页服务启动失败。" >&2
  exit 1
fi

echo "网页服务已启动；请在本机运行 start_inspire_hand_web_local.sh。"
echo "DDS -> Modbus bridge 运行中；若它退出，网页服务会一并停止。"
python decoupled_wbc/scripts/inspire_modbus_hand.py \
  --mode dds --network "$DDS_NETWORK" \
  --left-ip 192.168.123.210 --right-ip 192.168.123.211 \
  --hand-task "$HAND_TASK" --side "$HAND_SIDE" --publish-state
