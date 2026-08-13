#!/usr/bin/env bash
# 在本机启动：建立 SSH 隧道并打开本地浏览器。
set -euo pipefail

ROBOT_HOST="${ROBOT_HOST:-g1_bjutech_remote}"
LOCAL_PORT="${LOCAL_PORT:-5001}"
REMOTE_PORT="${REMOTE_PORT:-5000}"
URL="http://127.0.0.1:${LOCAL_PORT}"

if ! command -v ssh >/dev/null; then
  echo "未找到 ssh。" >&2
  exit 1
fi
if ! command -v xdg-open >/dev/null; then
  echo "未找到 xdg-open；隧道建立后请手动打开：$URL" >&2
fi

echo "建立 SSH 隧道：$URL -> ${ROBOT_HOST}:127.0.0.1:${REMOTE_PORT}"
echo "保持此终端运行；按 Ctrl+C 即关闭网页访问。"
# SSH 配置中包含与本网页无关的 RemoteForward 10408；该远程转发失败
# 不应影响本地 5001 -> 机器人 5000 的网页隧道。
ssh -o ExitOnForwardFailure=no -N \
  -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$ROBOT_HOST" &
SSH_PID=$!
cleanup() { kill "$SSH_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

sleep 1
if ! kill -0 "$SSH_PID" 2>/dev/null; then
  echo "SSH 隧道启动失败。" >&2
  exit 1
fi
if command -v xdg-open >/dev/null; then
  xdg-open "$URL" >/dev/null 2>&1 || echo "请手动在浏览器打开：$URL"
fi
wait "$SSH_PID"
