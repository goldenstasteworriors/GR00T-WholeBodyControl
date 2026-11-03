#!/usr/bin/env bash
# Run on server02. Uses the existing NaVILA environment and weights.
set -euo pipefail
SESSION=navila-server02
ROOT=/home/user/ykj/project/NaVILA
LOG=/data/ykj/navila/logs/server.log
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    printf -v start_cmd 'bash %q > %q 2>&1' "$ROOT/deploy_server02/serve.sh" "$LOG"
    tmux new-session -d -s "$SESSION" "$start_cmd"
fi
echo 'NaVILA 服务启动中...'
while ! ss -lnt | grep -q '127.0.0.1:54321'; do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        tail -n 30 "$LOG"
        exit 1
    fi
    sleep 1
done
echo 'NaVILA 已监听 127.0.0.1:54321'
echo "日志：$LOG"
echo '此服务器只推理，不启动机器人控制。'
