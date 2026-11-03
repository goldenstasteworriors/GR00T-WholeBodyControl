#!/usr/bin/env bash
# Run on the workstation with the existing server02 and ThinkPad SSH keys.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=navila-g1-relay
case "${1:-status}" in
  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      exec bash "$0" status
    fi
    tmux new-session -d -s "$SESSION" -n relay
    tmux set-option -t "$SESSION" remain-on-exit on
    tmux split-window -h -t "$SESSION:relay"
    printf -v first 'ssh -F %q -N -L 127.0.0.1:15432:127.0.0.1:54321 server02' "$HERE/relay_ssh_config"
    printf -v second 'ssh -F %q -N -R 127.0.0.1:54321:127.0.0.1:15432 thinkpad' "$HERE/relay_ssh_config"
    tmux respawn-pane -k -t "$SESSION:relay.0" "$first"
    tmux respawn-pane -k -t "$SESSION:relay.1" "$second"
    sleep 2
    bash "$0" status
    ;;
  status)
    tmux list-panes -t "$SESSION" -F '#{pane_index} alive=#{==:#{pane_dead},0} command=#{pane_current_command} exit=#{pane_dead_status}'
    for pane in 0 1; do
      if [[ "$(tmux display-message -p -t "$SESSION:relay.$pane" '#{pane_dead}')" == 1 ]]; then
        tmux capture-pane -p -t "$SESSION:relay.$pane" -S -20
        exit 1
      fi
    done
    ;;
  stop) tmux kill-session -t "$SESSION" ;;
  *) echo "Usage: bash $0 {start|status|stop}" >&2; exit 2 ;;
esac
