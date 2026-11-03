#!/usr/bin/env bash
# All image paths refer to ThinkPad. No keys or images are copied.
set -euo pipefail
printf -v remote_cmd '%q ' bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/preview.sh "$@"
exec ssh -n -o ClearAllForwardings=yes -o BatchMode=yes -o ConnectTimeout=5 -- 'thinkpad' "$remote_cmd"
