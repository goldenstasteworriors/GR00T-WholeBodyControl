#!/usr/bin/env bash
set -euo pipefail
exec ssh -n -o ClearAllForwardings=yes -o BatchMode=yes -o ConnectTimeout=5 -- 'thinkpad' 'bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/exit_thinkpad.sh'
