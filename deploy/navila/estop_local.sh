#!/usr/bin/env bash
set -euo pipefail
exec ssh -o ClearAllForwardings=yes -o BatchMode=yes -o ConnectTimeout=3 -- 'thinkpad' 'bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/estop_thinkpad.sh'
