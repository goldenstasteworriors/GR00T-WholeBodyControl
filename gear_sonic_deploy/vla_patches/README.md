# VLA SONIC hybrid deployment

This directory keeps the production-source patch that is applied to:

```text
/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
```

The Git branch itself lives in the data-collection repository. Runtime models,
planner assets, and the real-robot SONIC executable always come from the VLA
tree above.

## Control ownership

- SONIC is the only `rt/lowcmd` publisher.
- SONIC owns leg motors 0–11 and waist motors 12–14.
- The decoupled teleop process publishes a 14-DOF `arm_position` /
  `arm_velocity` ZMQ field in MuJoCo order (left arm 15–21, right arm 22–28).
- Before inference, those targets replace only the arm entries in all future
  motion-reference frames.
- After inference, those targets replace only arm motor commands 15–28.
- Overridden targets are converted back to normalized IsaacLab action order and
  written into `last_action`.
- A SONIC stop command ends the authorized SONIC process. Relaunch the tmux
  session before starting another control session.

## Build the dedicated binary

Run from this data-collection worktree:

```bash
bash gear_sonic_deploy/scripts/deploy_vla_sonic_hybrid.sh
```

The script backs up the current VLA source and production binary, applies the
tracked patch, builds SONIC, installs:

```text
/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref_sonic_hybrid
```

and restores the original `g1_deploy_onnx_ref` byte-for-byte.
