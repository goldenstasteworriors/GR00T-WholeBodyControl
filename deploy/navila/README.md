# NaVILA on server02 with ThinkPad and G1

All deployment filenames, menus, and instructions in this directory are English.
The original Balance/Walk models and controller behavior remain unchanged.

## Preview only: no robot control

Use this mode to inspect the navigation VLA output. It never creates a WBC
publisher, starts the robot controller, or executes a predicted action.
There is no `--execute` option in the preview program.

On ThinkPad, run one complete line:

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/preview.sh
```

The default input is the previously captured G1 image at
`.runtime/navila/ego-view.jpg`. This is a saved image, not a live view.
It is already present on this ThinkPad and is not committed to Git.

To read the existing live camera stream without controlling G1:

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/preview.sh --live --steps 20
```

For a different instruction and image on ThinkPad:

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/preview.sh --image /absolute/path/to/image.jpg --instruction "Find the beaker and stop in front of it."
```

On the local workstation:

```bash
bash /home/ykj/Desktop/navila_preview.sh
```

Pass `--live --steps 20` for live input. All `--image` paths refer to
ThinkPad, including when launched from the workstation. Use `--steps 0`
to continue until Ctrl+C. The model service and SSH relay must be running.
Preview does not start camera services or stop other independently running
controllers. Live mode requires the existing camera stream at port 5555.

Each result includes the raw `response`, optional `parsed_action`,
`round_trip_s`, `source`, and `execute: false`. An unrecognized response
is still printed with a parsing error instead of being discarded.
Images do not change in response to predicted actions, because none are executed.

## One-command startup

See `QUICKSTART.txt`. Run each command on its named host:

```bash
# server02
bash /home/user/ykj/project/NaVILA/deploy_server02/one_click.sh
```

```bash
# Local workstation
bash /home/ykj/Desktop/navila_start.sh
```

```bash
# ThinkPad
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/start_thinkpad.sh
```

ThinkPad menu:
- `PREVIEW`: saved-image output only; available before START.
- `LIVE`: live-camera output only; available before START.
- `START`: explicit confirmation to initialize real joint control and camera preview.
- Press Enter once in the menu to initialize; after `ready`, press Enter a second time in the same menu to stand only; after `running`, press Enter a third time to navigate.
- `STAND`: alias for the second Enter; `GO`: alias for the third Enter; the default budget is 20 actions.
- `x` or Ctrl+C in the menu: explicitly interrupt workers and close this interface. `q` only detaches.
- `exit_thinkpad.sh` / workstation `exit_local.sh`: independent explicit exit.
- `e`: manual damping estop; `s`: pane status; `q`: detach without stopping processes.

Workstation menu: `p` for saved-image preview, `l` for live preview,
`e` for SSH estop, `s` for relay status, and `q` to detach.
Preview opens beside the menu so the estop menu stays available.
Repeated startup reuses the existing session and task settings.

## Manual estop

Local workstation:

```bash
bash /home/ykj/Desktop/navila_estop.sh
```

ThinkPad:

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/estop_thinkpad.sh
```

The Control pane also accepts `d`. The scripts send SIGUSR1 to the
registered controller, reusing the original DampingShutdown implementation.
A current `damping_latched` acknowledgement confirms the controller call,
not a measurement that the robot has physically stopped. A read-only
diagnostic acknowledgement explicitly says no actuator writes occurred.
The workstation requires SSH connectivity to ThinkPad; ThinkPad's local
method does not depend on server02 or the workstation connection.

These methods target the controller launched by this project's
`run_control.sh`, not arbitrary official G1 control programs.
PID/start-time checks select the intended process. The acknowledgement
wait is only for reporting a manual estop request, not a navigation watchdog.

## Control behavior

At the user's request, the added navigation lease/watchdog, camera/state
freshness checks, inference deadlines, action caps, publisher conflict
checks, and automatic zero command on bridge exceptions were removed.
They have not been reintroduced. The original controller and whole-body
policy match baseline `c1495a7`; only optional estop registration and
acknowledgement were added to `damping_shutdown.py`.

Execution uses the original ROS topic `ControlPolicy/upper_body_pose`
with `navigate_cmd=[vx,vy,wz]` and `target_time`.
Default speeds are 0.5 m/s and pi/6 rad/s, configurable through
`--linear-speed` and `--angular-speed`, without additional clipping.
A 75 cm or 45 degree action therefore requests 1.5 seconds at the default
speed. These are time-based estimates, not odometry feedback.

An action's normal completion sends zero velocity before the next
observation. A model stop also sends zero. Manual damping ends the task.
Closing the bridge alone is not an estop method. Original controller
initialization and joint handling are unchanged.

## Deployment

- ThinkPad project: `/home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link`
- Environment: existing Conda `decoupled_vla_collection`
- G1 camera: `192.168.123.164:5555`, stream `ego_view`, RGB 640x480
- ThinkPad Ethernet: `enx00e04c36c4e8`
- Local ROS domain: 73; robot SDK remains separate
- Model service: server02 loopback `127.0.0.1:54321`, NaVILA-8B FP16
- Relay: workstation SSH tunnels; the workstation must remain online
- Model input: eight history/current frames, front-padded with black images initially

No dependencies, GPU drivers, or CUDA versions were changed for these launchers.
The full real-robot workflow is in `REAL_ROBOT_TEST.txt`.

## Validation

Previous tests covered real RGB-to-model requests, the original ONNX models,
real 29-joint read-only computation, and both manual estop commands against a
read-only controller. No real walking or physical estop effectiveness was tested.

The new preview was tested with the saved image and the live G1 camera.
Example raw replies included turning right 45 degrees, turning left 45
degrees, and moving forward 25 cm. They were only displayed, never executed.
These tests demonstrate the pipeline, not beaker approach accuracy.

Relevant reports are under `.runtime/navila/`: `preview-image.jsonl`,
`preview-live.jsonl`, `manual-estop-unit.json`, `estop-status.json`,
and the one-click startup checks. Earlier watchdog test reports refer
only to the superseded implementation.

## Files

- `preview.py`, `preview.sh`, `preview_local.sh`: output-only inference.
- `one_click.py`: English tmux menus, three-Enter startup, explicit worker exit, and START/GO aliases.
- `start_thinkpad.sh`, `start_workstation.sh`, `start_server02.sh`: launchers.
- `one_click_camera.sh`: existing camera service/viewer startup.
- `bridge.py`: navigation inference and optional original WBC command publication.
- `estop.py`, `estop_local.sh`, `estop_thinkpad.sh`: manual damping estop.
- `QUICKSTART.txt`, `REAL_ROBOT_TEST.txt`: English instructions.

Branch: `codex/navila-g1-01a19ca8`.
Session: `01a19ca8-ac6e-7460-a4da-6a193efe13ca`.

## Real robot execution monitor

On the workstation, run `bash /home/ykj/Desktop/navila_monitor.sh`.
The dashboard opens at `http://127.0.0.1:8768`. It reads actual bridge JSON
stdout from the existing ThinkPad navigation pane and subscribes to the camera.
It never invokes inference, creates a motion publisher, or restarts robot control.
Latest prediction, recent commands, issued velocities and planned segment duration
are displayed. Process generation filters exclude previous runs while the process
is alive. Pane history limits the available command history. Physical completion
and displacement are not measured by this observer.
Use the third Enter in the robot menu to start navigation.
