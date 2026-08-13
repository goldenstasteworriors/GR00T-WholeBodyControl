# Decoupled WBC VLA Data Collection

This path records decoupled WBC teleoperation into the same LeRobot dataset
schema used by the Sonic VLA exporter.

## Setup

From the repository root:

```bash
bash install_scripts/install_decoupled_vla_collection.sh
conda activate decoupled_vla_collection
source "$CONDA_PREFIX/setup.bash"
```

The install script creates one conda environment that contains ROS 2 Humble,
`decoupled_wbc`, `gear_sonic`, Unitree Python SDK, XRoboToolkit, and Isaac
Teleop / CloudXR dependencies. It does not modify CUDA or GPU drivers.

## Real Robot

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "pick up the cup" \
  --dataset-name real__test_002 \
  --hand-task open_door
```

The launcher starts a tmux session with:

- decoupled WBC control loop
- decoupled PICO teleop loop
- Sonic-VLA-compatible decoupled data exporter
- camera viewer
- optional PICO metadata streamer for `teleop.smpl_*` and `teleop.vr_3pt_*`

The metadata streamer and decoupled teleop reuse the same RoboticsService
process. Full-body tracking is optional: when it is unavailable, the exporter
still records headset/controller data in `teleop.vr_3pt_*`, while
`teleop.smpl_*` remains zero-filled. If PICO metadata is not needed, disable the
extra streamer with:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "pick up the cup" \
  --dataset-name real__test_002 \
  --hand-task open_door \
  --no-pico-data-streamer
```

With `--no-pico-data-streamer`, the dataset schema stays the same, but PICO-only
SMPL/VR3PT fields are filled with defaults unless they can be derived from
decoupled state.

### Unitree official lower-body control

For a 29-DOF G1 EDU with a 3-DOF waist, keep the Unitree loco service active and
send only the arm targets through `rt/arm_sdk`:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_7_23 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --lower-body-controller unitree_loco
```

When `A+B+X+Y` requests startup, the official backend follows the G1 `sport`
service API with a non-blocking `Damp (FSM 1) -> StandUp (FSM 4) -> Start
(FSM 501) -> zero velocity -> prepare arms` sequence. Before FSM 501 it waits
for fresh low-state measurements, low leg velocity, and an upright torso. Once
the dynamic locomotion FSM is active, it keeps sending zero velocity and checks
fresh state plus torso tilt without requiring every leg joint's instantaneous
velocity to fall below the locked-standing threshold. Only then does it blend
motors 15--28 into `rt/arm_sdk`. As in `xr_teleoperate --motion`, the complete
29-DOF arm-sdk packet is first seeded once from the measured pose, after which
only the dual-arm targets at motors 15--28 are updated; the waist targets are
not continuously rewritten by upper-body IK. Firmware that accepts velocity directly from its standing FSM can opt
out of the explicit Start transition with
`--unitree-loco-start-fsm-id -1`. `A+X` remains disabled until the complete
sequence succeeds. A second `A+B+X+Y`, an RPC error, or a startup timeout
releases the arm weight, commands zero velocity, and requests Damp.

This mode intentionally rejects waist IK. The first `A+B+X+Y` press starts the
full sequence above and enables navigation only after the final confirmation.
`A+X` pauses or resumes upper-body teleoperation while preserving the official
standing controller. The next `A+B+X+Y` press releases `arm_sdk`, sends zero
velocity, and requests the official damp state. The PICO left joystick controls
forward/backward and sideways velocity; the right joystick controls yaw. Do not
send simultaneous movement commands from the Unitree remote during collection.

The onboard wrapper selects `unitree_loco` by default, so the robot-local launch
command below does not need an extra controller argument. Pass
`--lower-body-controller decoupled` explicitly only when reverting to the old
local Balance/Walk ONNX lower-body controller.

### Robot-onboard wireless keyboard collection

For cable-free operation, keep the Unitree DDS control loop and dataset writer
on the robot and use the workstation only to SSH into the robot's tmux session.
PICO DHCP and PC Service are not required. Start the robot camera separately,
then run:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection_onboard.py \
  --task-prompt "grab the red bottle" \
  --dataset-name keyboard_loco_test \
  --hand-task open_door \
  --keyboard-lower-body-control \
  --keyboard-loco-command-timeout 0.5 \
  --body-control-device dummy \
  --hand-control-device dummy \
  --no-enable-real-device \
  --no-pico-data-streamer \
  --no-with-hands
```

Focus the tmux control pane before entering commands. `G` starts the non-blocking
`Damp (FSM 1) -> StandUp (FSM 4) -> Start (FSM 501) -> zero velocity -> prepare arms` sequence;
pressing `G` again or pressing `Space` requests the safe emergency stop. Hold `W/S` for
fixed +/-0.1 m/s forward/backward motion, `A/D` for fixed +/-0.1 m/s lateral
motion, and `Q/E` for fixed +/-0.1 rad/s yaw. `Z` resets all velocity components
immediately. The controller also resets velocity after 0.5 seconds without a
movement key, including when the SSH client disconnects.
`C` starts/saves a recording episode and `X` discards it. Dataset output remains
under `outputs/onboard/<dataset-name>`.

### Robot-onboard wired PICO collection

The robot's external RJ45 ports share the onboard `eth0` network. The PICO
Ethernet adapter requests an address with DHCP, so start the robot-local DHCP
and ARM64 XRoboToolkit PC Service before opening the collection program. The
service script only leases `192.168.123.200` to the known PICO Ethernet MAC and
does not change the robot's existing `192.168.123.164/24` configuration.

From the repository root:

```bash
bash scripts/onboard_pico_services.sh start
bash scripts/onboard_pico_services.sh status
```

`start` asks for the robot sudo password once, starts `dnsmasq` as a managed
background daemon, and starts the unprivileged PC Service in tmux. It also
repairs a partial startup when only one of the two services is running. Keep
the PC Service tmux session running. To inspect it, use:

```bash
bash scripts/onboard_pico_services.sh attach
```

DHCP status and its log path are shown by `status`; the log is stored under
`.runtime/onboard_pico/`. The `stop` command stops both the tmux PC Service and
the managed DHCP daemon (and may ask for sudo once).

Inside the PICO XRoboToolkit app, set `PC Service` to `192.168.123.164` and
select `Reconnect`. The expected PICO address is `192.168.123.200`.

Use the dedicated onboard wrapper so the original ThinkPad launch commands and
defaults remain unchanged:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection_onboard.py \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_7_23 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer
```

The wrapper defaults to the robot-local camera endpoint `192.168.123.164` and
the `decoupled_vla_collection` conda environment. Datasets are stored locally
on the robot under:

```text
/home/unitree/data_collection/GR00T-WholeBodyControl/outputs/onboard/<dataset-name>
```

#### Low-priority Inspire tactile collection

For a left RH56DFTP hand, add `--collect-tactile` to the normal onboard launch.
The conservative default reads 11 taxel batches plus one six-channel force
batch, targeting two complete physical refreshes per second. The exporter is
still 50 Hz: each dataset frame copies the newest complete cache without
waiting for Modbus. A failed batch keeps the previous values and advances the
per-region age instead of replacing the region with zeros.
The default 16 ms state-deadline guard was selected from onboard read-only
measurements so the existing Inspire state loop retains priority.

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection_onboard.py \
  --task-prompt "grab the beaker and shake" \
  --dataset-name 8_12_shake_beaker_1 \
  --hand-task open_door \
  --no-pico-data-streamer \
  --with-hands \
  --inspire-hand-bridge \
  --left-hand-only \
  --collect-tactile \
  --unitree-loco-navigation-enabled \
  --pico-navigation-range 1 \
  --pico-fixed-side right
```

The dataset adds raw taxels, per-region validity/age/update counters, force
feedback, and rolling Modbus metrics under `observation.tactile.*`. The bridge
also appends a project-local JSONL record to
`logs/tactile_modbus_<dataset-name>.jsonl` every five seconds. In particular,
inspect `state_cycle_hz`, `state_deadline_miss_ratio`, `modbus_busy_ratio`,
`tactile_io_p95_ms`, and `estimated_safe_max_full_refresh_hz` before increasing
`--tactile-full-refresh-hz`. The estimate reserves 40 percent of the measured
Modbus time as safety headroom; do not increase the rate while the 50 Hz state
loop has deadline misses.

The collection launcher runs in its own `decoupled_vla_collection` tmux
session. Detach with `Ctrl+b`, then `d`, and reattach with:

```bash
tmux attach -t decoupled_vla_collection
```

To display the robot camera on the local workstation while collection and
dataset storage remain onboard, open another terminal in the local checkout
and run:

```bash
bash scripts/view_onboard_camera.sh
```

This creates an SSH tunnel through the configured `g1_bjutech_remote` host and
starts the OpenCV viewer in the local `decoupled_vla_collection` conda
environment. Press `Q` in the viewer window to close it. The tunnel is closed
automatically. The robot camera server must already be listening on port
`5555`; the helper reports an error instead of opening an empty viewer when it
is not running.

After collection has stopped, verify the dataset before copying it off the
robot:

```bash
du -sh outputs/onboard/<dataset-name>
find outputs/onboard/<dataset-name>/meta -maxdepth 1 -type f -print
```

Stop the PICO network and PC Service only after collection has finished:

```bash
bash scripts/onboard_pico_services.sh stop
```

## Simulation

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --sim \
  --task-prompt "pick up the cup" \
  --dataset-name sim__test_002 \
  --hand-task open_door
```

In simulation, the launcher starts `decoupled_wbc/control/main/teleop/run_sim_loop.py`
in a separate tmux window with image publishing enabled. The control loop uses
`--simulator none` and communicates with the simulator through the same Unitree
SDK bridge used by the official decoupled workflow.

## Dataset Compatibility

The exporter writes the feature and modality config from:

- `gear_sonic/data/features_sonic_vla.py`

The data keys match `gear_sonic/scripts/run_data_exporter.py`, including:

- `observation.images.ego_view`
- `observation.state`
- `observation.eef_state`
- `action.wbc`
- `observation.root_orientation`
- `observation.projected_gravity`
- `observation.cpp_rotation_offset`
- `observation.init_base_quat`
- `teleop.delta_heading`
- `action.motion_token`
- `teleop.smpl_joints`
- `teleop.smpl_pose`
- `teleop.body_quat_w`
- `teleop.target_body_orientation`
- `teleop.left_hand_joints`
- `teleop.right_hand_joints`
- `teleop.smpl_frame_index`
- `teleop.left_wrist_joints`
- `teleop.right_wrist_joints`
- `teleop.stream_mode`
- `teleop.planner_mode`
- `teleop.planner_movement`
- `teleop.planner_facing`
- `teleop.planner_speed`
- `teleop.planner_height`
- `teleop.vr_3pt_position`
- `teleop.vr_3pt_orientation`

Fields with decoupled-native values:

- `observation.state`: full 43-DOF joint state from decoupled ROS state
- `action.wbc`: full 43-DOF WBC action from decoupled ROS state
- `observation.eef_state`: decoupled wrist pose, with FK fallback
- root orientation and projected gravity: decoupled floating-base state
- hand and wrist joints: PICO pose first, decoupled joint state fallback

Fields kept for Sonic VLA schema compatibility:

- `action.motion_token`: zero vector unless a token is explicitly present
- `teleop.planner_*`: default values unless Sonic PICO manager mode publishes planner data
- `teleop.stream_mode`: manager stream mode if available, otherwise pose mode when fresh PICO pose is present

## Recording Controls

The exporter listens to the same recording toggles as the decoupled control
loop:

- `c`: start or stop/save an episode
- `x`: discard the current episode

If the optional PICO metadata streamer is active, PICO `A` / `B` collection
toggles from `pico_manager_thread_server.py` are also accepted by the exporter.
