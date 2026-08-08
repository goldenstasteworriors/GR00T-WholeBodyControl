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

### ThinkPad keyboard test with Unitree loco

Use the explicit keyboard mode to test the official lower-body controller
without connecting PICO. The camera server must already be running on the
robot. Run this from the ThinkPad checkout:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --camera-host 192.168.123.164 \
  --task-prompt "grab the red bottle" \
  --dataset-name keyboard_loco_test \
  --hand-task open_door \
  --lower-body-controller unitree_loco \
  --keyboard-lower-body-control \
  --body-control-device dummy \
  --hand-control-device dummy \
  --no-enable-real-device \
  --no-pico-data-streamer \
  --no-with-hands
```

Focus the tmux control pane before pressing robot-control keys. `G` starts the
non-blocking `Damp (FSM 1) -> StandUp (FSM 4) -> locomotion (FSM 501)` sequence;
pressing `G` again requests the safe emergency stop. `W/S` adjust forward and
backward velocity, `A/D` adjust lateral velocity, `Q/E` adjust yaw rate, and
`Z` immediately resets all velocity components to zero. `Space` always requests
the safe emergency stop. Movement keys are ignored until startup is confirmed.
Each movement key press changes its component by 0.1 and the command persists,
so press `Z` before changing focus or leaving the keyboard. `C` starts/saves a
recording episode and `X` discards it.

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
