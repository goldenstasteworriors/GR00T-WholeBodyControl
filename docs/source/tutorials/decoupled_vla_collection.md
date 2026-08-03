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

### Robot-onboard wired PICO collection

On the G1 onboard computer, use the dedicated wrapper so the original ThinkPad
launch commands and defaults remain unchanged:

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection_onboard.py \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_7_23 \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer
```

The wrapper defaults to the robot-local camera endpoint `192.168.123.164` and
the `decoupled_vla_collection` conda environment. The PICO XRoboToolkit app
should point its PC Service address at the robot's wired address.

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
