import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.teleop.device.pico.xr_client import XrClient
from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput
from gear_sonic.utils.teleop.pico_buttons import PicoButtonEventSampler, PicoButtonState

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


_SMPL_PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22,
]


def _run_main_smpl_visualizer(frame_queue, stop_event):
    """Run Main collection's full-body visualizer in an isolated process."""
    import torch

    from gear_sonic.scripts.pico_manager_thread_server import (
        ThreePointPose,
        compute_from_body_poses,
    )

    three_point = ThreePointPose(
        enable_vis_vr3pt=True,
        with_g1_robot=True,
        enable_smpl_vis=True,
        log_prefix="DecoupledPoseVis",
    )
    device = torch.device("cpu")
    print("[PicoStreamer] Main full-body PICO visualization enabled", flush=True)
    try:
        while not stop_event.is_set():
            try:
                body_poses = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            latest_data = compute_from_body_poses(
                _SMPL_PARENT_INDICES,
                device,
                np.asarray(body_poses),
            )
            smpl_joints = latest_data["smpl_joints_local"].detach().cpu().numpy()[0]
            three_point.process_smpl_pose(
                np.asarray(body_poses),
                smpl_joints_local=smpl_joints,
            )
    finally:
        three_point.close()


HAND_STATE_DEBOUNCE_S = 0.1


class HandStateDebouncer:
    """Confirm a left-trigger/right-grip state after a stable interval."""

    def __init__(self, stable_for_s: float = HAND_STATE_DEBOUNCE_S):
        self.stable_for_s = float(stable_for_s)
        self._confirmed = (False, False)
        self._candidate: tuple[bool, bool] | None = None
        self._candidate_since: float | None = None

    def update(
        self, left_trigger: float, right_grip: float, *, now: float | None = None
    ) -> tuple[float, float]:
        timestamp = time.monotonic() if now is None else float(now)
        requested = (left_trigger > 0.5, right_grip > 0.5)
        if requested == self._confirmed:
            self._candidate = None
            self._candidate_since = None
        elif requested != self._candidate:
            self._candidate = requested
            self._candidate_since = timestamp
        elif self._candidate_since is not None and timestamp - self._candidate_since >= self.stable_for_s:
            self._confirmed = requested
            self._candidate = None
            self._candidate_since = None
        return float(self._confirmed[0]), float(self._confirmed[1])

class PicoStreamer(BaseStreamer):
    def __init__(
        self,
        enable_smpl_visualization: bool = False,
        navigation_range: float = 1.0,
    ):
        if navigation_range <= 0.0:
            raise ValueError("PICO navigation range must be positive")
        self.navigation_range = float(navigation_range)
        self.run_pico_service()
        self.xr_client = XrClient()
        self._xr_lock = threading.Lock()
        self._left_hand_state_debouncer = HandStateDebouncer()
        self._button_sampler = PicoButtonEventSampler(
            self._read_official_button_state,
            poll_hz=200.0,
        )
        self._button_events_consumed = {
            "ax": 0,
            "by": 0,
            "start_stop": 0,
            "data_collection": 0,
            "data_abort": 0,
            "emergency_stop": 0,
        }
        # Keep the last event identifier in every teleop message so a
        # collection command cannot be lost between equal-rate ROS loops.
        self._data_collection_event_id = 0
        self._data_abort_event_id = 0
        self._button_sampler.start()
        self.enable_smpl_visualization = enable_smpl_visualization
        self._smpl_context = mp.get_context("spawn")
        self._smpl_queue = None
        self._smpl_stop = None
        self._smpl_process = None

        self.reset_status(reset_control_enabled=True)

    def run_pico_service(self):
        existing_service = subprocess.run(
            ["pgrep", "-f", "[rR]obotics[Ss]ervice[Pp]rocess"],
            capture_output=True,
            text=True,
            check=False,
        )
        if existing_service.returncode == 0:
            self.pico_service_pid = None
            print(
                "Pico service is already running; reusing it for decoupled teleop "
                f"(pid {existing_service.stdout.splitlines()[0]})."
            )
            return

        # Start the real service process directly.  runService.sh backgrounds
        # it, so the returned PID belongs to a short-lived shell and cannot be
        # cleaned up when the collection session exits.
        service_dir = Path("/opt/apps/roboticsservice")
        service_env = os.environ.copy()
        service_env["LD_LIBRARY_PATH"] = ":".join(
            filter(
                None,
                [
                    service_env.get("LD_LIBRARY_PATH", ""),
                    str(service_dir),
                    str(service_dir / "lib"),
                    str(service_dir / "SDK/x64"),
                ],
            )
        )
        service_env["QT_PLUGIN_PATH"] = ":".join(
            filter(None, [str(service_dir / "plugins"), service_env.get("QT_PLUGIN_PATH", "")])
        )
        service_env["QT_QML_PATH"] = ":".join(
            filter(None, [str(service_dir / "qml"), service_env.get("QT_QML_PATH", "")])
        )
        launcher_process = subprocess.Popen(
            [str(service_dir / "RoboticsServiceProcess")],
            cwd=service_dir,
            env=service_env,
        )
        try:
            launcher_process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            service_pid = launcher_process.pid
        else:
            service_pid = None
        for _ in range(50):
            if service_pid is not None:
                break
            detected = subprocess.run(
                ["pgrep", "-f", "[rR]obotics[Ss]ervice[Pp]rocess"],
                capture_output=True,
                text=True,
                check=False,
            )
            if detected.returncode == 0:
                service_pid = int(detected.stdout.splitlines()[-1])
                break
            time.sleep(0.1)
        if service_pid is None:
            raise RuntimeError("RoboticsServiceProcess did not start within 5 seconds")
        self.pico_service_pid = service_pid
        print(f"Pico service running with pid {service_pid}")

    def stop_pico_service(self):
        if self.pico_service_pid is None:
            print("Pico service is shared with another process; leaving it running")
            return
        service_pid = self.pico_service_pid
        try:
            os.kill(service_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            try:
                os.kill(service_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(service_pid, signal.SIGKILL)
        print(f"Pico service stopped with pid {service_pid}")
        self.pico_service_pid = None

    def reset_status(self, reset_control_enabled: bool = False):
        self.current_base_height = 0.74  # Initial base height, 0.74m (standing height)
        self.combo_suppression_active = False
        self._button_sampler.clear_events()
        if reset_control_enabled or not hasattr(self, "control_enabled"):
            self.control_enabled = False

    def set_lower_body_policy_active(self, active: bool) -> None:
        """Mirror the control loop's confirmed state for start/stop button semantics."""
        self.control_enabled = bool(active)

    def get_button_diagnostics(self) -> dict:
        """Expose raw, latched, and consumed counters without changing controls."""
        sampler = self._button_sampler.diagnostics()
        return {
            "state": sampler.state,
            "samples": sampler.samples,
            "read_errors": sampler.read_errors,
            "last_error": sampler.last_error,
            "latched": {
                "ax": sampler.ax_latched,
                "by": sampler.by_latched,
                "start_stop": sampler.start_stop_latched,
                "data_collection": sampler.data_collection_latched,
                "data_abort": sampler.data_abort_latched,
                "emergency_stop": sampler.emergency_stop_latched,
            },
            "consumed": self._button_events_consumed.copy(),
        }

    def start_streaming(self):
        print("Waiting for PICO/XRoboToolkit headset and controller data...")
        last_report_time = 0.0
        first_timestamp = None
        while True:
            try:
                with self._xr_lock:
                    left_pose = self.xr_client.get_pose_by_name("left_controller")
                    right_pose = self.xr_client.get_pose_by_name("right_controller")
                    head_pose = self.xr_client.get_pose_by_name("headset")
                    timestamp = self.xr_client.get_timestamp_ns()
                poses_ready = (
                    self._is_valid_xr_pose(left_pose)
                    and self._is_valid_xr_pose(right_pose)
                    and self._is_valid_xr_pose(head_pose)
                )
                if poses_ready and timestamp > 0:
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    elif timestamp != first_timestamp:
                        print("PICO headset and controller data received.")
                        self._start_smpl_visualizer()
                        return
            except Exception as exc:
                if time.monotonic() - last_report_time >= 1.0:
                    print(f"waiting for headset/controller data... ({exc})")
                    last_report_time = time.monotonic()
                time.sleep(0.1)
                continue

            if time.monotonic() - last_report_time >= 1.0:
                print("waiting for headset/controller data...")
                last_report_time = time.monotonic()
            time.sleep(0.1)

    @staticmethod
    def _is_valid_xr_pose(pose) -> bool:
        pose = np.asarray(pose)
        if pose.shape[0] < 7 or not np.isfinite(pose[:7]).all():
            return False
        return np.linalg.norm(pose[3:7]) > 1e-6

    def stop_streaming(self):
        if self._smpl_stop is not None:
            self._smpl_stop.set()
        if self._smpl_process is not None:
            self._smpl_process.join(timeout=2.0)
        self._button_sampler.stop()
        self.xr_client.close()
        self.stop_pico_service()

    def _start_smpl_visualizer(self):
        if not self.enable_smpl_visualization or self._smpl_process is not None:
            return
        self._smpl_queue = self._smpl_context.Queue(maxsize=1)
        self._smpl_stop = self._smpl_context.Event()
        self._smpl_process = self._smpl_context.Process(
            target=_run_main_smpl_visualizer,
            args=(self._smpl_queue, self._smpl_stop),
            name="pico-main-smpl-visualizer",
            daemon=True,
        )
        self._smpl_process.start()

    def _publish_smpl_visualization_frame(self, pico_data):
        if self._smpl_queue is None:
            return
        body_data = pico_data.get("body_tracking_data")
        if not body_data or "poses" not in body_data:
            return
        body_poses = np.asarray(body_data["poses"]).copy()
        try:
            self._smpl_queue.put_nowait(body_poses)
        except queue.Full:
            try:
                self._smpl_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._smpl_queue.put_nowait(body_poses)
            except queue.Full:
                pass

    def get(self) -> StreamerOutput:
        pico_data = self._get_pico_data()

        raw_data = self._generate_unified_raw_data(pico_data)
        return raw_data

    def __del__(self):
        pass

    def _get_pico_data(self):
        with self._xr_lock:
            return self._get_pico_data_locked()

    def _get_pico_data_locked(self):
        pico_data = {}

        # Get the pose of the left and right controllers and the headset
        pico_data["left_pose"] = self.xr_client.get_pose_by_name("left_controller")
        pico_data["right_pose"] = self.xr_client.get_pose_by_name("right_controller")
        pico_data["head_pose"] = self.xr_client.get_pose_by_name("headset")

        # Get key value of the left and right controllers
        pico_data["left_trigger"] = self.xr_client.get_key_value_by_name("left_trigger")
        pico_data["right_trigger"] = self.xr_client.get_key_value_by_name("right_trigger")
        pico_data["right_grip"] = self.xr_client.get_key_value_by_name("right_grip")

        # The official combinations are sampled and edge-latched at 200 Hz.
        button_state = self._button_sampler.latest_state()
        pico_data["A"] = button_state.a
        pico_data["B"] = button_state.b
        pico_data["X"] = button_state.x
        pico_data["Y"] = button_state.y
        pico_data["left_grip"] = button_state.left_grip
        pico_data["left_axis_click"] = button_state.left_axis_click
        pico_data["right_axis_click"] = button_state.right_axis_click

        # Get the remaining button state of the left and right controllers
        pico_data["left_menu_button"] = self.xr_client.get_button_state_by_name("left_menu_button")
        pico_data["right_menu_button"] = self.xr_client.get_button_state_by_name(
            "right_menu_button"
        )
        # Get the timestamp of the left and right controllers
        pico_data["timestamp"] = self.xr_client.get_timestamp_ns()

        # Get the hand tracking state of the left and right controllers
        pico_data["left_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("left")
        pico_data["right_hand_tracking_state"] = self.xr_client.get_hand_tracking_state("right")

        # Get the joystick state of the left and right controllers
        pico_data["left_joystick"] = self.xr_client.get_joystick_state("left")
        pico_data["right_joystick"] = self.xr_client.get_joystick_state("right")

        # Get the motion tracker data
        pico_data["motion_tracker_data"] = self.xr_client.get_motion_tracker_data()

        # Get the body tracking data
        pico_data["body_tracking_data"] = self.xr_client.get_body_tracking_data()

        return pico_data

    def _read_official_button_state(self) -> PicoButtonState:
        with self._xr_lock:
            return PicoButtonState(
                a=bool(self.xr_client.get_button_state_by_name("A")),
                b=bool(self.xr_client.get_button_state_by_name("B")),
                x=bool(self.xr_client.get_button_state_by_name("X")),
                y=bool(self.xr_client.get_button_state_by_name("Y")),
                left_grip=float(self.xr_client.get_key_value_by_name("left_grip")),
                left_axis_click=bool(
                    self.xr_client.get_button_state_by_name("left_axis_click")
                ),
                right_axis_click=bool(
                    self.xr_client.get_button_state_by_name("right_axis_click")
                ),
            )

    def _generate_unified_raw_data(self, pico_data):
        self._publish_smpl_visualization_frame(pico_data)

        # Get controller position and orientation in z up world frame
        left_controller_T = self._process_xr_pose(pico_data["left_pose"], pico_data["head_pose"])
        right_controller_T = self._process_xr_pose(pico_data["right_pose"], pico_data["head_pose"])

        # Get navigation commands
        lin_vel_x, lin_vel_y, ang_vel_z = self._joystick_navigation_command(
            pico_data["left_joystick"],
            pico_data["right_joystick"],
        )

        button_events = self._button_sampler.consume_events()
        for event in button_events:
            self._button_events_consumed["ax"] += int(event.ax_pressed)
            self._button_events_consumed["by"] += int(event.by_pressed)
            self._button_events_consumed["start_stop"] += int(event.start_stop_pressed)
            self._button_events_consumed["data_collection"] += int(
                event.toggle_data_collection
            )
            self._button_events_consumed["data_abort"] += int(event.toggle_data_abort)
            self._button_events_consumed["emergency_stop"] += int(
                event.emergency_stop_pressed
            )
        emergency_stop_event = any(
            event.emergency_stop_pressed for event in button_events
        )
        start_stop_event = any(event.start_stop_pressed for event in button_events)
        upper_body_event = not (emergency_stop_event or start_stop_event) and any(
            event.ax_pressed for event in button_events
        )
        toggle_data_collection = any(
            event.toggle_data_collection for event in button_events
        )
        toggle_data_abort = any(event.toggle_data_abort for event in button_events)
        if toggle_data_collection:
            self._data_collection_event_id = time.time_ns()
            print(
                "[PicoStreamer] Left Grip+A detected: toggling recording "
                f"(event {self._data_collection_event_id})",
                flush=True,
            )
        if toggle_data_abort:
            self._data_abort_event_id = time.time_ns()
            print(
                "[PicoStreamer] Left Grip+B detected: discarding recording "
                f"(event {self._data_abort_event_id})",
                flush=True,
            )

        face_combo_pressed = pico_data["A"] and pico_data["B"] and pico_data["X"] and pico_data["Y"]
        upper_body_combo_pressed = pico_data["A"] and pico_data["X"] and not (
            pico_data["B"] or pico_data["Y"]
        )
        face_button_pressed = pico_data["A"] or pico_data["B"] or pico_data["X"] or pico_data["Y"]

        if face_combo_pressed or upper_body_combo_pressed:
            self.combo_suppression_active = True
        elif self.combo_suppression_active and not face_button_pressed:
            self.combo_suppression_active = False

        set_policy_action = None
        set_teleop_active = None
        toggle_activation = False
        emergency_stop = False
        if emergency_stop_event:
            self.control_enabled = False
            set_policy_action = False
            set_teleop_active = False
            emergency_stop = True
            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel_z = 0.0
            print(
                "[PicoStreamer] Both thumbsticks clicked: unconditional emergency Damp",
                flush=True,
            )
        elif start_stop_event:
            self.control_enabled = not self.control_enabled
            set_policy_action = self.control_enabled
            emergency_stop = not self.control_enabled
            print(
                "[PicoStreamer] A+B+X+Y detected: "
                f"{'starting policy' if self.control_enabled else 'stopping policy'}",
                flush=True,
            )
            if emergency_stop:
                set_teleop_active = False
                lin_vel_x = 0.0
                lin_vel_y = 0.0
                ang_vel_z = 0.0
        elif upper_body_event:
            toggle_activation = True
            print("[PicoStreamer] A+X detected: toggling upper-body teleop", flush=True)

        # Get base height command
        height_increment = 0.01  # Small step per call when button is pressed
        if not self.combo_suppression_active and pico_data["Y"]:
            self.current_base_height += height_increment
        elif (
            not self.combo_suppression_active
            and pico_data["B"]
            and pico_data["left_grip"] <= 0.5
        ):
            self.current_base_height -= height_increment
        self.current_base_height = np.clip(self.current_base_height, 0.2, 0.74)

        # Get gripper commands
        left_fingers = self._generate_finger_data(pico_data, "left")
        right_fingers = self._generate_finger_data(pico_data, "right")

        control_data = {
            "base_height_command": self.current_base_height,
            "navigate_cmd": [lin_vel_x, lin_vel_y, ang_vel_z],
            "right_trigger": float(pico_data["right_trigger"]),
            "toggle_policy_action": False,
        }
        if set_policy_action is not None:
            control_data["set_policy_action"] = set_policy_action
        if emergency_stop:
            control_data["emergency_stop"] = True

        teleop_data = {
            "toggle_activation": toggle_activation,
        }
        if set_teleop_active is not None:
            teleop_data["set_active"] = set_teleop_active

        return StreamerOutput(
            ik_data={
                "left_wrist": left_controller_T,
                "right_wrist": right_controller_T,
                "left_fingers": {"position": left_fingers},
                "right_fingers": {"position": right_fingers},
            },
            control_data=control_data,
            teleop_data=teleop_data,
            data_collection_data={
                "toggle_data_collection": toggle_data_collection,
                "toggle_data_abort": toggle_data_abort,
                "data_collection_event_id": self._data_collection_event_id,
                "data_abort_event_id": self._data_abort_event_id,
            },
            source="pico",
        )

    def _process_xr_pose(self, controller_pose, headset_pose):
        # Convert controller pose to x, y, z, w quaternion
        xr_pose_xyz = np.array(controller_pose)[:3]  # x, y, z
        xr_pose_quat = np.array(controller_pose)[3:]  # x, y, z, w

        # Handle all-zero quaternion case by using identity quaternion
        if np.allclose(xr_pose_quat, 0):
            xr_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        xr_pose_xyz = R_HEADSET_TO_WORLD @ xr_pose_xyz
        xr_pose_rotation = R.from_quat(xr_pose_quat).as_matrix()
        xr_pose_rotation = R_HEADSET_TO_WORLD @ xr_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Convert headset pose to x, y, z, w quaternion
        headset_pose_xyz = np.array(headset_pose)[:3]
        headset_pose_quat = np.array(headset_pose)[3:]

        if np.allclose(headset_pose_quat, 0):
            headset_pose_quat = np.array([0, 0, 0, 1])  # identity quaternion: x, y, z, w

        # Convert from y up to z up
        headset_pose_xyz = R_HEADSET_TO_WORLD @ headset_pose_xyz
        headset_pose_rotation = R.from_quat(headset_pose_quat).as_matrix()
        headset_pose_rotation = R_HEADSET_TO_WORLD @ headset_pose_rotation @ R_HEADSET_TO_WORLD.T

        # Calculate the delta between the controller and headset positions
        xr_pose_xyz_delta = xr_pose_xyz - headset_pose_xyz

        # Calculate the yaw of the headset
        R_headset_to_world = R.from_matrix(headset_pose_rotation)
        headset_pose_yaw = R_headset_to_world.as_euler("xyz")[2]  # Extract yaw (Z-axis rotation)
        inverse_yaw_rotation = R.from_euler("z", -headset_pose_yaw).as_matrix()

        # Align with headset yaw to controller position delta and rotation
        xr_pose_xyz_delta_compensated = inverse_yaw_rotation @ xr_pose_xyz_delta
        xr_pose_rotation_compensated = inverse_yaw_rotation @ xr_pose_rotation

        xr_pose_T = np.eye(4)
        xr_pose_T[:3, :3] = xr_pose_rotation_compensated
        xr_pose_T[:3, 3] = xr_pose_xyz_delta_compensated
        return xr_pose_T

    def _joystick_navigation_command(
        self,
        left_joystick,
        right_joystick,
    ) -> np.ndarray:
        """Map PICO sticks to x/y/yaw, keeping only one left-stick axis."""
        fwd_bwd_input = np.clip(float(left_joystick[1]), -1.0, 1.0)
        strafe_input = np.clip(-float(left_joystick[0]), -1.0, 1.0)
        # Do not send diagonal translation to the robot.  Select the dominant
        # left-stick axis; a tie intentionally prefers forward/backward.
        if abs(fwd_bwd_input) >= abs(strafe_input):
            strafe_input = 0.0
        else:
            fwd_bwd_input = 0.0
        yaw_input = np.clip(-float(right_joystick[0]), -1.0, 1.0)
        return np.asarray(
            [fwd_bwd_input, strafe_input, yaw_input],
            dtype=np.float64,
        ) * self.navigation_range

    def _generate_finger_data(self, pico_data, hand):
        """Encode Pico inputs as G1 hand gestures.

        Left hand uses left trigger plus right grip to select open, pressed,
        grip, or grip_pressed. The combined state is debounced for 100 ms so
        a short intermediate right-grip-only sample cannot trigger state 3.
        """
        fingertips = np.zeros([25, 4, 4])
        thumb_tip = 4
        index_tip = 9
        middle_tip = 14
        ring_tip = 19
        fingertips[thumb_tip, 0, 3] = 1.0

        if hand == "left":
            left_trigger, right_grip = self._left_hand_state_debouncer.update(
                pico_data["left_trigger"], pico_data["right_grip"]
            )
            if right_grip > 0.5 and left_trigger > 0.5:
                fingertips[middle_tip, 0, 3] = 1.0  # grip_pressed
            elif right_grip > 0.5:
                fingertips[ring_tip, 0, 3] = 1.0  # grip
            elif left_trigger > 0.5:
                fingertips[index_tip, 0, 3] = 1.0  # pressed
        elif pico_data["right_trigger"] > 0.5:
            # Keep the existing right-hand trigger behavior unchanged.
            fingertips[middle_tip, 0, 3] = 1.0

        return fingertips


if __name__ == "__main__":
    # from decoupled_wbc.control.utils.debugger import wait_for_debugger
    # wait_for_debugger()

    streamer = PicoStreamer()
    streamer.start_streaming()
    while True:
        raw_data = streamer.get()
        print(
            f"left_wrist: {raw_data.ik_data['left_wrist']}, right_wrist: {raw_data.ik_data['right_wrist']}"
        )
        time.sleep(0.1)
