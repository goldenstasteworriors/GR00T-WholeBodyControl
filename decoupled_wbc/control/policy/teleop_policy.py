from contextlib import contextmanager
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from decoupled_wbc.control.base.policy import Policy
from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.teleop.teleop_streamer import TeleopStreamer


class TeleopPolicy(Policy):
    """
    Robot-agnostic teleop policy.
    Clean separation: IK processing vs command passing.
    All robot-specific properties are abstracted through robot_model and hand_ik_solvers.
    """

    def __init__(
        self,
        body_control_device: str,
        hand_control_device: str,
        robot_model: RobotModel,
        retargeting_ik: TeleopRetargetingIK,
        body_streamer_ip: str = "192.168.?.?",
        body_streamer_keyword: str = "shoulder",
        enable_real_device: bool = True,
        replay_data_path: Optional[str] = None,
        replay_speed: float = 1.0,
        wait_for_activation: int = 5,
        activate_keyboard_listener: bool = True,
        return_to_initial_duration: float = 2.0,
        activation_hold_duration: float = 0.5,
    ):
        if activate_keyboard_listener:
            from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber

            self.keyboard_listener = KeyboardListenerSubscriber()
        else:
            self.keyboard_listener = None

        self.wait_for_activation = wait_for_activation

        self.teleop_streamer = TeleopStreamer(
            robot_model=robot_model,
            body_control_device=body_control_device,
            hand_control_device=hand_control_device,
            enable_real_device=enable_real_device,
            body_streamer_ip=body_streamer_ip,
            body_streamer_keyword=body_streamer_keyword,
            replay_data_path=replay_data_path,
            replay_speed=replay_speed,
        )
        self.robot_model = robot_model
        self.retargeting_ik = retargeting_ik
        self.is_active = False
        self.return_to_initial_duration = return_to_initial_duration
        self.activation_hold_duration = activation_hold_duration
        self._teleop_state = "paused"
        self._return_deadline: Optional[float] = None
        self._activation_deadline: Optional[float] = None
        self._initial_upper_body_pose = robot_model.get_initial_upper_body_pose().copy()

        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

    def set_goal(self, goal: dict[str, any]):
        # The current teleop policy doesn't take higher level commands yet.
        pass

    def get_action(self) -> dict[str, any]:
        # Get structured data
        streamer_output = self.teleop_streamer.get_streamer_data()

        # Handle activation using teleop_data commands
        self.check_activation(
            streamer_output.teleop_data, wait_for_activation=self.wait_for_activation
        )

        action = {}

        # Process streamer data only after the arming hold has completed.  During
        # pause and return-to-initial phases the upper-body target is explicitly
        # held at the safe initial pose instead of retaining a stale IK target.
        if self.is_active and streamer_output.ik_data:
            body_data = streamer_output.ik_data["body_data"]
            left_hand_data = streamer_output.ik_data["left_hand_data"]
            right_hand_data = streamer_output.ik_data["right_hand_data"]

            left_wrist_name = self.robot_model.supplemental_info.hand_frame_names["left"]
            right_wrist_name = self.robot_model.supplemental_info.hand_frame_names["right"]
            self.latest_left_wrist_data = body_data[left_wrist_name]
            self.latest_right_wrist_data = body_data[right_wrist_name]
            self.latest_left_fingers_data = left_hand_data
            self.latest_right_fingers_data = right_hand_data

            # TODO: This stores the same data again
            ik_data = {
                "body_data": body_data,
                "left_hand_data": left_hand_data,
                "right_hand_data": right_hand_data,
            }
            action["ik_data"] = ik_data

        # Wrist poses (pos and quat)
        # TODO: This stores the same wrist poses in two different formats
        left_wrist_matrix = self.latest_left_wrist_data
        right_wrist_matrix = self.latest_right_wrist_data
        left_wrist_pose = np.concatenate(
            [
                left_wrist_matrix[:3, 3],
                R.from_matrix(left_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )
        right_wrist_pose = np.concatenate(
            [
                right_wrist_matrix[:3, 3],
                R.from_matrix(right_wrist_matrix[:3, :3]).as_quat(scalar_first=True),
            ]
        )

        # Combine IK results with control commands (no teleop_data commands)
        action.update(
            {
                "left_wrist": self.latest_left_wrist_data,
                "right_wrist": self.latest_right_wrist_data,
                "left_fingers": self.latest_left_fingers_data,
                "right_fingers": self.latest_right_fingers_data,
                "wrist_pose": np.concatenate([left_wrist_pose, right_wrist_pose]),
                **streamer_output.control_data,  # Only control & data collection commands pass through
                **streamer_output.data_collection_data,
            }
        )

        # Run retargeting IK only in active teleoperation.  A return command is
        # published with one fixed deadline so the control-side interpolator can
        # generate a continuous, bounded trajectory to the initial pose.
        if self._teleop_state == "returning":
            action["target_upper_body_pose"] = self._initial_upper_body_pose.copy()
            action["target_time"] = self._return_deadline
        elif self._teleop_state in {"paused", "arming"}:
            action["target_upper_body_pose"] = self._initial_upper_body_pose.copy()
        else:
            if "ik_data" in action:
                self.retargeting_ik.set_goal(action["ik_data"])
            action["target_upper_body_pose"] = self.retargeting_ik.get_action()

        if self._teleop_state in {"returning", "paused", "arming"}:
            # A clutch/return operation must not preserve a joystick command
            # that was held when A+X was pressed.
            action["navigate_cmd"] = np.zeros(3)

        return action

    def close(self) -> bool:
        self.teleop_streamer.stop_streaming()
        return True

    def check_activation(self, teleop_data: dict, wait_for_activation: int = 5):
        """Activation logic only looks at teleop data commands"""
        key = self.keyboard_listener.read_msg() if self.keyboard_listener else ""
        toggle_activation_by_keyboard = key == "l"
        reset_teleop_policy_by_keyboard = key == "k"
        toggle_activation_by_teleop = teleop_data.get("toggle_activation", False)
        set_active = teleop_data.get("set_active", None)

        if reset_teleop_policy_by_keyboard:
            print("Resetting teleop policy")
            self.reset()

        now = time.monotonic()

        if self._teleop_state == "returning":
            if now >= self._return_deadline:
                self._enter_paused()
            return

        if self._teleop_state == "arming":
            if now >= self._activation_deadline:
                self._teleop_state = "active"
                self.is_active = True
                self._activation_deadline = None
                print("Teleop policy active")
            return

        # A+B+X+Y sends an explicit set_active=False together with the
        # lower-body emergency-stop request.  Preserve the existing last upper
        # body target for that separate emergency path; the A+X clutch path
        # below is the only path that performs a controlled return-to-initial.
        if set_active is False:
            self.is_active = False
            self._teleop_state = "emergency_paused"
            self._return_deadline = None
            self._activation_deadline = None
            print("Teleop stopped by emergency request")
            return

        requested_active = None
        if set_active is not None:
            requested_active = bool(set_active)
        elif toggle_activation_by_keyboard or toggle_activation_by_teleop:
            requested_active = self._teleop_state in {"paused", "emergency_paused"}

        if requested_active is None:
            return

        if requested_active:
            self._arm_teleop(now)
        elif self._teleop_state == "active":
            self._start_return_to_initial(now)

    def _start_return_to_initial(self, now: float) -> None:
        """Safely leave teleoperation through a time-bounded initial-pose return."""
        self.is_active = False
        self._teleop_state = "returning"
        self._return_deadline = now + self.return_to_initial_duration
        print(
            "Pausing teleop: returning upper body to the initial pose "
            f"over {self.return_to_initial_duration:.1f}s"
        )

    def _enter_paused(self) -> None:
        self.is_active = False
        self._teleop_state = "paused"
        self._return_deadline = None
        self.retargeting_ik.reset()
        print("Teleop paused at the initial pose")

    def _arm_teleop(self, now: float) -> None:
        """Calibrate at the already-held initial pose before accepting IK motion."""
        self.is_active = False
        self.retargeting_ik.reset()
        self.teleop_streamer.calibrate()
        self._teleop_state = "arming"
        self._activation_deadline = now + self.activation_hold_duration
        print(
            "Teleop calibrated at the initial pose; holding for "
            f"{self.activation_hold_duration:.1f}s before activation"
        )

    @contextmanager
    def activate(self):
        try:
            yield self
        finally:
            self.close()

    def handle_keyboard_button(self, keycode):
        """
        Handle keyboard input with proper state toggle.
        """
        if keycode == "l":
            # Toggle start state
            self.is_active = not self.is_active
            # Reset initialization when stopping
            if not self.is_active:
                self._initialized = False
        if keycode == "k":
            print("Resetting teleop policy")
            self.reset()

    def activate_policy(self, wait_for_activation: int = 5):
        """activate the teleop policy"""
        self.is_active = False
        self.check_activation(
            teleop_data={"toggle_activation": True}, wait_for_activation=wait_for_activation
        )

    def reset(self, wait_for_activation: int = 5, auto_activate: bool = False):
        """Reset the teleop policy to the initial state, and re-activate it."""
        self.teleop_streamer.reset()
        self.retargeting_ik.reset()
        self.is_active = False
        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

        if auto_activate:
            self.activate_policy(wait_for_activation)
