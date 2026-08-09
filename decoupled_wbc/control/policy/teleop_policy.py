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
        pico_vis_smpl: bool = False,
        wait_for_activation: int = 5,
        activate_keyboard_listener: bool = True,
        activation_hold_duration: float = 0.5,
        resume_max_joint_delta: float = 0.005,
        pre_activation_upper_body_pose: Optional[np.ndarray] = None,
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
            pico_vis_smpl=pico_vis_smpl,
        )
        self.robot_model = robot_model
        self.retargeting_ik = retargeting_ik
        self.is_active = False
        self.activation_hold_duration = activation_hold_duration
        self.resume_max_joint_delta = resume_max_joint_delta
        self._teleop_state = "paused"
        self._lower_body_policy_active = False
        self._activation_deadline: Optional[float] = None
        self._resume_ramp_deadline: Optional[float] = None
        self._latest_robot_q: Optional[np.ndarray] = None
        self._startup_reference_synced = False
        self._held_body_q = robot_model.default_body_pose.copy()
        initial_upper_body_pose = robot_model.get_initial_upper_body_pose().copy()
        if pre_activation_upper_body_pose is not None:
            pre_activation_upper_body_pose = np.asarray(
                pre_activation_upper_body_pose, dtype=np.float64
            )
            if pre_activation_upper_body_pose.shape != initial_upper_body_pose.shape:
                raise ValueError(
                    "pre_activation_upper_body_pose must match the upper-body shape "
                    f"{initial_upper_body_pose.shape}, got {pre_activation_upper_body_pose.shape}"
                )
            initial_upper_body_pose = pre_activation_upper_body_pose.copy()
        self._pre_activation_upper_body_pose = (
            None
            if pre_activation_upper_body_pose is None
            else pre_activation_upper_body_pose.copy()
        )
        self._held_upper_body_pose = initial_upper_body_pose
        self._last_safe_upper_target = self._held_upper_body_pose.copy()

        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

    def set_goal(self, goal: dict[str, any]):
        # The current teleop policy doesn't take higher level commands yet.
        pass

    def set_robot_state(self, robot_state: dict) -> None:
        """Store the newest control-loop observation for clutch hold/rebase."""
        q = robot_state.get("q")
        if q is None:
            return
        q = np.asarray(q, dtype=np.float64)
        if q.shape != self.robot_model.default_body_pose.shape:
            return
        self._latest_robot_q = q.copy()
        if not self._startup_reference_synced:
            upper_indices = self.robot_model.get_joint_group_indices("upper_body")
            self._held_body_q = q.copy()
            if self._pre_activation_upper_body_pose is None:
                self._held_upper_body_pose = q[upper_indices].copy()
            else:
                self._held_upper_body_pose = self._pre_activation_upper_body_pose.copy()
                self._held_body_q[upper_indices] = self._held_upper_body_pose
            self._last_safe_upper_target = self._held_upper_body_pose.copy()
            self.retargeting_ik.reset(reference_full_q=self._held_body_q)
            self._startup_reference_synced = True
            if self._pre_activation_upper_body_pose is None:
                print("Teleop startup reference synchronized from robot state")
            else:
                print("Teleop holding configured upper-body preparation pose before A+X")

    def set_lower_body_policy_active(self, active: bool) -> None:
        """Synchronize the confirmed lower-body state and enforce activation ordering."""
        active = bool(active)
        was_active = self._lower_body_policy_active
        self._lower_body_policy_active = active
        self.teleop_streamer.set_lower_body_policy_active(active)

        if was_active and not active and self._teleop_state in {"active", "arming"}:
            self.is_active = False
            self._teleop_state = "emergency_paused"
            self._activation_deadline = None
            self._resume_ramp_deadline = None
            print("Upper-body teleop stopped because the lower-body policy is inactive")

    def get_action(self) -> dict[str, any]:
        # Get structured data
        streamer_output = self.teleop_streamer.get_streamer_data()

        # Handle activation using teleop_data commands
        self.check_activation(
            streamer_output.teleop_data, wait_for_activation=self.wait_for_activation
        )

        action = {}

        # Process streamer data only after the arming hold has completed. During
        # clutch pause the upper-body target is the last command sent to the
        # controller, so the pause itself never changes the commanded target.
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

        # A pause freezes the measured upper-body pose. On resume, controller
        # coordinates were rebased against this same pose before IK is enabled.
        if self._teleop_state in {"paused", "arming"}:
            action["target_upper_body_pose"] = self._held_upper_body_pose.copy()
        else:
            if "ik_data" in action:
                self.retargeting_ik.set_goal(action["ik_data"])
            raw_target = self.retargeting_ik.get_action()
            target = raw_target
            if self._resume_ramp_deadline is not None:
                target = np.clip(
                    target,
                    self._last_safe_upper_target - self.resume_max_joint_delta,
                    self._last_safe_upper_target + self.resume_max_joint_delta,
                )
                # Do not remove the limiter merely because the initial hold
                # elapsed: doing so could release a large latent IK delta in a
                # single control cycle.  Keep rate-limiting until the IK result
                # has caught up with the commanded target.
                if (
                    time.monotonic() >= self._resume_ramp_deadline
                    and np.allclose(
                        raw_target,
                        target,
                        atol=self.resume_max_joint_delta,
                        rtol=0.0,
                    )
                ):
                    self._resume_ramp_deadline = None
            action["target_upper_body_pose"] = target
            self._last_safe_upper_target = target.copy()

        if self._teleop_state in {"paused", "arming"}:
            # A clutch operation must not preserve a joystick command that was
            # held when A+X was pressed.
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

        if self._teleop_state == "arming":
            if now >= self._activation_deadline:
                self._teleop_state = "active"
                self.is_active = True
                self._activation_deadline = None
                self._resume_ramp_deadline = now + self.activation_hold_duration
                print("Teleop policy active")
            return

        # A+B+X+Y sends an explicit set_active=False together with the
        # lower-body emergency-stop request.  Preserve the existing last upper
        # body target for that separate emergency path; the A+X clutch path
        # below is the only path that performs a controlled return-to-initial.
        if set_active is False:
            self.is_active = False
            self._teleop_state = "emergency_paused"
            self._activation_deadline = None
            self._resume_ramp_deadline = None
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
            if not self._lower_body_policy_active:
                print(
                    "Ignoring A+X activation: lower-body policy has not confirmed startup"
                )
                return
            self._arm_teleop(now)
        elif self._teleop_state == "active":
            self._enter_clutch_pause()

    def _enter_clutch_pause(self) -> None:
        """Hold the last controller target and use it as the next IK reference."""
        self.is_active = False
        self._teleop_state = "paused"
        upper_indices = self.robot_model.get_joint_group_indices("upper_body")
        if self._latest_robot_q is not None:
            self._held_body_q = self._latest_robot_q.copy()
        else:
            self._held_body_q = self.robot_model.default_body_pose.copy()
            print("WARNING: no fresh robot state while pausing; using nominal lower-body reference")

        # Feedback q can lag the interpolated command slightly, especially at
        # the wrists under gravity.  Replacing the command with feedback here
        # creates the visible downward step reported at pause.  Retain feedback
        # for the non-upper-body reference, but always hold the last published
        # upper-body command exactly.
        self._held_body_q[upper_indices] = self._last_safe_upper_target
        self._held_upper_body_pose = self._last_safe_upper_target.copy()
        self._resume_ramp_deadline = None
        self.retargeting_ik.reset(reference_full_q=self._held_body_q)
        print("Teleop paused: holding the last commanded upper-body pose")

    def _arm_teleop(self, now: float) -> None:
        """Rebase PICO at the held robot pose before accepting IK motion."""
        self.is_active = False
        self.retargeting_ik.reset(reference_full_q=self._held_body_q)
        self.teleop_streamer.calibrate(reference_body_q=self._held_body_q)
        self._teleop_state = "arming"
        self._activation_deadline = now + self.activation_hold_duration
        print(
            "Teleop calibrated at the held pose; holding for "
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
            self.check_activation({"toggle_activation": True})
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
        self._teleop_state = "paused"
        self._latest_robot_q = None
        self._startup_reference_synced = False
        self._held_body_q = self.robot_model.default_body_pose.copy()
        if self._pre_activation_upper_body_pose is None:
            self._held_upper_body_pose = self.robot_model.get_initial_upper_body_pose().copy()
        else:
            self._held_upper_body_pose = self._pre_activation_upper_body_pose.copy()
            upper_indices = self.robot_model.get_joint_group_indices("upper_body")
            self._held_body_q[upper_indices] = self._held_upper_body_pose
        self._last_safe_upper_target = self._held_upper_body_pose.copy()
        self._activation_deadline = None
        self._resume_ramp_deadline = None
        self.latest_left_wrist_data = np.eye(4)
        self.latest_right_wrist_data = np.eye(4)
        self.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
        self.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

        if auto_activate:
            self.activate_policy(wait_for_activation)
