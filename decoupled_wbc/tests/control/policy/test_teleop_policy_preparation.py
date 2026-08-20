from unittest.mock import Mock

import numpy as np

from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.policy.wbc_policy_factory import (
    get_unitree_loco_startup_body_pose,
)
from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer


def test_unitree_startup_pose_raises_both_elbows_equally():
    robot_model = Mock()
    robot_model.initial_body_pose = np.array([0.1, 0.2, 0.3, 0.4])
    robot_model.dof_index.side_effect = {
        "left_elbow_joint": 1,
        "right_elbow_joint": 3,
    }.__getitem__

    startup_pose = get_unitree_loco_startup_body_pose(robot_model, -0.2617993877991494)

    np.testing.assert_allclose(
        startup_pose,
        [0.1, -0.2617993877991494, 0.3, -0.2617993877991494],
    )
    np.testing.assert_allclose(robot_model.initial_body_pose, [0.1, 0.2, 0.3, 0.4])


def test_unitree_left_only_startup_pose_leaves_right_elbow_at_initial_angle():
    robot_model = Mock()
    robot_model.initial_body_pose = np.array([0.1, 0.2, 0.3, 0.4])
    robot_model.dof_index.side_effect = {
        "left_elbow_joint": 1,
        "right_elbow_joint": 3,
    }.__getitem__

    startup_pose = get_unitree_loco_startup_body_pose(
        robot_model,
        -0.2617993877991494,
        raise_right_arm=False,
    )

    np.testing.assert_allclose(startup_pose, [0.1, -0.2617993877991494, 0.3, 0.4])


def test_pico_left_stick_tie_prefers_forward_backward_axis():
    streamer = PicoStreamer.__new__(PicoStreamer)
    streamer.navigation_range = 1.0

    command = streamer._joystick_navigation_command(
        left_joystick=[-1.0, 1.0],
        right_joystick=[-1.0, 0.0],
    )

    np.testing.assert_allclose(command, [1.0, 0.0, 1.0])


def test_pico_navigation_range_parameter_scales_and_clips_all_axes():
    streamer = PicoStreamer.__new__(PicoStreamer)
    streamer.navigation_range = 0.4

    command = streamer._joystick_navigation_command(
        left_joystick=[2.0, -2.0],
        right_joystick=[2.0, 0.0],
    )

    np.testing.assert_allclose(command, [-0.4, 0.0, -0.4])


def test_pico_left_stick_uses_only_dominant_strafe_axis():
    streamer = PicoStreamer.__new__(PicoStreamer)
    streamer.navigation_range = 1.0

    command = streamer._joystick_navigation_command(
        left_joystick=[-0.9, -0.2],
        right_joystick=[0.0, 0.0],
    )

    np.testing.assert_allclose(command, [0.0, 0.9, 0.0])


def test_pico_raw_right_trigger_is_forwarded_as_control_data():
    streamer = PicoStreamer.__new__(PicoStreamer)
    streamer._publish_smpl_visualization_frame = Mock()
    streamer._process_xr_pose = Mock(return_value=np.eye(4))
    streamer._joystick_navigation_command = Mock(return_value=np.zeros(3))
    streamer._generate_finger_data = Mock(return_value=np.zeros((25, 4, 4)))
    streamer._button_sampler = Mock()
    streamer._button_sampler.consume_events.return_value = []
    streamer._button_events_consumed = {
        "ax": 0,
        "by": 0,
        "start_stop": 0,
        "data_collection": 0,
        "data_abort": 0,
        "emergency_stop": 0,
    }
    streamer._data_collection_event_id = 0
    streamer._data_abort_event_id = 0
    streamer.combo_suppression_active = False
    streamer.control_enabled = False
    streamer.current_base_height = 0.5
    pico_data = {
        "left_pose": np.zeros(7),
        "right_pose": np.zeros(7),
        "head_pose": np.zeros(7),
        "left_joystick": np.zeros(2),
        "right_joystick": np.zeros(2),
        "A": False,
        "B": False,
        "X": False,
        "Y": False,
        "left_grip": 0.0,
        "right_trigger": 0.8,
    }

    output = streamer._generate_unified_raw_data(pico_data)

    assert output.control_data["right_trigger"] == 0.8


def test_configured_preparation_pose_is_not_overwritten_by_startup_feedback():
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.robot_model = Mock()
    policy.robot_model.default_body_pose = np.zeros(6)
    policy.robot_model.get_joint_group_indices.return_value = np.array([2, 4])
    policy.retargeting_ik = Mock()
    policy._latest_robot_q = None
    policy._startup_reference_synced = False
    policy._pre_activation_upper_body_pose = np.array([0.4, -0.3])
    policy._held_body_q = np.zeros(6)
    policy._held_upper_body_pose = np.zeros(2)
    policy._last_safe_upper_target = np.zeros(2)

    measured_q = np.array([1.0, 2.0, 0.1, 3.0, 0.2, 4.0])
    policy.set_robot_state({"q": measured_q})

    np.testing.assert_allclose(policy._latest_robot_q, measured_q)
    np.testing.assert_allclose(policy._held_upper_body_pose, [0.4, -0.3])
    np.testing.assert_allclose(policy._held_body_q, [1.0, 2.0, 0.4, 3.0, -0.3, 4.0])
    policy.retargeting_ik.reset.assert_called_once()
    np.testing.assert_allclose(
        policy.retargeting_ik.reset.call_args.kwargs["reference_full_q"],
        policy._held_body_q,
    )


def test_default_startup_behavior_still_holds_measured_upper_body_pose():
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.robot_model = Mock()
    policy.robot_model.default_body_pose = np.zeros(5)
    policy.robot_model.get_joint_group_indices.return_value = np.array([1, 3])
    policy.retargeting_ik = Mock()
    policy._latest_robot_q = None
    policy._startup_reference_synced = False
    policy._pre_activation_upper_body_pose = None
    policy._held_body_q = np.zeros(5)
    policy._held_upper_body_pose = np.zeros(2)
    policy._last_safe_upper_target = np.zeros(2)

    measured_q = np.array([0.0, 0.2, 0.0, -0.1, 0.0])
    policy.set_robot_state({"q": measured_q})

    np.testing.assert_allclose(policy._held_upper_body_pose, [0.2, -0.1])
    np.testing.assert_allclose(policy._held_body_q, measured_q)


def test_ax_pause_holds_last_command_instead_of_releasing_arms():
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.robot_model = Mock()
    policy.robot_model.get_joint_group_indices.return_value = np.array([1, 3])
    policy.retargeting_ik = Mock()
    policy.is_active = True
    policy._teleop_state = "active"
    policy._latest_robot_q = np.array([1.0, 0.1, 2.0, 0.2, 3.0])
    policy._last_safe_upper_target = np.array([0.6, -0.5])
    policy._held_body_q = np.zeros(5)
    policy._activation_deadline = 1.0
    policy._resume_ramp_deadline = 2.0

    policy._enter_clutch_pause()

    assert not policy.is_active
    assert policy._teleop_state == "paused"
    np.testing.assert_allclose(policy._held_upper_body_pose, [0.6, -0.5])
    np.testing.assert_allclose(policy._held_body_q, [1.0, 0.6, 2.0, -0.5, 3.0])
    policy.retargeting_ik.reset.assert_called_once()


def test_ax_resume_recalibrates_pico_at_held_pose_before_following():
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.retargeting_ik = Mock()
    policy.teleop_streamer = Mock()
    policy._held_body_q = np.array([1.0, 0.6, 2.0, -0.5, 3.0])
    policy.activation_hold_duration = 0.5
    policy.is_active = False

    policy._arm_teleop(now=10.0)

    assert not policy.is_active
    assert policy._teleop_state == "arming"
    assert policy._activation_deadline == 10.5
    policy.retargeting_ik.reset.assert_called_once()
    policy.teleop_streamer.calibrate.assert_called_once()
    np.testing.assert_allclose(
        policy.teleop_streamer.calibrate.call_args.kwargs["reference_body_q"],
        policy._held_body_q,
    )


def test_paused_upper_body_does_not_zero_lower_body_navigation():
    policy = TeleopPolicy.__new__(TeleopPolicy)
    policy.teleop_streamer = Mock()
    streamer_output = Mock()
    streamer_output.teleop_data = {}
    streamer_output.ik_data = None
    streamer_output.control_data = {
        "navigate_cmd": np.array([0.2, -0.1, 0.3]),
        "right_trigger": 0.75,
    }
    streamer_output.data_collection_data = {}
    policy.teleop_streamer.get_streamer_data.return_value = streamer_output
    policy.check_activation = Mock()
    policy.wait_for_activation = 5
    policy.is_active = False
    policy._teleop_state = "paused"
    policy._held_upper_body_pose = np.array([0.4, -0.3])
    policy.latest_left_wrist_data = np.eye(4)
    policy.latest_right_wrist_data = np.eye(4)
    policy.latest_left_fingers_data = {"position": np.zeros((25, 4, 4))}
    policy.latest_right_fingers_data = {"position": np.zeros((25, 4, 4))}

    action = policy.get_action()

    np.testing.assert_allclose(action["navigate_cmd"], [0.2, -0.1, 0.3])
    assert action["right_trigger"] == 0.75
    np.testing.assert_allclose(action["target_upper_body_pose"], [0.4, -0.3])
