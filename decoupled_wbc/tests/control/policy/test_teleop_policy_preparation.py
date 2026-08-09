from unittest.mock import Mock

import numpy as np

from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy


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
