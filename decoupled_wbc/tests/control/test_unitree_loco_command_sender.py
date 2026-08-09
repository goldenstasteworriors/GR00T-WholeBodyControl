from unittest.mock import Mock, patch

import numpy as np

from decoupled_wbc.control.envs.g1.utils.command_sender import UnitreeLocoArmCommandSender


def _make_sender() -> UnitreeLocoArmCommandSender:
    sender = UnitreeLocoArmCommandSender.__new__(UnitreeLocoArmCommandSender)
    sender.loco = Mock()
    sender.loco.SetFsmId.return_value = 0
    sender.loco.SetVelocity.return_value = 0
    sender.active = False
    sender._activation_requested = False
    sender._activation_stage = "idle"
    sender._activation_deadline = 0.0
    sender._activation_timeout = 25.0
    sender._damp_fsm_id = 1
    sender._stand_fsm_id = 4
    sender._start_fsm_id = 501
    sender._stand_transition_seen = False
    sender._locomotion_zeroed = False
    sender._last_wait_log = 0.0
    sender._last_fsm_query = 0.0
    sender._last_fsm_id = None
    sender._last_start_request = 0.0
    sender._start_retry_interval = 1.0
    sender._stable_since = None
    sender._release_arms = Mock()
    sender._velocity_period = 0.1
    sender._last_velocity_send = 0.0
    sender._max_linear_velocity = 0.05
    sender._max_angular_velocity = 0.1
    sender._navigation_enabled = False
    return sender


def test_startup_begins_with_damp_without_private_control_authority_rpc():
    sender = _make_sender()
    sender._set_activation_stage = Mock()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.set_active(True)

    sender._release_arms.assert_called_once_with()
    sender.loco.SetFsmId.assert_called_once_with(1)
    sender.loco._Call.assert_not_called()
    assert sender._activation_requested


def test_motion_fsm_is_retried_until_501_is_observed():
    sender = _make_sender()
    sender._activation_requested = True
    sender._activation_stage = "wait_locomotion"
    sender._activation_deadline = 20.0
    sender._last_start_request = 8.0
    sender._query_fsm_id = Mock(return_value=4)
    sender._log_waiting = Mock()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    sender.loco.SetFsmId.assert_called_once_with(501)
    assert sender._last_start_request == 10.0
    assert not sender.active


def test_motion_mode_sends_only_zero_when_navigation_is_disabled():
    sender = _make_sender()
    sender.active = True

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.send_velocity(np.array([0.5, -0.5, 1.0]))

    sender.loco.SetVelocity.assert_called_once_with(0.0, 0.0, 0.0, 0.25)


def test_locked_standing_mode_never_sends_velocity():
    sender = _make_sender()
    sender.active = True
    sender._start_fsm_id = None

    sender.send_velocity(np.zeros(3))

    sender.loco.SetVelocity.assert_not_called()


def test_operator_ready_waits_for_arm_preparation_ramp():
    sender = _make_sender()
    sender.active = True
    sender._arm_control_enabled = True
    sender._weight_ramp_duration = 2.0
    sender._activation_time = 10.0
    sender._arm_ready_reported = False

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=11.9,
    ):
        assert not sender.operator_ready()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=12.0,
    ):
        assert sender.operator_ready()
        assert sender._arm_ready_reported


def test_emergency_stop_requests_damp_without_following_move():
    sender = _make_sender()
    sender.active = True

    sender.set_active(False, emergency=True)

    sender._release_arms.assert_called_once_with()
    sender.loco.SetFsmId.assert_called_once_with(1)
    sender.loco.SetVelocity.assert_not_called()
    assert not sender.active
    assert not sender._activation_requested
