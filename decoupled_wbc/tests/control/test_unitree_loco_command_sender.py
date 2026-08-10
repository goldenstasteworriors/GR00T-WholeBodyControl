from types import SimpleNamespace
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
    sender._latest_waist_q = None
    sender._held_waist_q = None
    sender._release_arms = Mock()
    sender._velocity_period = 0.1
    sender._last_velocity_send = 0.0
    sender._max_linear_velocity = 0.05
    sender._max_angular_velocity = 0.1
    sender._navigation_enabled = False
    sender._arm_control_enabled = False
    sender._weight_ramp_duration = 2.0
    sender._arm_preparing = False
    sender._arm_preparation_started = 0.0
    sender._arm_preparation_complete = False
    sender._waist_kp = 60.0
    sender._waist_kd = 1.5
    return sender


def _configure_upper_body_output(sender: UnitreeLocoArmCommandSender) -> None:
    sender.config = {
        "MOTOR2JOINT": {index: index for index in sender.ARM_MOTOR_INDICES},
        "MOTOR_KP": [0.0] * 30,
        "MOTOR_KD": [0.0] * 30,
    }
    sender.low_cmd = SimpleNamespace(
        motor_cmd=[
            SimpleNamespace(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
            for _ in range(30)
        ],
        crc=0,
    )
    sender.crc = Mock()
    sender.crc.Crc.return_value = 123
    sender.publisher = Mock()


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


def test_operator_ready_requires_preparation_before_active_locomotion():
    sender = _make_sender()
    sender.active = True
    sender._arm_control_enabled = True

    assert not sender.operator_ready()

    sender._arm_preparation_complete = True
    assert sender.operator_ready()


def test_arm_output_is_enabled_during_locked_standing_preparation():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True

    assert sender._arm_output_active()
    assert not sender.active


def test_arm_output_is_disabled_before_lower_body_startup():
    sender = _make_sender()
    sender._arm_control_enabled = True

    assert not sender._arm_output_active()


def test_arm_sdk_holds_captured_waist_with_positive_gains():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True
    sender._arm_preparation_started = 10.0
    sender._held_waist_q = np.array([0.12, -0.08, 0.04])
    _configure_upper_body_output(sender)
    cmd_q = np.arange(29, dtype=np.float64) / 100.0
    cmd_dq = np.zeros(29)
    cmd_tau = np.zeros(29)

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=11.0,
    ):
        sender.send_command(cmd_q, cmd_dq, cmd_tau)

    for offset, motor_index in enumerate(sender.WAIST_MOTOR_INDICES):
        motor = sender.low_cmd.motor_cmd[motor_index]
        assert motor.q == sender._held_waist_q[offset]
        assert motor.dq == 0.0
        assert motor.tau == 0.0
        assert motor.kp == 60.0
        assert motor.kd == 1.5
    assert sender.low_cmd.motor_cmd[sender.ARM_WEIGHT_INDEX].q == 0.5
    sender.publisher.Write.assert_called_once_with(sender.low_cmd)


def test_stable_locked_standing_starts_arm_preparation_before_fsm_501():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._activation_requested = True
    sender._activation_stage = "wait_stand"
    sender._activation_deadline = 20.0
    sender._stage_started = 0.0
    sender._stand_duration = 5.0
    sender._stand_transition_seen = True
    sender._latest_waist_q = np.array([0.1, -0.05, 0.02])
    sender._query_fsm_id = Mock(return_value=4)
    sender._stable_for_required_duration = Mock(return_value=(True, "stable"))

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._activation_stage == "prepare_arms"
    assert sender._arm_preparing
    assert sender._arm_preparation_started == 10.0
    np.testing.assert_array_equal(sender._held_waist_q, sender._latest_waist_q)
    sender.loco.SetFsmId.assert_not_called()


def test_transitional_fsm_does_not_start_arm_preparation():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._activation_requested = True
    sender._activation_stage = "wait_stand"
    sender._activation_deadline = 20.0
    sender._stage_started = 0.0
    sender._stand_duration = 5.0
    sender._stand_transition_seen = True
    sender._query_fsm_id = Mock(return_value=3)
    sender._log_waiting = Mock()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._activation_stage == "wait_stand"
    assert not sender._arm_preparing
    sender.loco.SetFsmId.assert_not_called()


def test_fsm_501_is_requested_only_after_arms_prepare_and_body_resettles():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True
    sender._arm_preparation_started = 7.0
    sender._activation_requested = True
    sender._activation_stage = "prepare_arms"
    sender._activation_deadline = 20.0
    sender._query_fsm_id = Mock(return_value=4)
    sender._stable_for_required_duration = Mock(return_value=(True, "stable"))

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._arm_preparation_complete
    assert not sender._arm_preparing
    sender.loco.SetFsmId.assert_called_once_with(501)
    assert sender._activation_stage == "wait_locomotion"


def test_fsm_501_is_not_requested_while_arm_weight_is_still_ramping():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True
    sender._arm_preparation_started = 7.0
    sender._activation_requested = True
    sender._activation_stage = "prepare_arms"
    sender._activation_deadline = 20.0
    sender._query_fsm_id = Mock(return_value=4)
    sender._log_waiting = Mock()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=8.5,
    ):
        sender.update_status()

    sender.loco.SetFsmId.assert_not_called()
    assert sender._arm_preparing
    assert not sender._arm_preparation_complete


def test_emergency_stop_requests_damp_without_following_move():
    sender = _make_sender()
    sender.active = True

    sender.set_active(False, emergency=True)

    sender._release_arms.assert_called_once_with()
    sender.loco.SetFsmId.assert_called_once_with(1)
    sender.loco.SetVelocity.assert_not_called()
    assert not sender.active
    assert not sender._activation_requested
