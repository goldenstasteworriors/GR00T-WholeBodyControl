from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from decoupled_wbc.control.envs.g1.utils.command_sender import UnitreeLocoArmCommandSender


def _make_sender() -> UnitreeLocoArmCommandSender:
    sender = UnitreeLocoArmCommandSender.__new__(UnitreeLocoArmCommandSender)
    sender.loco = Mock()
    sender.loco.SetFsmId.return_value = 0
    sender.loco.SetVelocity.return_value = 0
    sender.robot_state = Mock()
    sender._system_service_name = "ai_sport"
    sender._system_service_start_timeout = 15.0
    sender.active = False
    sender._activation_requested = False
    sender._activation_stage = "idle"
    sender._activation_deadline = 0.0
    sender._activation_timeout = 25.0
    sender._damp_fsm_id = 1
    sender._stand_fsm_id = 4
    sender._start_fsm_id = 501
    sender._stand_transition_seen = False
    sender._last_wait_log = 0.0
    sender._last_fsm_query = 0.0
    sender._last_fsm_id = None
    sender._last_start_request = 0.0
    sender._start_retry_interval = 1.0
    sender._stable_since = None
    sender._latest_body_q = None
    sender._latest_leg_dq = None
    sender._latest_torso_quat = None
    sender._latest_robot_state_time = 0.0
    sender._state_timeout = 0.5
    sender._max_leg_velocity = 0.35
    sender._max_torso_tilt = 0.35
    sender._stability_duration = 0.5
    sender._release_arms = Mock()
    sender._velocity_period = 0.1
    sender._last_velocity_send = 0.0
    sender._consecutive_velocity_timeouts = 0
    sender._max_linear_velocity = 0.05
    sender._max_angular_velocity = 0.1
    sender._navigation_enabled = False
    sender._arm_control_enabled = False
    sender._weight_ramp_duration = 2.0
    sender._arm_preparing = False
    sender._arm_preparation_started = 0.0
    sender._arm_preparation_complete = False
    sender._arm_handoff_active = False
    sender._arm_handoff_q = None
    sender._arm_handoff_max_delta = 0.01
    return sender


def _service_state(status: int, name: str = "ai_sport") -> SimpleNamespace:
    return SimpleNamespace(name=name, status=status, protect=False)


def _configure_upper_body_output(sender: UnitreeLocoArmCommandSender) -> None:
    sender.config = {
        "MOTOR2JOINT": list(range(29)),
        "MOTOR_KP": [0.0] * 30,
        "MOTOR_KD": [0.0] * 30,
        "UNITREE_LEGGED_CONST": {"MODE_PR": 0, "MODE_MACHINE": 5},
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


def test_enabled_ai_sport_is_left_running_without_service_switch():
    sender = _make_sender()
    sender.robot_state.ServiceList.return_value = (0, [_service_state(0)])

    sender._ensure_system_service_enabled()

    sender.robot_state.ServiceSwitch.assert_not_called()


def test_disabled_ai_sport_is_enabled_once_and_never_stopped():
    sender = _make_sender()
    sender.robot_state.ServiceList.side_effect = [
        (0, [_service_state(1)]),
        (0, [_service_state(0)]),
    ]
    sender.robot_state.ServiceSwitch.return_value = 0

    sender._ensure_system_service_enabled()

    sender.robot_state.ServiceSwitch.assert_called_once_with("ai_sport", True)


def test_missing_ai_sport_fails_without_switching_another_service():
    sender = _make_sender()
    sender.robot_state.ServiceList.return_value = (0, [_service_state(0, "sport_mode")])

    try:
        sender._ensure_system_service_enabled()
    except RuntimeError as exc:
        assert "ai_sport" in str(exc)
        assert "sport_mode" in str(exc)
    else:
        raise AssertionError("missing ai_sport must fail")

    sender.robot_state.ServiceSwitch.assert_not_called()


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


def test_navigation_range_one_reaches_set_velocity_without_additional_clipping():
    sender = _make_sender()
    sender.active = True
    sender._navigation_enabled = True
    sender._max_linear_velocity = 1.0
    sender._max_angular_velocity = 1.0

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.send_velocity(np.array([1.0, -1.0, 1.0]))

    sender.loco.SetVelocity.assert_called_once_with(1.0, -1.0, 1.0, 0.25)


def test_single_set_velocity_timeout_is_retried_by_the_next_control_cycle():
    sender = _make_sender()
    sender.active = True
    sender._navigation_enabled = True
    sender.loco.SetVelocity.side_effect = [3104, 0]

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        side_effect=[10.0, 10.01, 11.0, 11.01],
    ):
        sender.send_velocity(np.array([0.1, 0.0, 0.0]))
        sender.send_velocity(np.array([0.1, 0.0, 0.0]))

    assert sender.loco.SetVelocity.call_count == 2
    assert sender._consecutive_velocity_timeouts == 0


def test_repeated_set_velocity_timeouts_remain_fatal():
    sender = _make_sender()
    sender.active = True
    sender._navigation_enabled = True
    sender.loco.SetVelocity.return_value = 3104

    try:
        with patch(
            "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
            side_effect=[10.0, 10.01, 11.0, 11.01, 12.0, 12.01],
        ):
            sender.send_velocity(np.array([0.1, 0.0, 0.0]))
            sender.send_velocity(np.array([0.1, 0.0, 0.0]))
            sender.send_velocity(np.array([0.1, 0.0, 0.0]))
    except RuntimeError as exc:
        assert "3104" in str(exc)
    else:
        raise AssertionError("three consecutive SetVelocity timeouts must fail")


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


def test_arm_output_is_enabled_only_during_post_locomotion_preparation():
    sender = _make_sender()
    sender._arm_control_enabled = True

    assert not sender._arm_output_active()

    sender._arm_preparing = True
    assert sender._arm_output_active()
    assert not sender.active


def test_arm_sdk_updates_only_dual_arms_after_full_pose_seed():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True
    sender._arm_preparation_started = 10.0
    _configure_upper_body_output(sender)
    waist_before = []
    for motor_index in range(12, 15):
        motor = sender.low_cmd.motor_cmd[motor_index]
        motor.q = 10.0 + motor_index
        motor.dq = 20.0 + motor_index
        motor.tau = 30.0 + motor_index
        motor.kp = 40.0 + motor_index
        motor.kd = 50.0 + motor_index
        waist_before.append((motor.q, motor.dq, motor.tau, motor.kp, motor.kd))
    cmd_q = np.arange(29, dtype=np.float64) / 100.0
    cmd_dq = np.zeros(29)
    cmd_tau = np.zeros(29)
    sender._latest_body_q = cmd_q.copy()
    sender._arm_handoff_q = cmd_q.copy()

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=11.0,
    ):
        sender.send_command(cmd_q, cmd_dq, cmd_tau)

    for expected, motor_index in zip(waist_before, range(12, 15)):
        motor = sender.low_cmd.motor_cmd[motor_index]
        assert (motor.q, motor.dq, motor.tau, motor.kp, motor.kd) == expected
    for motor_index in sender.ARM_MOTOR_INDICES:
        assert sender.low_cmd.motor_cmd[motor_index].q == cmd_q[motor_index]
    assert sender.low_cmd.motor_cmd[sender.ARM_WEIGHT_INDEX].q == 0.5
    sender.publisher.Write.assert_called_once_with(sender.low_cmd)


def test_stable_locked_standing_requests_fsm_501_before_arm_preparation():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._activation_requested = True
    sender._activation_stage = "wait_stand"
    sender._activation_deadline = 20.0
    sender._stage_started = 0.0
    sender._stand_duration = 5.0
    sender._stand_transition_seen = True
    sender._query_fsm_id = Mock(return_value=4)
    sender._stable_for_required_duration = Mock(return_value=(True, "stable"))

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._activation_stage == "wait_locomotion"
    assert not sender._arm_preparing
    sender.loco.SetFsmId.assert_called_once_with(501)


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


def test_confirmed_fsm_501_starts_arm_preparation_before_becoming_active():
    sender = _make_sender()
    _configure_upper_body_output(sender)
    sender._arm_control_enabled = True
    sender._activation_requested = True
    sender._activation_stage = "wait_locomotion"
    sender._activation_deadline = 20.0
    sender._query_fsm_id = Mock(return_value=501)
    sender._stable_for_required_duration = Mock(return_value=(True, "stable"))
    sender._last_velocity_send = 10.0
    sender._latest_body_q = np.arange(29, dtype=np.float64) / 100.0

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._arm_preparing
    assert not sender._arm_preparation_complete
    assert sender._activation_stage == "prepare_arms"
    assert not sender.active
    sender.loco.SetFsmId.assert_not_called()
    for motor_index in range(29):
        motor = sender.low_cmd.motor_cmd[motor_index]
        assert motor.mode == 1
        assert motor.q == sender._latest_body_q[motor_index]
    sender._stable_for_required_duration.assert_called_once_with(
        10.0, require_settled_legs=False
    )


def test_fsm_501_dynamic_leg_velocity_does_not_block_arm_preparation():
    sender = _make_sender()
    _configure_upper_body_output(sender)
    sender._arm_control_enabled = True
    sender._activation_requested = True
    sender._activation_stage = "wait_locomotion"
    sender._activation_deadline = 20.0
    sender._stability_duration = 0.0
    sender._latest_leg_dq = np.full(12, 9.0)
    sender._latest_body_q = np.arange(29, dtype=np.float64) / 100.0
    sender._latest_torso_quat = np.array([1.0, 0.0, 0.0, 0.0])
    sender._latest_robot_state_time = 10.0
    sender._last_velocity_send = 10.0
    sender._query_fsm_id = Mock(return_value=501)

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender._arm_preparing
    assert sender._activation_stage == "prepare_arms"


def test_arm_preparation_completes_in_fsm_501_without_requesting_another_fsm():
    sender = _make_sender()
    sender._arm_control_enabled = True
    sender._arm_preparing = True
    sender._arm_preparation_started = 7.0
    sender._activation_requested = True
    sender._activation_stage = "prepare_arms"
    sender._activation_deadline = 20.0
    sender._query_fsm_id = Mock(return_value=501)
    sender._stable_for_required_duration = Mock(return_value=(True, "stable"))
    sender._last_velocity_send = 10.0

    with patch(
        "decoupled_wbc.control.envs.g1.utils.command_sender.time.monotonic",
        return_value=10.0,
    ):
        sender.update_status()

    assert sender.active
    assert sender._arm_preparation_complete
    assert not sender._arm_preparing
    assert sender._activation_stage == "active"
    sender.loco.SetFsmId.assert_not_called()
    sender._stable_for_required_duration.assert_called_once_with(
        10.0, require_settled_legs=False
    )


def test_emergency_stop_requests_damp_without_following_move():
    sender = _make_sender()
    sender.active = True

    sender.set_active(False, emergency=True)

    sender._release_arms.assert_called_once_with()
    sender.loco.SetFsmId.assert_called_once_with(1)
    sender.loco.SetVelocity.assert_not_called()
    assert not sender.active
    assert not sender._activation_requested
    sender.robot_state.ServiceSwitch.assert_not_called()
