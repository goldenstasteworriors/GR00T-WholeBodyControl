import json
from unittest.mock import Mock

from decoupled_wbc.control.envs.g1.utils.command_sender import UnitreeLocoArmCommandSender


def _make_sender() -> UnitreeLocoArmCommandSender:
    sender = UnitreeLocoArmCommandSender.__new__(UnitreeLocoArmCommandSender)
    sender.loco = Mock()
    sender._user_control_selected = False
    return sender


def _make_attach_sender() -> UnitreeLocoArmCommandSender:
    sender = _make_sender()
    sender.active = False
    sender._activation_requested = False
    sender._active_fsm_ids = {500, 501, 802}
    sender._query_fsm_id = Mock(return_value=501)
    sender._robot_stability = Mock(return_value=(True, "stable"))
    sender._release_arms = Mock()
    sender._stop_move = Mock()
    return sender


def test_switch_to_user_control_calls_firmware_rpc_without_sdk_method():
    sender = _make_sender()
    sender.loco._Call.return_value = (0, "")

    sender._switch_to_user_control()

    sender.loco._Call.assert_called_once_with(
        sender.SWITCH_TO_USER_CTRL_API_ID,
        json.dumps({"data": False}),
    )
    assert sender._user_control_selected


def test_switch_to_internal_passive_control_returns_firmware_authority():
    sender = _make_sender()
    sender._user_control_selected = True
    sender.loco._Call.return_value = (0, "")

    sender._switch_to_internal_passive_control()

    sender.loco._Call.assert_called_once_with(
        sender.SWITCH_TO_INTERNAL_CTRL_API_ID,
        json.dumps({"data": sender.INTERNAL_CTRL_PASSIVE_MODE}),
    )
    assert not sender._user_control_selected


def test_switch_to_internal_control_is_noop_without_user_authority():
    sender = _make_sender()

    sender._switch_to_internal_passive_control()

    sender.loco._Call.assert_not_called()


def test_switch_to_user_control_propagates_firmware_error():
    sender = _make_sender()
    sender.loco._Call.return_value = (3201, "")

    try:
        sender._switch_to_user_control()
    except RuntimeError as exc:
        assert "SwitchToUserCtrl failed with code 3201" in str(exc)
    else:
        raise AssertionError("firmware RPC failure must abort activation")

    assert not sender._user_control_selected


def test_attach_existing_regular_mode_does_not_request_fsm_transition():
    sender = _make_attach_sender()

    sender.set_active(True)

    assert sender.active
    assert not sender._activation_requested
    assert sender._activation_stage == "active"
    assert sender._last_fsm_id == 501
    sender._release_arms.assert_called_once_with()
    sender._stop_move.assert_called_once_with()
    sender.loco._Call.assert_not_called()


def test_non_locomotion_fsm_falls_back_to_firmware_startup():
    sender = _make_attach_sender()
    sender._query_fsm_id.return_value = 4
    sender._activation_timeout = 15.0
    sender._stand_transition_seen = False
    sender._locomotion_zeroed = False
    sender._last_wait_log = 0.0
    sender._last_fsm_query = 0.0
    sender._last_fsm_id = None
    sender._switch_to_user_control = Mock()
    sender._set_fsm = Mock()
    sender._set_activation_stage = Mock()
    sender._damp_fsm_id = 1
    sender._stand_fsm_id = 4
    sender._start_fsm_id = 500

    sender.set_active(True)

    sender._switch_to_user_control.assert_called_once_with()
    sender._set_fsm.assert_called_once_with(1, "Damp")
