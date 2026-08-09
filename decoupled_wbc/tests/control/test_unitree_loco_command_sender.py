import json
from unittest.mock import Mock

from decoupled_wbc.control.envs.g1.utils.command_sender import UnitreeLocoArmCommandSender


def _make_sender() -> UnitreeLocoArmCommandSender:
    sender = UnitreeLocoArmCommandSender.__new__(UnitreeLocoArmCommandSender)
    sender.loco = Mock()
    sender._user_control_selected = False
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
