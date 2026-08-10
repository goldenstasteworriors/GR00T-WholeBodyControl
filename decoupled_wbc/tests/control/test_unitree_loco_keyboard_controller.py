import numpy as np

from decoupled_wbc.control.policy.unitree_loco_keyboard_controller import (
    UnitreeLocoKeyboardController,
)


def test_space_requests_emergency_stop_when_full_keyboard_control_is_disabled():
    controller = UnitreeLocoKeyboardController(full_control_enabled=False)

    controller.handle_keyboard_button("space")

    assert controller.get_control_goal(lower_body_active=True) == {
        "emergency_stop": True,
        "set_policy_action": False,
    }
    assert controller.get_control_goal(lower_body_active=False) == {}


def test_pico_mode_ignores_non_emergency_keyboard_controls():
    controller = UnitreeLocoKeyboardController(full_control_enabled=False)

    for key in ("g", "w", "s", "a", "d", "q", "e", "z"):
        controller.handle_keyboard_button(key)

    assert controller.get_control_goal(lower_body_active=False) == {}


def test_full_keyboard_control_keeps_start_and_navigation_behavior():
    controller = UnitreeLocoKeyboardController(full_control_enabled=True)

    controller.handle_keyboard_button("g")
    start_goal = controller.get_control_goal(lower_body_active=False)

    assert start_goal["set_policy_action"] is True
    np.testing.assert_array_equal(start_goal["navigate_cmd"], np.zeros(3))
