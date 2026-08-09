from gear_sonic.utils.teleop.pico_buttons import PicoButtonEdgeDetector, PicoButtonState


def test_both_thumbstick_clicks_raise_one_emergency_event_per_press():
    detector = PicoButtonEdgeDetector()

    first = detector.update(
        PicoButtonState(left_axis_click=True, right_axis_click=True)
    )
    held = detector.update(
        PicoButtonState(left_axis_click=True, right_axis_click=True)
    )
    detector.update(PicoButtonState())
    second = detector.update(
        PicoButtonState(left_axis_click=True, right_axis_click=True)
    )

    assert first.emergency_stop_pressed
    assert not held.emergency_stop_pressed
    assert second.emergency_stop_pressed


def test_single_thumbstick_click_is_not_an_emergency():
    detector = PicoButtonEdgeDetector()

    left = detector.update(PicoButtonState(left_axis_click=True))
    detector.update(PicoButtonState())
    right = detector.update(PicoButtonState(right_axis_click=True))

    assert not left.emergency_stop_pressed
    assert not right.emergency_stop_pressed
