import threading
import time

import numpy as np


class UnitreeLocoKeyboardController:
    """Translate control-pane key presses into safe Unitree loco goals."""

    def __init__(
        self,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 1.0,
        linear_step: float = 0.1,
        angular_step: float = 0.1,
        command_timeout: float = 0.5,
        full_control_enabled: bool = True,
    ):
        self._max_linear_velocity = float(max_linear_velocity)
        self._max_angular_velocity = float(max_angular_velocity)
        self._linear_step = float(linear_step)
        self._angular_step = float(angular_step)
        self._command_timeout = float(command_timeout)
        self._full_control_enabled = bool(full_control_enabled)
        if self._command_timeout <= 0.0:
            raise ValueError("command_timeout must be positive")
        self._velocity = np.zeros(3, dtype=np.float64)
        self._last_motion_key_time = 0.0
        self._reported_active = False
        self._start_pending = False
        self._emergency_pending = False
        self._lock = threading.Lock()

    def _zero_velocity(self) -> None:
        self._velocity.fill(0.0)

    def _print_velocity(self) -> None:
        print(
            "Keyboard loco velocity: "
            f"vx={self._velocity[0]:+.2f}, vy={self._velocity[1]:+.2f}, "
            f"wz={self._velocity[2]:+.2f}",
            flush=True,
        )

    def handle_keyboard_button(self, keycode: str | None) -> None:
        if not keycode:
            return
        key = str(keycode).lower()
        with self._lock:
            if key == "space":
                self._start_pending = False
                self._emergency_pending = True
                self._zero_velocity()
                print("Keyboard loco emergency stop requested", flush=True)
                return

            if not self._full_control_enabled:
                return

            if key == "g":
                if self._reported_active or self._start_pending:
                    self._start_pending = False
                    self._emergency_pending = True
                    self._zero_velocity()
                    print("Keyboard loco emergency stop requested", flush=True)
                else:
                    self._start_pending = True
                    print("Keyboard loco startup requested", flush=True)
                return

            if key == "z":
                self._zero_velocity()
                self._print_velocity()
                return

            if key not in {"w", "s", "a", "d", "q", "e"}:
                return
            if not self._reported_active:
                print(
                    f"Ignoring keyboard loco key {key!r}: lower body is not active",
                    flush=True,
                )
                return

            if key == "w":
                self._velocity[0] = self._linear_step
            elif key == "s":
                self._velocity[0] = -self._linear_step
            elif key == "a":
                self._velocity[1] = self._linear_step
            elif key == "d":
                self._velocity[1] = -self._linear_step
            elif key == "q":
                self._velocity[2] = self._angular_step
            elif key == "e":
                self._velocity[2] = -self._angular_step

            self._velocity[:2] = np.clip(
                self._velocity[:2],
                -self._max_linear_velocity,
                self._max_linear_velocity,
            )
            self._velocity[2] = np.clip(
                self._velocity[2],
                -self._max_angular_velocity,
                self._max_angular_velocity,
            )
            self._last_motion_key_time = time.monotonic()
            self._print_velocity()

    def get_control_goal(self, lower_body_active: bool) -> dict:
        """Return the latest navigation command and pending transition request."""
        with self._lock:
            lower_body_active = bool(lower_body_active)
            goal = {}
            if self._full_control_enabled:
                if self._reported_active and not lower_body_active:
                    self._zero_velocity()
                self._reported_active = lower_body_active
                if lower_body_active:
                    self._start_pending = False

                if np.any(self._velocity) and (
                    time.monotonic() - self._last_motion_key_time > self._command_timeout
                ):
                    self._zero_velocity()
                    print(
                        "Keyboard loco command timed out; velocity reset to zero",
                        flush=True,
                    )
                goal["navigate_cmd"] = self._velocity.copy()
            if self._emergency_pending:
                goal["emergency_stop"] = True
                goal["set_policy_action"] = False
                self._emergency_pending = False
            elif self._full_control_enabled and self._start_pending:
                goal["set_policy_action"] = True
            return goal
