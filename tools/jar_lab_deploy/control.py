"""Position-reference guards shared by dry-run and real G1 Dex1 execution."""
import numpy as np


class ReferenceGuard:
    def __init__(self, low, high, arm_speed=.25, gripper_speed=.5, max_observation_age=.3, max_prediction_age=.75):
        self.low, self.high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
        if self.low.shape != (16,) or self.high.shape != (16,) or np.any(self.low >= self.high):
            raise ValueError("Need physical joint bounds for all 16 observed joints")
        self.speed = np.array([arm_speed] * 14 + [gripper_speed] * 2)
        self.max_observation_age, self.max_prediction_age = max_observation_age, max_prediction_age
        self.last = None

    def reset(self, measured):
        measured = np.asarray(measured, dtype=np.float64)
        if measured.shape != (16,) or not np.isfinite(measured).all():
            raise ValueError("Invalid measured robot state")
        if np.any(measured < self.low - .02) or np.any(measured > self.high + .02):
            raise ValueError("Measured position is outside configured physical bounds")
        self.last = np.clip(measured, self.low, self.high)

    def step(self, measured, target, dt, observation_age, prediction_age):
        measured, target = np.asarray(measured), np.asarray(target)
        if self.last is None:
            self.reset(measured)
        if (measured.shape != (16,) or target.shape != (16,) or not np.isfinite(measured).all()
                or not np.isfinite(target).all() or not 0 < dt <= .1):
            raise ValueError("Invalid state/reference/control period")
        if not 0 <= observation_age <= self.max_observation_age:
            raise TimeoutError("Robot observations stale; refuse further reference updates")
        if np.any(np.abs(self.last - measured) > .150001):
            raise RuntimeError("Robot moved outside the held reference envelope; stop instead of jumping targets")
        if not 0 <= prediction_age <= self.max_prediction_age:
            return self.last.copy(), "hold_stale_prediction"
        target = np.clip(target, self.low, self.high)
        # A long controller pause cannot create a large recovery jump.
        delta = self.speed * dt
        proposal = np.clip(target, self.last - delta, self.last + delta)
        proposal = np.clip(proposal, measured - .15, measured + .15)
        self.last = np.clip(proposal, self.low, self.high)
        return self.last.copy(), "tracking_reference"
