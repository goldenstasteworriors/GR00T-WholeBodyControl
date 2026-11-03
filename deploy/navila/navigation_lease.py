"""Fail-closed velocity lease, enforced inside the ThinkPad WBC loop.

Protocol uses host monotonic time; publishers must run on the same machine.
Zero velocity means Balance/standing, not motor damping. This cannot protect
against failure of the WBC process or the robot's own DDS link.
"""
import math


class NavigationLease:
    def __init__(self):
        self.expires = 0.0
        self.last_sent = -1.0
        self.velocity = [0.0, 0.0, 0.0]

    def update(self, goal, now, running):
        if goal is not None:
            try:
                lease = goal["navila"]
                sent = float(lease["sent_monotonic"])
                expires = float(lease["expires_monotonic"])
                vector = [float(value) for value in goal["navigate_cmd"]]
                valid = (len(vector) == 3 and all(math.isfinite(v) for v in vector)
                         and math.isfinite(sent) and math.isfinite(expires)
                         and self.last_sent < sent <= now
                         and 0 <= now - sent <= 0.25
                         and now < expires <= sent + 0.25
                         and 0 <= vector[0] <= 0.2 and vector[1] == 0
                         and abs(vector[2]) <= 0.3
                         and not (vector[0] and vector[2]))
                if not valid:
                    raise ValueError("Invalid navigation lease")
                self.last_sent, self.expires, self.velocity = sent, expires, vector
            except (KeyError, TypeError, ValueError, OverflowError):
                self.expires = 0.0
                self.velocity = [0.0, 0.0, 0.0]
        if not running or now >= self.expires:
            self.expires = 0.0
            self.velocity = [0.0, 0.0, 0.0]
        return self.velocity.copy()
