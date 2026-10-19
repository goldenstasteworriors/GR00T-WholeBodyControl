"""Software camera grouping on a single capture host's timestamp clock.

USB timestamps currently describe host read completion, not sensor exposure.
This bounds inter-view timestamp skew; it is not hardware trigger synchronization.
"""

from collections import deque
import math

import numpy as np

THREE_CAMERA_MOUNTS = ("ego_view", "left_wrist", "right_wrist")


def validate_rgb_bundle(data, mounts, max_skew_s, shape=(480, 640, 3)):
    stamps = []
    for mount in mounts:
        image = data.get("images", {}).get(mount)
        stamp = float(data.get("timestamps", {}).get(mount, float("nan")))
        if not isinstance(image, np.ndarray) or image.shape != shape or image.dtype != np.uint8:
            raise ValueError(f"Missing/invalid {mount}: expected uint8 RGB {shape}")
        if not math.isfinite(stamp):
            raise ValueError(f"Missing/invalid {mount} timestamp")
        stamps.append(stamp)
    if max(stamps) - min(stamps) > max_skew_s + 1e-9:
        raise ValueError(f"Camera timestamp skew {(max(stamps)-min(stamps))*1000:.2f} ms exceeds "
                         f"{max_skew_s*1000:.2f} ms")
    return tuple(stamps)


class ImageBundleBuffer:
    """Hold a validated group briefly, using receive monotonic time for freshness.

    Capture and receiver wall clocks need not agree. Repeated capture timestamps
    do not refresh this buffer's age.
    """

    def __init__(self, mounts, max_skew_s=0.03, max_hold_s=0.25):
        if (not math.isfinite(max_skew_s) or not math.isfinite(max_hold_s)
                or max_skew_s < 0 or max_hold_s <= 0):
            raise ValueError("Invalid camera skew or hold limit")
        self.mounts = tuple(mounts)
        self.max_skew_s = max_skew_s
        self.max_hold_s = max_hold_s
        self.latest = None
        self.stamps = None
        self.received_at = float("-inf")

    def update(self, data, now):
        try:
            stamps = validate_rgb_bundle(data, self.mounts, self.max_skew_s)
        except (ValueError, TypeError):
            self.latest = None
            raise
        if self.stamps is not None and not all(a > b for a, b in zip(stamps, self.stamps)):
            return False
        self.latest = data
        self.stamps = stamps
        self.received_at = now
        return True

    def current(self, now):
        return self.latest if now - self.received_at <= self.max_hold_s else None


class TimestampSynchronizer:
    """Consume each source once, retaining incomplete groups across polls."""

    def __init__(self, mounts, max_skew_s=0.03, max_age_s=0.25, queue_size=6):
        if not mounts or len(set(mounts)) != len(mounts):
            raise ValueError("Expected unique camera mounts")
        if (not math.isfinite(max_skew_s) or not math.isfinite(max_age_s)
                or not 0 <= max_skew_s <= max_age_s or max_age_s <= 0 or queue_size < 1):
            raise ValueError("Expected 0 <= sync skew <= max age and positive queue size")
        self.buffers = {mount: deque(maxlen=queue_size) for mount in mounts}
        self.last_seen = {mount: float("-inf") for mount in mounts}
        self.max_skew_s = max_skew_s
        self.max_age_s = max_age_s
        self.dropped = 0

    def push(self, mount, frame):
        stamp = float(frame.get("timestamps", {}).get(mount, float("nan")))
        if (not math.isfinite(stamp) or stamp <= self.last_seen[mount]
                or mount not in frame.get("images", {})):
            self.dropped += 1
            return
        self.last_seen[mount] = stamp
        if len(self.buffers[mount]) == self.buffers[mount].maxlen:
            self.dropped += 1
        self.buffers[mount].append(frame)

    def pop(self, now):
        for mount, frames in self.buffers.items():
            while frames:
                age = now - float(frames[0]["timestamps"][mount])
                if -0.05 <= age <= self.max_age_s:
                    break
                frames.popleft()
                self.dropped += 1
        while all(self.buffers.values()):
            stamps = {mount: float(frames[0]["timestamps"][mount])
                      for mount, frames in self.buffers.items()}
            if max(stamps.values()) - min(stamps.values()) <= self.max_skew_s + 1e-9:
                # Preserve subsequent valid groups for the next publish. Draining
                # all matches here needlessly halves output when camera phases
                # cause two groups to become ready in the same server poll.
                return {mount: frames.popleft() for mount, frames in self.buffers.items()}
            else:
                oldest = min(stamps, key=stamps.get)
                self.buffers[oldest].popleft()
                self.dropped += 1
        return None
