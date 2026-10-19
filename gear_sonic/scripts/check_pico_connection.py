"""Read-only PICO pose/trigger check. No DDS or robot controller is created."""

import argparse
import math
import time

import numpy as np


def check(client, timeout):
    start = time.monotonic()
    previous = None
    advances = 0
    first_advance = None
    while time.monotonic() - start < timeout:
        timestamp = client.get_timestamp_ns()
        poses = [np.asarray(client.get_pose_by_name(name))
                 for name in ("headset", "left_controller", "right_controller")]
        valid = all(p.shape == (7,) and np.isfinite(p).all()
                    and np.linalg.norm(p[3:7]) > 1e-6 for p in poses)
        if timestamp > 0 and valid:
            if previous is not None and timestamp > previous:
                advances += 1
                if first_advance is None:
                    first_advance = time.monotonic()
                if advances >= 6 and time.monotonic() - first_advance >= 0.2:
                    triggers = [float(client.get_key_value_by_name(f"{s}_trigger"))
                                for s in ("left", "right")]
                    if not all(math.isfinite(t) and 0 <= t <= 1 for t in triggers):
                        raise RuntimeError("PICO trigger values are invalid")
                    print(f"PICO OK: headset + both controllers advancing; left/right triggers={triggers}")
                    return True
            elif previous is not None and timestamp < previous:
                advances = 0
                first_advance = None
            previous = timestamp
        else:
            advances = 0
            first_advance = None
            previous = None
        time.sleep(0.02)
    print("PICO NOT READY: connect XRoboToolkit to this PC; enable head/controller tracking and Send")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    from decoupled_wbc.control.teleop.device.pico.xr_client import XrClient

    client = XrClient()
    try:
        success = check(client, args.timeout)
    finally:
        client.close()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
