"""Inspect or exercise Dex1 opening angles, without creating a body controller.

Default: read feedback only. --execute enables native-angle commands with a
speed-limited ramp. This is a permanent operator tool, not a temporary test.
"""

import argparse
import json
import math
import socket
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from decoupled_wbc.control.envs.g1.g1_dex1_hand import G1Dex1Hand


def move(hands, targets, speed, duration):
    starts = {side: float(hand.observe()["dex1_hand_q"][0]) for side, hand in hands.items()}
    start = time.monotonic()
    next_print = start
    reached_since = None
    while time.monotonic() - start < duration:
        now = time.monotonic()
        elapsed = now - start
        feedback = {}
        for side, target in targets.items():
            delta = target - starts[side]
            ramp = starts[side] + math.copysign(min(abs(delta), speed * elapsed), delta)
            hands[side].command_position(ramp)
            feedback[side] = float(hands[side].observe()["dex1_hand_q"][0])
        if now >= next_print:
            print(json.dumps(dict(target_q=targets, measured_q=feedback)), flush=True)
            next_print = now + 0.25
        reached = all(abs(feedback[side] - target) < 0.08 for side, target in targets.items())
        reached_since = (reached_since or now) if reached else None
        if reached_since is not None and now - reached_since >= 0.5:
            return
        time.sleep(0.02)
    raise RuntimeError("Target not reached within duration; check feedback and physical obstruction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--left-q", type=float, default=None, help="Left target rad: 0 closed, 5.4 open")
    parser.add_argument("--right-q", type=float, default=None, help="Right target rad: 0 closed, 5.4 open")
    parser.add_argument("--speed", type=float, default=0.5, help="Ramp speed in rad/s, maximum 1")
    parser.add_argument("--duration", type=float, default=15, help="Read duration or movement timeout, seconds")
    parser.add_argument("--execute", action="store_true", help="Enable gripper motion")
    parser.add_argument("--return-to-start", action="store_true", help="After reaching targets, ramp back to initial angles")
    args = parser.parse_args()
    if args.network not in dict(socket.if_nameindex()).values():
        parser.error("network interface does not exist")
    if not 0 < args.speed <= 1 or not 0 < args.duration <= 120:
        parser.error("speed must be in (0, 1] and duration in (0, 120]")
    targets = {side: q for side in ("left", "right") if (q := getattr(args, f"{side}_q")) is not None}
    for q in targets.values():
        G1Dex1Hand.validate_target(q)
    if args.execute and not targets:
        parser.error("--execute requires --left-q or --right-q")
    ChannelFactoryInitialize(0, args.network)
    hands = {}
    try:
        for side in (targets if args.execute else ("left", "right")):
            hands[side] = G1Dex1Hand(is_left=side == "left")
            hands[side].calibrate_hand()  # Wait for feedback; does not set motor zero.
        if args.execute:
            initial = {side: float(hand.observe()["dex1_hand_q"][0]) for side, hand in hands.items()}
            print(json.dumps(dict(initial_q=initial, target_q=targets, speed_rad_s=args.speed)), flush=True)
            move(hands, targets, args.speed, args.duration)
            if args.return_to_start:
                move(hands, initial, args.speed, args.duration)
            print("PASS: requested gripper angles reached", flush=True)
        else:
            print("READ ONLY: no commands; add --execute to move to the supplied targets", flush=True)
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                print(json.dumps({side: float(hand.observe()["dex1_hand_q"][0]) for side, hand in hands.items()}), flush=True)
                time.sleep(0.25)
    finally:
        # Drop position gains on exit, including timeout and Ctrl+C.
        for hand in hands.values():
            hand.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
