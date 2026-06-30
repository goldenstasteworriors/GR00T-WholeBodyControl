import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from decoupled_wbc.control.envs.g1.utils.command_sender import (
    INSPIRE_GRASP_Q,
    INSPIRE_OPEN_Q,
    InspireHandCommandSender,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test RH56DFTP Inspire hand open/grasp commands.")
    parser.add_argument("--network", default="", help="DDS network interface, e.g. eth0.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--side",
        choices=["left", "right", "both"],
        default="both",
        help="Hand side to control.",
    )
    parser.add_argument(
        "--command",
        choices=["open", "grasp", "toggle"],
        default="toggle",
        help="Command to send.",
    )
    parser.add_argument("--period", type=float, default=1.0, help="Toggle period in seconds.")
    parser.add_argument("--count", type=int, default=6, help="Number of toggle commands.")
    return parser.parse_args()


def send(left_sender, right_sender, side: str, q: np.ndarray):
    if side in ("left", "both"):
        left_sender.send_command(q)
    if side in ("right", "both"):
        right_sender.send_command(q)


def main():
    args = parse_args()
    if args.network:
        ChannelFactoryInitialize(args.domain_id, args.network)
    else:
        ChannelFactoryInitialize(args.domain_id)

    left_sender = InspireHandCommandSender(is_left=True)
    right_sender = InspireHandCommandSender(is_left=False)

    if args.command == "toggle":
        for i in range(args.count):
            q = INSPIRE_GRASP_Q if i % 2 == 0 else INSPIRE_OPEN_Q
            label = "grasp" if i % 2 == 0 else "open"
            print(f"sending {label}: {q.tolist()}")
            send(left_sender, right_sender, args.side, q)
            time.sleep(args.period)
    else:
        q = INSPIRE_GRASP_Q if args.command == "grasp" else INSPIRE_OPEN_Q
        print(f"sending {args.command}: {q.tolist()}")
        send(left_sender, right_sender, args.side, q)


if __name__ == "__main__":
    main()
