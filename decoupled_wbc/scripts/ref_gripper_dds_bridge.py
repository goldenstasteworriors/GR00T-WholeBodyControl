#!/usr/bin/env python3
"""Drive the right AgiBot REF USB gripper from the Pico right trigger."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

from gear_sonic.utils.data_collection.inspire_hand_tasks import (
    DEFAULT_HAND_TASK,
    HAND_TASK_CONFIG_ENV,
    resolve_hand_task_pose,
)
from gear_sonic.utils.data_collection.ref_gripper_tasks import (
    resolve_right_gripper_positions,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_


DEFAULT_REF_CONTROL_DIR = Path("/home/unitree/data_collection/USB_Fibre_Ubuntu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to rt/inspire/cmd and control the right REF USB gripper."
    )
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--hand-task", default=DEFAULT_HAND_TASK)
    parser.add_argument("--hand-task-config", default="")
    parser.add_argument("--ref-control-dir", default=str(DEFAULT_REF_CONTROL_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hand_task_config:
        os.environ[HAND_TASK_CONFIG_ENV] = args.hand_task_config

    ref_control_dir = Path(args.ref_control_dir).expanduser()
    sys.path.insert(0, str(ref_control_dir))
    from agibot_ref import (  # noqa: PLC0415
        CAPTURED_INTERFACE_CRC,
        PyUsbTransport,
        RefClient,
    )
    from ref_control import DEFAULT_FIRMWARE_SERIAL  # noqa: PLC0415

    pressed_hand_q = np.asarray(
        resolve_hand_task_pose(args.hand_task, pressed=True), dtype=np.float64
    )
    open_position, pressed_position = resolve_right_gripper_positions(
        args.hand_task, args.hand_task_config or None
    )

    transport = PyUsbTransport()
    client = RefClient(transport, interface_crc=CAPTURED_INTERFACE_CRC)
    client.initialize(check_schema=False, expected_serial=DEFAULT_FIRMWARE_SERIAL)
    client.prepare()
    client.set_pos(open_position)
    last_position = open_position

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_position
        right_q = np.asarray([msg.cmds[index].q for index in range(6)], dtype=np.float64)
        position = (
            pressed_position
            if np.allclose(right_q, pressed_hand_q, rtol=0.0, atol=1e-6)
            else open_position
        )
        if position != last_position:
            client.set_pos(position)
            last_position = position
            state = "pressed" if position == pressed_position else "open"
            print(f"Pico right trigger -> {state}: set_pos({position})")

    ChannelFactoryInitialize(args.domain_id, args.network)
    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)
    print(
        "REF gripper bridge running on rt/inspire/cmd: "
        f"open={open_position}, pressed={pressed_position}"
    )
    try:
        while True:
            time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    main()
