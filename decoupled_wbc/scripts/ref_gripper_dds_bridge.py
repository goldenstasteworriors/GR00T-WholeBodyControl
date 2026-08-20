#!/usr/bin/env python3
"""Drive the right AgiBot REF USB gripper from the raw Pico right trigger."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgSubscriber
from gear_sonic.utils.data_collection.inspire_hand_tasks import (
    DEFAULT_HAND_TASK,
    HAND_TASK_CONFIG_ENV,
)
from gear_sonic.utils.data_collection.ref_gripper_tasks import (
    resolve_right_gripper_positions,
)


DEFAULT_REF_CONTROL_DIR = Path("/home/unitree/data_collection/USB_Fibre_Ubuntu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to the raw Pico right trigger and control the REF USB gripper."
        )
    )
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

    open_position, pressed_position = resolve_right_gripper_positions(
        args.hand_task, args.hand_task_config or None
    )

    transport = PyUsbTransport()
    client = RefClient(transport, interface_crc=CAPTURED_INTERFACE_CRC)
    client.initialize(check_schema=False, expected_serial=DEFAULT_FIRMWARE_SERIAL)
    client.prepare()
    client.set_pos(open_position)
    last_position = open_position

    ros_manager = ROSManager(node_name="RefGripperBridge")
    trigger_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)
    print(
        f"REF gripper bridge running on {CONTROL_GOAL_TOPIC}: "
        f"open={open_position}, pressed={pressed_position}"
    )
    try:
        while ros_manager.ok():
            msg = trigger_subscriber.get_msg()
            if msg is None:
                time.sleep(0.01)
                continue
            position = (
                pressed_position if float(msg["right_trigger"]) > 0.5 else open_position
            )
            if position != last_position:
                client.set_pos(position)
                last_position = position
                state = "pressed" if position == pressed_position else "open"
                print(f"Pico right trigger -> {state}: set_pos({position})")
    finally:
        client.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    main()
