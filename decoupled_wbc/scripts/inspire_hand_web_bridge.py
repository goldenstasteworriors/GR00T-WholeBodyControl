#!/usr/bin/env python3
"""为 Inspire 手部网页调试提供不依赖任务配置的 DDS -> Modbus bridge。"""

from __future__ import annotations

import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

from decoupled_wbc.scripts.inspire_modbus_hand import INSPIRE_HAND_DOF, InspireModbusHand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Inspire 网页调试 DDS -> Modbus bridge。")
    parser.add_argument("--network", default="eth0", help="DDS 网卡。")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS 域 ID。")
    parser.add_argument("--left-ip", default="192.168.123.210")
    parser.add_argument("--right-ip", default="192.168.123.211")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--speed", type=int, default=3000)
    parser.add_argument("--force", type=int, default=12000)
    parser.add_argument("--side", choices=("left", "right", "both"), default="left")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_sides = ("left", "right") if args.side == "both" else (args.side,)
    hands = {
        "left": InspireModbusHand("left", args.left_ip, args.hand_port, args.device_id),
        "right": InspireModbusHand("right", args.right_ip, args.hand_port, args.device_id),
    }
    last_command: tuple[tuple[float, ...], ...] | None = None

    def callback(message: MotorCmds_) -> None:
        nonlocal last_command
        if len(message.cmds) < 2 * INSPIRE_HAND_DOF:
            print(f"忽略长度不足的 Inspire DDS 命令：{len(message.cmds)}")
            return
        values = {
            "right": tuple(float(message.cmds[index].q) for index in range(INSPIRE_HAND_DOF)),
            "left": tuple(
                float(message.cmds[index + INSPIRE_HAND_DOF].q)
                for index in range(INSPIRE_HAND_DOF)
            ),
        }
        command_key = tuple(values[side] for side in active_sides)
        if command_key == last_command:
            return
        try:
            for side in active_sides:
                angles = np.rint(np.clip(values[side], 0.0, 1.0) * 1000.0).astype(int).tolist()
                hands[side].set_angle(angles, speed=args.speed, force=args.force)
                print(f"网页 DDS -> Modbus {side}={angles}")
            last_command = command_key
        except Exception as exc:
            print(f"网页 DDS -> Modbus 失败：{exc}")

    ChannelFactoryInitialize(args.domain_id, args.network)
    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)
    print(
        "网页调试 bridge 已运行在 rt/inspire/cmd，"
        f"控制 {','.join(active_sides)}；不依赖 hand-task 配置。"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
