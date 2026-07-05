import argparse
import itertools
import socket
import struct
import time
from typing import Iterable

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_


REG_CLEAR_ERROR = 1004
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522

INSPIRE_HAND_DOF = 6
FULL_OPEN = [1000, 1000, 1000, 1000, 1000, 1000]
FULL_GRASP = [0, 0, 0, 0, 1000, 1000]


class ModbusTcpError(RuntimeError):
    pass


class InspireModbusHand:
    def __init__(self, side: str, ip: str, port: int = 6000, device_id: int = 1, timeout: float = 1.0):
        self.side = side
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.timeout = timeout
        self._transaction_ids = itertools.count(1)

    def _request(self, function_code: int, payload: bytes) -> bytes:
        transaction_id = next(self._transaction_ids) & 0xFFFF
        pdu = struct.pack(">B", function_code) + payload
        header = struct.pack(">HHHB", transaction_id, 0, len(pdu) + 1, self.device_id)

        with socket.create_connection((self.ip, self.port), timeout=self.timeout) as sock:
            sock.sendall(header + pdu)
            response_header = self._recv_exact(sock, 7)
            rx_transaction_id, protocol_id, length, _unit_id = struct.unpack(">HHHB", response_header)
            if rx_transaction_id != transaction_id or protocol_id != 0:
                raise ModbusTcpError(f"{self.side}: invalid Modbus response header")

            response_pdu = self._recv_exact(sock, length - 1)
            if not response_pdu:
                raise ModbusTcpError(f"{self.side}: empty Modbus response")
            if response_pdu[0] & 0x80:
                code = response_pdu[1] if len(response_pdu) > 1 else -1
                raise ModbusTcpError(f"{self.side}: Modbus exception code {code}")
            if response_pdu[0] != function_code:
                raise ModbusTcpError(f"{self.side}: unexpected function code {response_pdu[0]}")
            return response_pdu[1:]

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ModbusTcpError("connection closed while reading response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def write_register(self, address: int, value: int) -> None:
        payload = struct.pack(">HH", address, int(value) & 0xFFFF)
        self._request(0x06, payload)

    def write_registers(self, address: int, values: Iterable[int]) -> None:
        registers = [int(v) & 0xFFFF for v in values]
        payload = struct.pack(">HHB", address, len(registers), len(registers) * 2)
        payload += struct.pack(">" + "H" * len(registers), *registers)
        self._request(0x10, payload)

    def set_angle(self, values: Iterable[int], speed: int = 3000, force: int = 12000) -> None:
        angle_values = [max(0, min(1000, int(v))) for v in values]
        if len(angle_values) != INSPIRE_HAND_DOF:
            raise ValueError(f"{self.side}: expected 6 angle values, got {len(angle_values)}")

        self.write_register(REG_CLEAR_ERROR, 1)
        time.sleep(0.02)
        self.write_registers(REG_SPEED_SET, [max(0, min(4000, int(speed)))] * INSPIRE_HAND_DOF)
        time.sleep(0.02)
        self.write_registers(REG_FORCE_SET, [max(0, min(12000, int(force)))] * INSPIRE_HAND_DOF)
        time.sleep(0.02)
        self.write_registers(REG_ANGLE_SET, angle_values)


def normalized_to_angle(values: Iterable[float]) -> list[int]:
    q = np.asarray(list(values), dtype=np.float64)
    if q.shape != (INSPIRE_HAND_DOF,):
        raise ValueError(f"expected 6 normalized values, got shape {q.shape}")
    q = np.clip(q, 0.0, 1.0)
    return [int(round(v * 1000.0)) for v in q]


def send_to_target(hands: dict[str, InspireModbusHand], target: str, values: list[int], speed: int, force: int) -> None:
    sides = ["left", "right"] if target == "both" else [target]
    for side in sides:
        hands[side].set_angle(values, speed=speed, force=force)
        print(f"sent {side}: {values}")


def run_command(args, hands: dict[str, InspireModbusHand]) -> None:
    if args.command == "toggle":
        for i in range(args.count):
            values = FULL_GRASP if i % 2 == 0 else FULL_OPEN
            label = "grasp" if i % 2 == 0 else "open"
            print(f"sending {label}: {values}")
            send_to_target(hands, args.side, values, args.speed, args.force)
            time.sleep(args.period)
        return

    values = FULL_GRASP if args.command == "grasp" else FULL_OPEN
    print(f"sending {args.command}: {values}")
    send_to_target(hands, args.side, values, args.speed, args.force)


def run_dds_bridge(args, hands: dict[str, InspireModbusHand]) -> None:
    last_command = None

    def callback(msg: MotorCmds_) -> None:
        nonlocal last_command
        if len(msg.cmds) < 12:
            print(f"skip short inspire command: {len(msg.cmds)}")
            return

        right_q = tuple(float(msg.cmds[i].q) for i in range(6))
        left_q = tuple(float(msg.cmds[i + 6].q) for i in range(6))
        command_key = (right_q, left_q)
        if command_key == last_command:
            return
        last_command = command_key

        try:
            right_angle = normalized_to_angle(right_q)
            left_angle = normalized_to_angle(left_q)
            hands["right"].set_angle(right_angle, speed=args.speed, force=args.force)
            hands["left"].set_angle(left_angle, speed=args.speed, force=args.force)
            print(f"DDS -> Modbus right={right_angle} left={left_angle}")
        except Exception as exc:
            print(f"DDS -> Modbus failed: {exc}")

    ChannelFactoryInitialize(args.domain_id, args.network)
    subscriber = ChannelSubscriber("rt/inspire/cmd", MotorCmds_)
    subscriber.Init(callback, 10)
    print("DDS -> Modbus bridge running on rt/inspire/cmd. Press Ctrl+C to stop.")
    while True:
        time.sleep(1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Control Inspire hands over Modbus TCP or bridge DDS commands.")
    parser.add_argument("--left-ip", default="192.168.123.210")
    parser.add_argument("--right-ip", default="192.168.123.211")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--speed", type=int, default=3000)
    parser.add_argument("--force", type=int, default=12000)
    parser.add_argument("--mode", choices=["command", "dds"], default="command")
    parser.add_argument("--network", default="enp7s0", help="DDS network interface for --mode dds.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--command", choices=["open", "grasp", "toggle"], default="toggle")
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    hands = {
        "left": InspireModbusHand("left", args.left_ip, args.hand_port, args.device_id),
        "right": InspireModbusHand("right", args.right_ip, args.hand_port, args.device_id),
    }
    if args.mode == "dds":
        run_dds_bridge(args, hands)
    else:
        run_command(args, hands)


if __name__ == "__main__":
    main()
