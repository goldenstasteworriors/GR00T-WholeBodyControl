"""Sole robot-local LowCmd publisher with a latched, local Balance fallback.

Normal commands retain the ThinkPad's existing joint targets and gains. Only
after a lost command connection does local inference replace legs/waist targets.
No position packets are accepted again in that process, even after an estop.
"""

import argparse
import copy
import json
from pathlib import Path
import signal
import socket
import threading
import time

from decoupled_wbc.control.policy.onboard_balance_hold import OnboardBalanceHold
from decoupled_wbc.control.utils.onboard_body_link import receive_packet, send_packet


class BodyHoldSession:
    def __init__(self, balance, loss_after=0.1):
        self.balance = balance
        self.loss_after = loss_after
        self.lock = threading.RLock()
        self.last = None
        self.last_received = None
        self.latched = False
        self.phase = "waiting"
        self.exit_requested = False

    def receive(self, packet, now):
        with self.lock:
            kind = packet["kind"]
            if kind == "exit":
                self.exit_requested = True
                self.phase = "damping"
            elif kind == "damping":
                self.last = copy.deepcopy(packet)
                self.last_received = now
                self.phase = "damping"
            elif kind == "position" and not self.latched:
                self.last = copy.deepcopy(packet)
                self.last_received = now
                self.phase = "position"

    def disconnect(self):
        with self.lock:
            if self.latched:
                return
            self.latched = True
            context = self.last.get("context") if self.last else None
            if self.phase == "position" and context and context["running"]:
                self.balance.start(context)
                self.phase = "balance_hold"
            elif self.phase == "position":
                self.phase = "pose_hold"
            print(f"[BodyHold] DISCONNECTED: {self.phase}; restart required", flush=True)

    def step(self, state, now):
        with self.lock:
            if self.last_received is not None and now - self.last_received > self.loss_after:
                self.disconnect()
            if self.last is None:
                return None
            motors = copy.deepcopy(self.last["motors"])
            if self.phase == "damping":
                for m in motors:
                    m[1:] = [0.0, 0.0, 0.0, 0.0, 8.0]
            elif self.phase == "balance_hold":
                targets = self.balance.step(
                    [m.q for m in state.motor_state[:29]],
                    [m.dq for m in state.motor_state[:29]],
                    state.imu_state.quaternion, state.imu_state.gyroscope,
                )
                for i, m in enumerate(motors):
                    m[2:4] = [0.0, 0.0]
                    if i < 15:
                        m[1] = float(targets[i])
            elif self.phase == "pose_hold":
                for m in motors:
                    m[2:4] = [0.0, 0.0]
            return motors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--bind", default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    resources = repo / "decoupled_wbc/sim2mujoco/resources/robots/g1"
    balance = OnboardBalanceHold(resources / "g1_gear_wbc.yaml",
                                resources / "policy/GR00T-WholeBodyControl-Balance.onnx")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(None, 0)
    publisher = None
    if not args.read_only:
        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
    command = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    session = BodyHoldSession(balance)
    stopped = threading.Event()
    damping_confirmed = threading.Event()
    observation_requested = threading.Event()
    observation_ready = threading.Event()
    observation = [None]
    status_name = "onboard_body_hold_status" + (f"_{args.port}" if args.port != 5560 else "")
    status_path = repo / f".runtime/{status_name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda *_: stopped.set())

    def read_observation():
        # A DDS read cannot prevent an explicit stop from reaching the damping
        # finally. This does not invent feedback or apply a feedback timeout.
        while True:
            observation_requested.wait()
            observation_requested.clear()
            observation[0] = subscriber.Read()
            observation_ready.set()

    threading.Thread(target=read_observation, daemon=True).start()

    def publish(motors):
        command.mode_machine = session.last["mode_machine"]
        command.mode_pr = session.last["mode_pr"]
        for motor, values in zip(command.motor_cmd, motors):
            motor.mode, motor.q, motor.dq, motor.tau, motor.kp, motor.kd = values
        command.crc = crc.Crc(command)
        if publisher is not None and not publisher.Write(command):
            raise RuntimeError("Local LowCmd publication failed")

    def serve(connection):
        try:
            while not stopped.is_set():
                packet = receive_packet(connection)
                if packet["kind"] in {"exit", "damping"}:
                    damping_confirmed.clear()
                session.receive(packet, time.monotonic())
                if packet["kind"] in {"exit", "damping"}:
                    with session.lock:
                        if session.last is not None:
                            publish(session.step(None, time.monotonic()))
                        damping_confirmed.set()
                    send_packet(connection, {"latched": session.latched,
                                             "sequence": packet.get("sequence", 0),
                                             "damping_confirmed": damping_confirmed.is_set()})
                    if packet["kind"] == "exit":
                        stopped.set()
                        return
                else:
                    send_packet(connection, {"latched": session.latched, "phase": session.phase,
                                             "sequence": packet.get("sequence", 0)})
        except (OSError, EOFError):
            session.disconnect()

    # Accept exactly one controller. There is deliberately no accept/reconnect loop.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.bind, args.port))
    listener.listen(1)
    listener.settimeout(0.2)
    connection = None
    try:
        print(f"[BodyHold] ready {args.bind}:{args.port}; read_only={args.read_only}", flush=True)
        status_path.write_text(json.dumps({"phase": "waiting", "latched": False,
                                           "read_only": args.read_only}) + "\n")
        while connection is None and not stopped.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                pass
        listener.close()
        if connection is None:
            return
        status_path.write_text(json.dumps({"phase": "connected", "latched": False,
                                           "read_only": args.read_only}) + "\n")
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(connection,), daemon=True).start()
        previous_status = None
        while not stopped.is_set():
            start = time.monotonic()
            observation_ready.clear()
            observation_requested.set()
            while not observation_ready.wait(0.02) and not stopped.is_set():
                pass
            if stopped.is_set():
                break
            state = observation[0]
            motors = session.step(state, start)
            if motors is not None:
                with session.lock:
                    # An estop received during inference takes precedence.
                    if session.phase == "damping":
                        motors = session.step(None, time.monotonic())
                    publish(motors)
                if session.phase == "damping":
                    damping_confirmed.set()
            status = {"phase": session.phase, "latched": session.latched,
                      "read_only": args.read_only}
            if status != previous_status:
                status_path.write_text(json.dumps(status) + "\n")
                previous_status = status
            stopped.wait(max(0.0, 0.02 - (time.monotonic() - start)))
    finally:
        session.latched = True
        if session.last is not None:
            session.phase = "damping"
            for _ in range(25):
                publish(session.step(None, time.monotonic()))
                time.sleep(0.02)
        if connection is not None:
            connection.close()
        listener.close()
        subscriber.Close()
        if publisher is not None:
            publisher.Close()


if __name__ == "__main__":
    main()
