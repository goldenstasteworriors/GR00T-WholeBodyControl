"""Local browser tool for dual Dex1 tuning and collection trigger profiles.

Start in the collection environment: python -m gear_sonic.scripts.run_dex1_gripper_web
Only loopback is served. Initial mode is read-only; browser enable + heartbeat
are required for commands. No body controller is created.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import signal
import socket
import threading
import time
from urllib.parse import urlparse

from gear_sonic.utils.dex1_profile import load_profile, save_profile, validate_profile

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "gear_sonic/tools/dex1_gripper_web/index.html"
SIDES = ("left", "right")


class GripperControl:
    def __init__(self, hands, profile_path):
        self.hands = hands
        self.profile_path = Path(profile_path)
        self.profile = (load_profile(profile_path) if self.profile_path.exists() else
                        dict(version=1, unit="rad", released_q=[5.4, 5.4], pressed_q=[0.0, 0.0]))
        self.lock = threading.RLock()
        self.owner = None
        self.heartbeat = 0.0
        self.targets = {}
        self.ramp = {}
        self.speed = 0.3
        self.error = ""
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def disable(self):
        with self.lock:
            self.owner = None
            errors = []
            for side, hand in self.hands.items():
                try:
                    hand.release_control()
                except Exception as exc:
                    errors.append(f"{side}: {exc}")
            if errors:
                self.error = "; ".join(errors)

    def measured(self):
        return {side: float(hand.observe()["dex1_hand_q"][0]) for side, hand in self.hands.items()}

    def status(self, client):
        with self.lock:
            states = {}
            for side, hand in self.hands.items():
                try:
                    q = float(hand.observe()["dex1_hand_q"][0])
                    states[side] = dict(q=q, connected=True, target=self.targets.get(side, q))
                except RuntimeError as exc:
                    states[side] = dict(q=None, connected=False, target=None, error=str(exc))
            return dict(hands=states, enabled=self.owner is not None,
                        mine=self.owner == client, speed=self.speed, error=self.error,
                        profile=self.profile, profile_path=str(self.profile_path),
                        profile_saved=self.profile_path.exists(),
                        collection_args=f"--hand-type dex1 --dex1-profile {self.profile_path}")

    def action(self, action, data, client):
        with self.lock:
            if action == "enable":
                if self.owner not in (None, client):
                    raise ValueError("另一个网页正在控制，请先在原网页停止")
                measured = self.measured()
                if any(not 0 <= q <= 5.4 for q in measured.values()):
                    raise ValueError("反馈角度不在0–5.4 rad内，不能启用")
                self.targets = measured.copy()
                self.ramp = measured.copy()
                self.error = ""
                self.owner = client
                self.heartbeat = time.monotonic()
            elif action in ("disable", "heartbeat", "target"):
                if self.owner != client:
                    raise ValueError("当前网页尚未启用控制")
                if action == "disable":
                    self.disable()
                elif action == "heartbeat":
                    self.heartbeat = time.monotonic()
                else:
                    targets = data.get("targets", {})
                    if not isinstance(targets, dict) or not targets or set(targets) - set(SIDES):
                        raise ValueError("请选择左或右夹爪目标")
                    checked = {}
                    for side, value in targets.items():
                        if isinstance(value, (str, bool)) or not isinstance(value, (int, float)):
                            raise ValueError("目标必须是数值")
                        if not math.isfinite(value) or not 0 <= value <= 5.4:
                            raise ValueError("角度必须在0–5.4 rad内")
                        checked[side] = float(value)
                    speed = float(data.get("speed", self.speed))
                    if not math.isfinite(speed) or not 0.05 <= speed <= 1:
                        raise ValueError("速度必须在0.05–1 rad/s内")
                    self.targets.update(checked)
                    self.speed = speed
            elif action == "profile":
                profile = validate_profile(data.get("profile"))
                self.profile = save_profile(self.profile_path, profile)
            else:
                raise ValueError("未知操作")
            return self.status(client)

    def run(self):
        previous = time.monotonic()
        while not self.stop_event.wait(0.02):
            now = time.monotonic()
            dt = min(now - previous, 0.1)
            previous = now
            with self.lock:
                if self.owner is None:
                    continue
                try:
                    if now - self.heartbeat > 1.0:
                        raise RuntimeError("网页心跳中断，已停止控制")
                    self.measured()  # Reject stale feedback from either hand before writing.
                    for side, hand in self.hands.items():
                        delta = self.targets[side] - self.ramp[side]
                        self.ramp[side] += math.copysign(min(abs(delta), self.speed * dt), delta)
                        hand.command_position(self.ramp[side])
                except Exception as exc:
                    self.error = str(exc)
                    self.disable()

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=2)
        self.disable()
        for hand in self.hands.values():
            hand.close()


def make_handler(control):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if urlparse(self.path).path == "/api/state":
                self.respond(control.status(self.headers.get("X-Dex1-Client", "")))
            elif self.path == "/":
                content = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON required")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Origin mismatch")
                client = self.headers.get("X-Dex1-Client", "")
                if not re.fullmatch(r"[a-zA-Z0-9_-]{16,80}", client):
                    raise ValueError("Invalid browser client")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Invalid request object")
                action = self.path.removeprefix("/api/")
                self.respond(control.action(action, data, client))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                self.respond(dict(error=str(exc)), 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="enx00e04c36c4e8")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--profile", default=".runtime/dex1_gripper_profile.json")
    args = parser.parse_args()
    if args.network not in dict(socket.if_nameindex()).values():
        parser.error("Robot-facing network interface does not exist")
    path = Path(args.profile).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        parser.error("Profile must be inside the project directory")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from decoupled_wbc.control.envs.g1.g1_dex1_hand import G1Dex1Hand

    ChannelFactoryInitialize(0, args.network)
    hands = {}
    control = None
    try:
        for side in SIDES:
            hands[side] = G1Dex1Hand(is_left=side == "left")
        control = GripperControl(hands, path)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(control))
        def interrupt_handler(*_args):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt_handler)
        print(f"Dex1 web: http://127.0.0.1:{args.port} (read-only until enabled)", flush=True)
        print(f"Collection profile: {path}", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            server.server_close()
    except KeyboardInterrupt:
        pass
    finally:
        if control is not None:
            control.close()
        else:
            for hand in hands.values():
                hand.close()


if __name__ == "__main__":
    main()
