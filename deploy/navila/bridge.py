"""G1 camera -> NaVILA -> original Balance/Walk navigation interface.

No added navigation watchdog, age check, action cap, or publisher-conflict
check. Manual damping estop is provided separately by estop_thinkpad.sh.
"""
import argparse
import base64
import json
import math
from pathlib import Path
import re
import signal
import socket
import struct
import threading
import time

import cv2
import numpy as np


def log(event, **fields):
    print(json.dumps({"event": event, "time": time.time(), **fields}), flush=True)


def parse_action(text):
    if not isinstance(text, str):
        raise ValueError(f"Model returned a non-string response: {text!r}")
    text = text.strip().lower()
    if text in {"stop", "stop.", "the next action is stop.",
                "i think i should stop because i have finished the instruction."}:
        return "stop", 0.0
    match = re.fullmatch(r"(?:the next action is )?move forward (\d+(?:\.\d+)?) cm\.?", text)
    if match:
        return "forward", float(match[1])
    match = re.fullmatch(r"(?:the next action is )?turn (left|right) (\d+(?:\.\d+)?) degrees?\.?", text)
    if match:
        return match[1], float(match[2])
    raise ValueError(f"Cannot parse navigation action: {text!r}")


def sample_history(history):
    if len(history) < 8:
        black = encode_image(np.zeros((224, 224, 3), dtype=np.uint8))
        return [black] * (8 - len(history)) + history
    indices = np.linspace(0, len(history) - 2, 7, dtype=int)
    return [history[i] for i in indices] + [history[-1]]


def encode_image(rgb):
    height, width = rgb.shape[:2]
    if max(height, width) > 640:
        scale = 640 / max(height, width)
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def receive_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("NaVILA connection closed before response completed")
        data.extend(chunk)
    return bytes(data)


def infer(host, port, images, instruction):
    payload = json.dumps({"images": images, "query": instruction}).encode()
    started = time.monotonic()
    with socket.create_connection((host, port), timeout=None) as sock:
        sock.sendall(struct.pack(">Q", len(payload)) + payload)
        size = struct.unpack(">Q", receive_exact(sock, 8))[0]
        response = json.loads(receive_exact(sock, size))
    return response, time.monotonic() - started


class Camera:
    def __init__(self, host, port, key):
        self.host, self.port, self.key = host, port, key
        self.sample = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        client = None
        try:
            client = ComposedCameraClientSensor(self.host, self.port)
            while not self.stop.is_set():
                item = client.read(blocking=False, new_only=True)
                if item:
                    frame = item.get("images", {}).get(self.key)
                    if frame is not None:
                        self.sample = (frame.copy(), item.get("timestamps", {}).get(self.key))
                self.stop.wait(0.005)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            if client is not None:
                client.close()

    def snapshot(self):
        frame, stamp = self.sample
        return encode_image(frame), list(frame.shape), stamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def segment(action, amount, linear_speed=0.5, angular_speed=math.pi / 6):
    if action == "forward":
        return (linear_speed, 0.0, 0.0), amount / 100 / linear_speed
    return (0.0, 0.0, angular_speed if action == "left" else -angular_speed), math.radians(amount) / angular_speed


class WBCPublisher:
    """Original ROS goal interface; only monitor explicit manual damping."""
    def __init__(self, linear_speed, angular_speed):
        from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC, LOWER_BODY_POLICY_STATUS_TOPIC
        from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
        self.manager = ROSManager("NaVILANavigation")
        self.pub = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.sub = ROSMsgSubscriber(LOWER_BODY_POLICY_STATUS_TOPIC)
        self.linear_speed, self.angular_speed = linear_speed, angular_speed
        self.status = {}
        self.estopped = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.observe_manual_stop, daemon=True)
        self.thread.start()

    def observe_manual_stop(self):
        while not self.stop.is_set():
            item = self.sub.get_msg()
            if item is not None:
                self.status = item
                if item.get("control_mode") == "damped":
                    self.estopped.set()
            self.stop.wait(0.02)

    def check_manual_stop(self):
        if self.estopped.is_set():
            raise KeyboardInterrupt

    def wait_ready(self):
        log("waiting_for_manual_standing", key="] in controller pane after ready")
        while self.status.get("control_mode") != "running":
            self.check_manual_stop()
            time.sleep(0.05)

    def publish(self, velocity):
        self.pub.publish({"navigate_cmd": list(velocity), "target_time": time.monotonic() + 0.02})

    def move(self, action, amount):
        velocity, duration = segment(action, amount, self.linear_speed, self.angular_speed)
        log("segment", action=action, amount=amount, velocity=velocity, duration_s=duration)
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.check_manual_stop()
            self.publish(velocity)
            time.sleep(0.02)
        # The requested segment has completed; this is not a watchdog stop.
        self.publish((0.0, 0.0, 0.0))

    def close(self):
        # No automatic velocity command on exceptions/process shutdown.
        self.stop.set()
        self.thread.join(timeout=2)
        self.manager.shutdown()
        if hasattr(self.manager, "thread"):
            self.manager.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-key", default="ego_view")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--linear-speed", type=float, default=0.5)
    parser.add_argument("--angular-speed", type=float, default=math.pi / 6)
    args = parser.parse_args()
    camera, wbc = None, None
    history = []
    try:
        if not args.image:
            camera = Camera(args.camera_host, args.camera_port, args.camera_key)
            while camera.sample is None:
                if camera.error:
                    raise RuntimeError(camera.error)
                time.sleep(0.05)
        if args.execute:
            wbc = WBCPublisher(args.linear_speed, args.angular_speed)
            wbc.wait_ready()
        for step in range(args.steps):
            if wbc:
                wbc.check_manual_stop()
            if camera:
                encoded, shape, source_stamp = camera.snapshot()
            else:
                bgr = cv2.imread(str(args.image))
                if bgr is None:
                    raise ValueError("Cannot read offline image")
                encoded = encode_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                shape, source_stamp = list(bgr.shape), None
            history.append(encoded)
            response, latency = infer(args.host, args.port, sample_history(history), args.instruction)
            action, amount = parse_action(response)
            log("inference", step=step + 1, response=response, action=action,
                amount=amount, round_trip_s=latency, camera_shape=shape,
                camera_source_timestamp=source_stamp, execute=args.execute)
            if wbc:
                wbc.check_manual_stop()
                if action == "stop":
                    wbc.publish((0.0, 0.0, 0.0))
                    break
                wbc.move(action, amount)
    finally:
        if wbc:
            wbc.close()
        if camera:
            camera.close()


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
    except Exception as exc:
        log("error", error=str(exc))
        raise
