"""G1 ego camera -> NaVILA TCP -> optional leased WBC navigation goals.

Default invocation only reads images and runs inference. No ROS publisher is
created until --execute is supplied. All clocks in the command lease are on
the same ThinkPad; never publish these goals directly from server02.
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
        raise ValueError(f"Model returned an error/non-string: {text!r}")
    text = text.strip().lower()
    if text in {"stop", "stop.", "the next action is stop.",
                "i think i should stop because i have finished the instruction."}:
        return "stop", 0
    match = re.fullmatch(r"(?:the next action is )?move forward (25|50|75) cm\.?", text)
    if match:
        return "forward", int(match[1])
    match = re.fullmatch(r"(?:the next action is )?turn (left|right) (15|30|45) degrees?\.?", text)
    if match:
        return match[1], int(match[2])
    raise ValueError(f"Unrecognized action; refusing movement: {text!r}")


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


def infer(host, port, images, instruction, timeout):
    payload = json.dumps({"images": images, "query": instruction}).encode()
    started = time.monotonic()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        # A wall-time budget as well as per-read timeout prevents slow trickles.
        deadline = started + timeout
        sock.sendall(struct.pack(">Q", len(payload)) + payload)
        size = struct.unpack(">Q", receive_exact(sock, 8))[0]
        if not 0 < size <= 65536:
            raise ValueError(f"Invalid response size: {size}")
        sock.settimeout(max(0.01, deadline - time.monotonic()))
        response = json.loads(receive_exact(sock, size))
    elapsed = time.monotonic() - started
    if elapsed > timeout:
        raise TimeoutError("Inference exceeded deadline")
    return response, elapsed


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
        last_stamp = None
        try:
            client = ComposedCameraClientSensor(self.host, self.port)
            while not self.stop.is_set():
                item = client.read(blocking=False, new_only=True)
                if item:
                    frame = item.get("images", {}).get(self.key)
                    stamp = item.get("timestamps", {}).get(self.key)
                    if frame is not None and stamp is not None and stamp != last_stamp:
                        last_stamp = stamp
                        self.sample = (time.monotonic(), frame.copy(), stamp)
                self.stop.wait(0.005)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            if client is not None:
                client.close()

    def fresh(self):
        sample = self.sample
        return self.error is None and sample is not None and time.monotonic() - sample[0] < 0.5

    def snapshot(self):
        sample = self.sample
        if not self.fresh():
            raise RuntimeError(f"Ego camera missing/stale: {self.error}")
        return encode_image(sample[1]), sample[0], list(sample[1].shape), sample[2]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


class WBCPublisher:
    """50 Hz publication independent of camera/model request latency."""
    def __init__(self, camera):
        from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC, LOWER_BODY_POLICY_STATUS_TOPIC
        from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
        self.manager = ROSManager("NaVILANavigation")
        self.topic = CONTROL_GOAL_TOPIC
        self.pub = ROSMsgPublisher(self.topic)
        self.sub = ROSMsgSubscriber(LOWER_BODY_POLICY_STATUS_TOPIC)
        self.camera = camera
        self.segment = (0.0, (0.0, 0.0, 0.0))
        self.status = None
        self.fault = None
        self.active = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def ready(self):
        state = self.status
        return (state is not None and time.monotonic() - state[0] < 0.25
                and state[1].get("navila_guard") is True
                and state[1].get("control_mode") == "running"
                and state[1].get("use_policy_action") is True)

    def run(self):
        try:
            while not self.stop.is_set():
                now = time.monotonic()
                item = self.sub.get_msg()
                if item is not None:
                    self.status = (now, item)
                if self.manager.node.count_publishers(self.topic) > 1:
                    self.fault = "Competing control publisher; stop PICO/replay before navigation"
                if self.active and (not self.ready() or not self.camera.fresh()):
                    self.fault = "Controller not running/fresh or camera stale; restart explicitly"
                deadline, velocity = self.segment
                if self.fault or not self.active or not self.ready() or now >= deadline:
                    velocity = (0.0, 0.0, 0.0)
                self.pub.publish({"navigate_cmd": list(velocity), "navila": {
                    "sent_monotonic": now, "expires_monotonic": now + 0.2}})
                self.stop.wait(0.02)
        except Exception as exc:
            self.fault = repr(exc)

    def wait_ready(self):
        log("waiting_for_manual_standing", timeout_s=120, key="] in controller pane after ready")
        deadline = time.monotonic() + 120
        while not self.ready():
            if self.fault:
                raise RuntimeError(self.fault)
            if time.monotonic() >= deadline:
                raise TimeoutError("Guarded controller not manually enabled within 120 s")
            time.sleep(0.05)
        self.active = True

    def move(self, action, amount):
        if self.fault or not self.ready() or not self.camera.fresh():
            raise RuntimeError(self.fault or "Not ready to move")
        # Initial commissioning caps; distance/yaw are time based, not odometry.
        if action == "forward":
            actual = min(amount, 25)
            velocity, duration = (0.15, 0.0, 0.0), actual / 100 / 0.15
        else:
            actual = min(amount, 15)
            velocity = (0.0, 0.0, 0.25 if action == "left" else -0.25)
            duration = math.radians(actual) / 0.25
        deadline = time.monotonic() + duration
        self.segment = (deadline, velocity)
        log("segment", action=action, requested=amount, capped=actual,
            velocity=velocity, duration_s=duration)
        try:
            while time.monotonic() < deadline:
                if self.fault or not self.thread.is_alive():
                    raise RuntimeError(self.fault or "Publisher stopped")
                time.sleep(0.02)
        finally:
            self.segment = (0.0, (0.0, 0.0, 0.0))
        time.sleep(0.5)

    def close(self):
        self.active = False
        self.segment = (0.0, (0.0, 0.0, 0.0))
        time.sleep(0.25)
        self.stop.set()
        self.thread.join(timeout=2)
        self.manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", required=True, help="English navigation instruction")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-key", default="ego_view")
    parser.add_argument("--image", type=Path, help="Offline RGB image for inference only")
    parser.add_argument("--execute", action="store_true", help="Publish navigation after manual standing enable")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100 or not 0 < args.timeout <= 30:
        parser.error("steps must be 1..100 and timeout (0, 30]")
    if args.execute and args.image:
        parser.error("Execution requires a live camera")
    camera, wbc = None, None
    history = []
    try:
        if not args.image:
            camera = Camera(args.camera_host, args.camera_port, args.camera_key)
            deadline = time.monotonic() + 10
            while not camera.fresh():
                if camera.error or time.monotonic() > deadline:
                    raise RuntimeError(f"No fresh {args.camera_key} within 10 s: {camera.error}")
                time.sleep(0.05)
        if args.execute:
            wbc = WBCPublisher(camera)
            wbc.wait_ready()
        for step in range(args.steps):
            if camera:
                encoded, captured_at, shape, source_stamp = camera.snapshot()
            else:
                bgr = cv2.imread(str(args.image))
                if bgr is None:
                    raise ValueError("Cannot read offline image")
                encoded = encode_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                captured_at, shape, source_stamp = time.monotonic(), list(bgr.shape), None
            history.append(encoded)
            response, latency = infer(args.host, args.port, sample_history(history),
                                      args.instruction, args.timeout)
            action, amount = parse_action(response)
            log("inference", step=step + 1, response=response, action=action,
                amount=amount, round_trip_s=latency, camera_shape=shape,
                camera_source_timestamp=source_stamp, execute=args.execute)
            if wbc:
                if wbc.fault or not camera.fresh() or time.monotonic() - captured_at > 4:
                    raise RuntimeError(wbc.fault or "Observation/model response too old; no movement")
                if action == "stop":
                    break
                wbc.move(action, amount)
            elif step + 1 < args.steps:
                time.sleep(0.5)
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
        log("stopped_by_operator")
    except Exception as exc:
        log("failed_closed", error=str(exc))
        raise
