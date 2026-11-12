"""Asynchronous JAR inference and 50Hz rate-limited G1 position references."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

from control import ReferenceGuard
from robot import UnitreeRobot
from rpc import Client
from sensors import Cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--dds-domain", type=int, default=0)
    parser.add_argument("--camera-endpoint", required=True)
    parser.add_argument("--policy-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--profile", default=str(Path(__file__).with_name("g1_dex1.json")))
    parser.add_argument("--live", action="store_true", help="Explicitly enable arm_sdk/Dex1 writers after confirmation")
    parser.add_argument("--duration", type=float, default=20)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    if args.live:
        if not sys.stdin.isatty() or input("G1 must be standing under its native controller. Type ARM to enable only arms/Dex1: ").strip() != "ARM":
            raise SystemExit("Motor outputs were not enabled")
        control_lock = Path(__file__).resolve().parents[2].joinpath("runtime/robot-control.lock").open("a")
        try:
            fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Checkpoint validation or another live controller is active; wait or stop-sync before retrying")
    stopped = threading.Event()
    def stop(signum, frame):
        stopped.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    robot = UnitreeRobot(args.interface, profile, live=args.live, domain=args.dds_domain)
    cameras = Cameras(args.camera_endpoint)
    guard = ReferenceGuard(profile["low"], profile["high"], profile["arm_max_speed"], profile["gripper_max_speed"],
                           profile["max_observation_age"], profile["max_prediction_age"])
    latest, lock = {}, threading.Lock()
    def infer():
        client = Client(args.policy_endpoint)
        try:
            while not stopped.is_set():
                try:
                    observed, age = robot.observe()
                    if age > profile["max_observation_age"]:
                        raise TimeoutError("Stale robot state")
                    images, ages = cameras.history()
                    captured = time.monotonic()
                    result = client.predict(observed, images, ages, steps=1)
                    with lock:
                        latest.update(reference=result["positions"][0], captured=captured, latency=result["round_trip_seconds"])
                except TimeoutError:
                    time.sleep(.05)
        except Exception as error:
            print(json.dumps(dict(event="inference_stopped", error=str(error))), flush=True)
            stopped.set()
        finally:
            client.close()
    worker = threading.Thread(target=infer, name="policy-worker")
    worker.start()
    last = None
    started, printed = time.monotonic(), 0.
    try:
        while not stopped.is_set() and time.monotonic() - started < args.duration:
            tick = time.monotonic()
            try:
                observed, age = robot.observe()
            except TimeoutError:
                if tick - started > 10:
                    raise
                time.sleep(.02)
                continue
            if guard.last is None:
                guard.reset(observed)
            with lock:
                prediction = latest.copy()
            target = prediction.get("reference", observed)
            prediction_age = tick - prediction.get("captured", tick - 999)
            last, mode = guard.step(observed, target, .02, age, prediction_age)
            if args.live:
                robot.send(last)
            if tick - printed >= 1:
                print(json.dumps(dict(mode="live" if args.live else "read-only", guard=mode,
                     observation_age=age, prediction_age=prediction_age, latency=prediction.get("latency"))), flush=True)
                printed = tick
            time.sleep(max(0, .02 - (time.monotonic() - tick)))
    finally:
        stopped.set()
        worker.join(timeout=17)
        cameras.close()
        robot.close(last)


if __name__ == "__main__":
    main()
