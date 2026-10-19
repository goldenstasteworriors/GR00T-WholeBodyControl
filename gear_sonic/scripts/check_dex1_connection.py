"""Read-only robot/Dex1 DDS probe. Never creates command publishers."""

import argparse
import json
import socket
import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def check(network: str, duration: float = 5.0) -> bool:
    if network not in dict(socket.if_nameindex()).values():
        raise ValueError(f"Network interface does not exist: {network}")
    ChannelFactoryInitialize(0, network)
    samples, lock = {}, threading.Lock()

    def receive(topic, msg):
        motors = msg.motor_state if topic == "rt/lowstate" else msg.states
        q = [float(m.q) for m in motors]
        with lock:
            count = samples.get(topic, {}).get("count", 0) + 1
            samples[topic] = dict(count=count, stamp=time.monotonic(), q=q)

    topics = [("rt/lowstate", LowState_)] + [
        (f"rt/dex1/{side}/state", MotorStates_) for side in ("left", "right")
    ]
    subscribers = []
    try:
        for topic, message_type in topics:
            sub = ChannelSubscriber(topic, message_type)
            sub.Init(lambda msg, topic=topic: receive(topic, msg), 10)
            subscribers.append(sub)
        time.sleep(duration)
        ok = True
        with lock:
            now = time.monotonic()
            for topic, _ in topics:
                sample = samples.get(topic)
                if sample is None:
                    print(f"FAIL {topic}: no state received", flush=True)
                    ok = False
                    continue
                age = now - sample["stamp"]
                valid = sample["count"] >= 2 and age < 0.25 and np.all(np.isfinite(sample["q"]))
                if topic != "rt/lowstate":
                    valid = valid and len(sample["q"]) == 1 and -0.2 <= sample["q"][0] <= 5.6
                else:
                    valid = valid and len(sample["q"]) >= 29
                print(json.dumps(dict(topic=topic, valid=bool(valid), count=sample["count"],
                                      age_s=round(age, 4), q=sample["q"])), flush=True)
                ok = ok and valid
        return bool(ok)
    finally:
        for sub in subscribers:
            sub.Close()


def check_camera(host: str, port: int, timeout: float = 5.0,
                 record_wrist_cameras: bool = True, sync_tolerance_ms: float = 30.0) -> bool:
    """Require complete RGB groups with advancing per-view timestamps."""
    import msgpack
    import msgpack_numpy
    import zmq

    from gear_sonic.camera.sensor_server import ImageMessageSchema
    from gear_sonic.camera.synchronization import THREE_CAMERA_MOUNTS, validate_rgb_bundle

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.setsockopt(zmq.LINGER, 0)
    mounts = THREE_CAMERA_MOUNTS if record_wrist_cameras else ("ego_view",)
    previous_stamps = None
    advancing_frames = 0
    try:
        subscriber.connect(f"tcp://{host}:{port}")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not subscriber.poll(100):
                continue
            payload = msgpack.unpackb(subscriber.recv(), object_hook=msgpack_numpy.decode)
            frame = ImageMessageSchema.deserialize(payload)
            stamps = validate_rgb_bundle(frame.asdict(), mounts, sync_tolerance_ms / 1000)
            if previous_stamps is not None and all(a > b for a, b in zip(stamps, previous_stamps)):
                advancing_frames += 1
            else:
                advancing_frames = 0
            previous_stamps = stamps
            if advancing_frames >= 5:
                print(f"Camera groups OK: {list(mounts)}, RGB 480x640, "
                      f"skew={(max(stamps)-min(stamps))*1000:.2f}ms, advancing all timestamps")
                return True
        print(f"FAIL camera {host}:{port}: no continuously updating complete groups: {list(mounts)}")
        return False
    except Exception as exc:
        print(f"FAIL camera {host}:{port}: {exc}")
        return False
    finally:
        subscriber.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--camera-host", default="")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--record-wrist-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-sync-tolerance-ms", type=float, default=30.0)
    args = parser.parse_args()
    if not 0 < args.duration <= 60:
        parser.error("duration must be in (0, 60]")
    if not np.isfinite(args.camera_sync_tolerance_ms) or args.camera_sync_tolerance_ms < 0:
        parser.error("camera sync tolerance must be finite and nonnegative")
    ok = check(args.network, args.duration)
    if args.camera_host:
        ok = check_camera(args.camera_host, args.camera_port,
                          record_wrist_cameras=args.record_wrist_cameras,
                          sync_tolerance_ms=args.camera_sync_tolerance_ms) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
