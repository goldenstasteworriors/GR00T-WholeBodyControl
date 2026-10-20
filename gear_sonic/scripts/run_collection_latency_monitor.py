"""Read-only collection timing monitor. No actuator publishers or mode changes."""

from collections import defaultdict, deque
from pathlib import Path
import json
import os
import signal
import time

import msgpack
import msgpack_numpy as mnp
import rclpy
from std_msgs.msg import ByteMultiArray
import tyro

from decoupled_wbc.control.main.constants import PIPELINE_LATENCY_TOPIC
from decoupled_wbc.control.utils.pipeline_latency import summarize


def main(duration: float = 20.0, output_json: Path | None = None, ros_domain_id: int = 73):
    os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
    rclpy.init()
    node = rclpy.create_node("collection_latency_monitor")
    values = defaultdict(lambda: deque(maxlen=3000))
    tracking = deque(maxlen=3000)
    frames = deque(maxlen=3000)
    samples = 0
    latest_mode = None
    latest_source = None
    first_sample = None
    last_sample = None

    def receive(message):
        nonlocal samples, latest_mode, latest_source, first_sample, last_sample
        record = msgpack.unpackb(b"".join(message.data), object_hook=mnp.decode, raw=False)
        now = time.monotonic()
        samples += 1
        first_sample = now if first_sample is None else first_sample
        last_sample = now
        latest_mode = record["control_mode"]
        source = record.get("pico_source_timestamp_ns")
        new_source = source is not None and source != latest_source
        if new_source:
            frames.append(now)
            latest_source = source
        for name, value in record["metrics_ms"].items():
            if name in {"pico_source_interval_ms", "pico_observed_interval_ms"} and not new_source:
                continue
            values[name].append(value)
        tracking.append(record["tracking_error_rms_rad"])

    node.create_subscription(ByteMultiArray, PIPELINE_LATENCY_TOPIC, receive, 10)

    def report():
        rate = (samples - 1) / (last_sample - first_sample) if samples > 1 else None
        pico_rate = (len(frames) - 1) / (frames[-1] - frames[0]) if len(frames) > 1 else None
        return {
            "samples": samples, "control_mode": latest_mode,
            "control_observed_hz": rate, "pico_new_frames_observed_hz": pico_rate,
            "window_samples_max": 3000,
            "metrics_ms": {name: summarize(series) for name, series in values.items()},
            "tracking_error_rms_rad": summarize(tracking) if tracking else None,
            "limits": [
                "PICO intervals are not absolute wireless one-way latency.",
                "DDS write call duration is not wired transit time or robot execution acknowledgment.",
                "Joint tracking error is not mechanical response latency.",
                "All cross-process timing spans use the same ThinkPad monotonic clock.",
            ],
        }

    def emit():
        result = report()
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2) + "\n")
        print("[PipelineLatency] " + json.dumps(result), flush=True)

    def interrupted(number, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    last_report = start
    try:
        while rclpy.ok() and (duration <= 0 or time.monotonic() - start < duration):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - last_report >= 2.0:
                emit()
                last_report = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        emit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
