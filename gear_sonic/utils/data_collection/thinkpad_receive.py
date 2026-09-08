"""ThinkPad-only inputs and receive-time recording; no robot command publishers."""
from __future__ import annotations

import json
from pathlib import Path
import select
import sys
import threading
import time
import uuid


class LatestInputs:
    """One immutable snapshot per stream; overwriting is intentional latest sampling."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.image = None
        self.state_seq = 0
        self.error = None

    def put_state(self, data, received_ns):
        with self.lock:
            self.state_seq += 1
            self.state = (data, received_ns, self.state_seq)

    def put_image(self, image, received_ns, frame_id, source_ms, domain):
        with self.lock:
            self.image = (image, received_ns, frame_id, source_ms, domain)

    def snapshot(self):
        with self.lock:
            return time.monotonic_ns(), self.state, self.image, self.error


def assess_snapshot(tick, state, image, max_age_ns):
    if state is None or image is None:
        return "waiting_for_inputs"
    ages = (tick - state[1], tick - image[1])
    if min(ages) < 0:
        return "invalid_receive_timestamp"
    if max(ages) > max_age_ns:
        return "stale_inputs"
    return None


def hand_command_to_pose(command, profiles):
    """Match robot command_sender.py::legacy_dex3_to_inspire (2026-09-08)."""
    import numpy as np
    q = np.asarray(command, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError("Expected seven legacy hand command values")
    if np.max(np.abs(q)) <= 0.05:
        key = "open"
    elif q[0] <= -0.45:
        key = "grasp"
    elif q[0] >= 0.45:
        key = "grip"
    elif abs(q[0]) <= 0.05 and q[3] <= -0.95 and q[5] <= -0.95:
        key = "grip_pressed"
    else:
        key = "grasp"
    fallback = "open" if key == "grip" else "grasp"
    return profiles.get(key, profiles[fallback]).copy()


def validate_robot_state(data, num_joints):
    """Reject incomplete payloads instead of recording model defaults as measurements."""
    import numpy as np

    for key in ("q", "left_hand_inspire_q", "right_hand_inspire_q"):
        if key not in data:
            return f"missing_state_field:{key}"
        values = np.asarray(data[key])
        expected = num_joints if key == "q" else 6
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            return f"invalid_state_field:{key}"
    action_key = next((k for k in ("action", "last_action", "q_des", "target_q")
                       if k in data), None)
    if action_key is None:
        return "missing_action"
    action = np.asarray(data[action_key])
    if action.shape != (num_joints,) or not np.all(np.isfinite(action)):
        return "invalid_action"
    return None


class ThinkPadInputs(LatestInputs):
    def __init__(self, node, config):
        super().__init__()
        import msgpack
        import msgpack_numpy as mnp
        from std_msgs.msg import ByteMultiArray

        self.node = node
        self.config = config
        self.stop = threading.Event()

        def callback(message):
            received = time.monotonic_ns()
            try:
                raw = message.data
                packed = bytes(raw) if not raw or isinstance(raw[0], int) else b"".join(raw)
                data = msgpack.unpackb(packed, object_hook=mnp.decode, raw=False)
                if not isinstance(data, dict):
                    raise ValueError("Robot state must be a dictionary")
                self.put_state(data, received)
            except Exception as exc:
                with self.lock:
                    self.error = f"state_decode: {exc}"

        self.subscription = node.create_subscription(
            ByteMultiArray, config.state_topic_name, callback, 1
        )
        self.thread = threading.Thread(target=self._camera, daemon=True)
        self.thread.start()

    def _camera(self):
        import numpy as np
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        started = False
        try:
            camera_config = rs.config()
            serial = self.config.realsense_serial
            if not serial:
                devices = rs.context().query_devices()
                if len(devices) != 1:
                    raise RuntimeError("Exactly one RealSense required, or set --realsense-serial")
                serial = devices[0].get_info(rs.camera_info.serial_number)
            camera_config.enable_device(serial)
            camera_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(camera_config)
            started = True
            previous_id = None
            while not self.stop.is_set():
                frames = pipeline.wait_for_frames(1000)
                frame = frames.get_color_frame()
                received = time.monotonic_ns()
                if not frame:
                    continue
                frame_id = frame.get_frame_number()
                if frame_id == previous_id:
                    continue
                if previous_id is not None and frame_id < previous_id:
                    raise RuntimeError("Camera frame counter reset; restart recorder")
                previous_id = frame_id
                pixels = np.asanyarray(frame.get_data()).copy()
                self.put_image(pixels, received, frame_id, frame.get_timestamp(),
                               str(frame.get_frame_timestamp_domain()))
        except Exception as exc:
            with self.lock:
                self.error = f"camera: {exc}"
        finally:
            if started:
                pipeline.stop()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.node.destroy_subscription(self.subscription)


def run_local(collector):
    """Local line-based keys only; source times never participate in matching."""
    import rclpy

    config = collector.config
    root = Path(collector.data_exporter.root)
    audit_path = root / "receive_time.jsonl"
    audit = audit_path.open("x", encoding="utf-8", buffering=1)
    inputs = None
    session = str(uuid.uuid4())
    max_age_ns = int(config.receive_max_age * 1e9)
    last_tick = None
    last_image = None
    segment = None
    stdin_open = True
    waiting_reason = None

    def emit(kind, **values):
        audit.write(json.dumps(dict(event=kind, session_id=session, **values)) + "\n")

    def recording():
        return collector._episode_state.get_state() == collector._episode_state.RECORDING

    def finish(reason, discarded=False):
        nonlocal segment, last_tick
        if not recording():
            return
        episode = int(collector.current_episode_index)
        size = collector.data_exporter.episode_buffer["size"]
        emit("episode_end_requested", episode_index=episode, segment_id=segment,
             reason=reason, frames=size, discarded=discarded)
        if size:
            if discarded:
                collector.data_exporter.save_episode_as_discarded()
            else:
                collector.data_exporter.save_episode()
        collector._episode_state.reset_state()
        collector._initial_yaw = None
        collector._episode_init_base_quat = None
        last_tick = None
        emit("episode_end", episode_index=episode, segment_id=segment,
             reason=reason, frames=size, discarded=discarded)
        print(f"Recording ended: {reason}; frames={size}", flush=True)

    emit("session_start", wall_time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
         alignment_mode="host_receive_latest", delay_compensation=False,
         hand_feedback_independent_age="unknown", camera_format="rgb8/640x480/30")
    print("Local recorder: c + Enter start/save; x + Enter discard; q + Enter save/quit.",
          flush=True)
    try:
        inputs = ThinkPadInputs(collector.node, config)
        while rclpy.ok():
            # Snapshot is atomic; no producer can change objects selected for this row.
            tick, state, image, error = inputs.snapshot()
            reason = error or assess_snapshot(tick, state, image, max_age_ns)
            if reason is None:
                reason = validate_robot_state(state[0], collector.robot_model.num_joints)
            if collector.node.count_publishers(config.state_topic_name) != 1:
                reason = "state_topic_requires_exactly_one_publisher"
            if reason != waiting_reason:
                print(f"Inputs: {reason or 'ready'}", flush=True)
                waiting_reason = reason
            if recording() and last_tick is not None:
                if tick - last_tick > max_age_ns:
                    reason = reason or "recording_loop_gap"
            if reason and recording():
                finish(reason)
            command = ""
            if stdin_open and select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if not line:
                    stdin_open = False
                    command = "q"
                else:
                    command = line.strip().lower()
            if command == "q":
                finish("user_quit")
                break
            if command == "x":
                finish("user_discard", discarded=True)
            if command == "c":
                if recording():
                    finish("user_stop")
                elif reason is None:
                    segment = str(uuid.uuid4())
                    collector._episode_state.change_state()
                    collector._initial_yaw = None
                    collector._episode_init_base_quat = None
                    last_image = None
                    emit("episode_start", episode_index=int(collector.current_episode_index),
                         segment_id=segment, monotonic_ns=tick)
                    print("Recording started", flush=True)
                else:
                    print(f"Cannot start: {reason}", flush=True)
            if recording() and reason is None:
                collector.latest_proprio_msg = state[0]
                collector.latest_image_msg = {"images": {"ego_view": image[0]}}
                episode = int(collector.current_episode_index)
                index = collector.data_exporter.episode_buffer["size"]
                collector._add_data_frame()
                if collector.data_exporter.episode_buffer["size"] != index + 1:
                    raise RuntimeError("Exporter did not append exactly one frame")
                emit("frame", episode_index=episode, frame_index=index, segment_id=segment,
                     tick_monotonic_ns=tick, state_receive_monotonic_ns=state[1],
                     state_receive_sequence=state[2],
                     raw_robot_q=list(map(float, state[0]["q"])),
                     raw_robot_action=list(map(float, next(state[0][k] for k in
                         ("action", "last_action", "q_des", "target_q") if k in state[0]))), image_receive_monotonic_ns=image[1],
                     image_frame_number=image[2], image_source_timestamp_ms=image[3],
                     image_source_clock_domain=image[4],
                     repeated_image=image[2] == last_image,
                     receive_delta_ms=(image[1] - state[1]) / 1e6,
                     state_age_ms=(tick - state[1]) / 1e6,
                     image_age_ms=(tick - image[1]) / 1e6)
                last_image = image[2]
                last_tick = tick
            if error:
                raise RuntimeError(error)
            remaining = collector.loop_period - (time.monotonic_ns() - tick) / 1e9
            if remaining > 0:
                time.sleep(remaining)
        finish("ros_shutdown")
    except KeyboardInterrupt:
        finish("keyboard_interrupt")
    except Exception:
        # Do not retry a partially failed save or report an incomplete segment as valid.
        emit("session_error", segment_id=segment)
        raise
    finally:
        if inputs is not None:
            inputs.close()
        collector.data_exporter.stop_video_writers()
        emit("session_end", monotonic_ns=time.monotonic_ns())
        audit.close()
