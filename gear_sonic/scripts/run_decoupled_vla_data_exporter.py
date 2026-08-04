"""
Decoupled WBC data exporter for the Sonic VLA LeRobot schema.

Data sources:
  1. Robot/WBC state -> ROS topic ``G1Env/env_state_act`` from
     decoupled_wbc.control.main.teleop.run_g1_control_loop.
  2. PICO/SMPL pose -> optional ZMQ SUB on ``pose``, ``planner`` and
     ``manager_state`` topics from gear_sonic/scripts/pico_manager_thread_server.py.
  3. Camera -> gear_sonic.camera.composed_camera.ComposedCameraClientSensor.

The output feature keys and modality config intentionally match
``gear_sonic/scripts/run_data_exporter.py`` so the generated dataset can be
consumed by the same Sonic VLA post-training pipeline. Decoupled-only runs do
not produce Sonic motion tokens or planner semantics; those fields are written
with schema-compatible defaults unless a matching PICO/manager message exists.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Any
import wave

import msgpack
import msgpack_numpy as mnp
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
import tyro
import zmq

from decoupled_wbc.control.main.constants import ROBOT_CONFIG_TOPIC, STATE_TOPIC_NAME
from decoupled_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from decoupled_wbc.control.utils.ros_utils import ROSMsgSubscriber
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    get_features_sonic_body29,
    get_features_sonic_inspire6,
    get_g1_robot_model,
    get_modality_config_sonic_body29,
    get_modality_config_sonic_inspire6,
    get_wrist_camera_features,
    get_wrist_camera_modality_config,
)
from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.inspire_hand_tasks import DEFAULT_HAND_TASK
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.text_to_speech import TextToSpeech
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d


IDENTITY_QUAT_F64 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
IDENTITY_QUAT_F32 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_PROJECTED_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)


@dataclass
class DecoupledVLADataExporterConfig:
    """CLI config for decoupled WBC -> Sonic VLA data export."""

    dataset_name: str | None = None
    """Dataset name. Defaults to a timestamp."""

    task_prompt: str = "demo"
    """Language task prompt stored in the LeRobot dataset."""

    root_output_dir: str = "outputs"
    """Root output directory."""

    data_collection_frequency: int = 50
    """Data collection frequency in Hz."""

    overwrite_existing_dataset: bool = False
    """Delete and recreate the dataset directory if it already exists."""

    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    state_topic_name: str = STATE_TOPIC_NAME
    """ROS topic published by the decoupled control loop."""

    robot_config_service_name: str = ROBOT_CONFIG_TOPIC
    """ROS service that returns the control-loop config for dataset metadata."""

    robot_config_timeout: float = 0.0
    """Seconds to wait for robot config service. 0 means wait forever."""

    sonic_zmq_host: str = "localhost"
    """Host for optional PICO/SMPL ZMQ pose messages."""

    sonic_zmq_port: int = 5556
    """Port for optional PICO/SMPL ZMQ pose messages."""

    sonic_pose_max_age: float = 0.2
    """Max accepted age in seconds for PICO/SMPL pose messages."""

    planner_max_age: float = 0.2
    """Max accepted age in seconds for planner/VR3PT messages."""

    use_sonic_pose_when_stream_off: bool = True
    """Use PICO pose data even if no Sonic manager_state stream mode is available."""

    default_stream_mode_when_pose_available: int = 1
    """stream_mode value to write when pose exists but manager_state is absent/off."""

    require_sonic_pose: bool = False
    """When True, skip recording frames until a fresh PICO pose message is available."""

    record_wrist_cameras: bool = False
    """Record left_wrist/right_wrist camera streams if the camera server provides them."""

    with_hands: bool = True
    """Record native Inspire hand state/action fields when hand control is enabled."""

    text_to_speech: bool = False
    """Use optional text-to-speech voice feedback; local tone cues are separate."""

    audio_cues: bool = True
    """Use local tone cues for start, stop and discard events."""

    audio_cue_volume: float = 0.35
    """Volume for start/stop tone cues."""

    discard_audio_cue_volume: float = 0.9
    """Volume for the discard tone cue."""

    profile_timing: bool = False
    """Print periodic timing breakdown."""

    profile_interval: float = 1.0
    """Seconds between timing profile log lines."""


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        super().__init__(f"{failure_count} failures in {reset_timeout_sec} seconds")


class AudioCue:
    """Serialize local tone cues so one state change cannot cut off another."""

    def __init__(self, volume: float = 0.35, sample_rate: int = 44100):
        self.volume = volume
        self.sample_rate = sample_rate
        self._audio_cmd = None
        try:
            import sounddevice as sd
        except Exception as exc:
            self._sd = None
            # Prefer the native PipeWire client on the collection laptop.
            if shutil.which("pw-play"):
                self._audio_cmd = ["pw-play", "-"]
            elif shutil.which("paplay"):
                self._audio_cmd = ["paplay", "-"]
            elif shutil.which("aplay"):
                self._audio_cmd = ["aplay", "-q", "-"]
            if self._audio_cmd is None:
                print(f"[AudioCue] disabled: sounddevice unavailable ({exc}); no aplay/paplay fallback")
        else:
            self._sd = sd

        self.patterns = {
            "start": [(880, 0.08), (1175, 0.10)],
            "stop": [(1175, 0.08), (880, 0.12)],
            "discard": [(260, 0.18), (0, 0.04), (180, 0.24), (0, 0.04), (140, 0.28)],
        }
        self._playback_queue: queue.Queue[tuple[list[tuple[int, float]], float]] = queue.Queue()
        self._playback_worker = threading.Thread(target=self._run_playback_worker, daemon=True)
        self._playback_worker.start()

    def play(self, cue: str, *, volume: float | None = None) -> None:
        pattern = self.patterns.get(cue)
        if pattern is None:
            return
        self._playback_queue.put((pattern, self.volume if volume is None else volume))

    def _run_playback_worker(self) -> None:
        while True:
            pattern, volume = self._playback_queue.get()
            try:
                self._play_pattern(pattern, volume)
            finally:
                self._playback_queue.task_done()

    def _play_pattern(self, pattern: list[tuple[int, float]], volume: float) -> None:
        try:
            samples = self._make_samples(pattern, volume)
            if self._sd is not None:
                # sounddevice has one shared output stream. Blocking inside the
                # dedicated worker keeps a later cue from interrupting this one.
                self._sd.play(samples, self.sample_rate, blocking=True)
            elif self._audio_cmd is not None:
                wav_data = self._samples_to_wav(samples)
                last_error = "unknown playback error"
                for _ in range(2):
                    result = subprocess.run(
                        self._audio_cmd,
                        input=wav_data,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=3.0,
                    )
                    if result.returncode == 0:
                        return
                    last_error = result.stderr.decode(errors="replace").strip() or (
                        f"exit code {result.returncode}"
                    )
                raise RuntimeError(f"{' '.join(self._audio_cmd)}: {last_error}")
        except Exception as exc:
            print(f"[AudioCue] failed to play cue: {exc}")

    def _make_samples(self, pattern: list[tuple[int, float]], volume: float) -> np.ndarray:
        chunks = []
        for freq, duration in pattern:
            n_samples = max(1, int(self.sample_rate * duration))
            if freq <= 0:
                chunks.append(np.zeros(n_samples, dtype=np.float32))
                continue
            t = np.linspace(0.0, duration, n_samples, endpoint=False)
            chunks.append((np.sin(2.0 * np.pi * freq * t) * volume).astype(np.float32))
        return np.concatenate(chunks)

    def _samples_to_wav(self, samples: np.ndarray) -> bytes:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buf.getvalue()


class TimingThresholdMonitor:
    def __init__(
        self,
        max_failures: int = 3,
        reset_timeout_sec: float = 5.0,
        time_delta: float = 0.2,
        raise_exception: bool = False,
    ):
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.time_delta = time_delta
        self.raise_exception = raise_exception
        self.failure_count = 0
        self.last_failure_time = 0.0

    def reset(self) -> None:
        self.failure_count = 0
        self.last_failure_time = 0.0

    def log_time_delta(self, time_delta_sec: float) -> None:
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in "
                f"{self.reset_timeout_sec} seconds, time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self) -> bool:
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


def _as_1d_array(
    value: Any,
    *,
    dtype: np.dtype | type,
    length: int | None = None,
    default: np.ndarray | None = None,
) -> np.ndarray:
    if value is None:
        if default is not None:
            return np.asarray(default, dtype=dtype).reshape(-1).copy()
        if length is None:
            return np.array([], dtype=dtype)
        return np.zeros(length, dtype=dtype)

    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if length is not None and arr.size != length:
        if arr.size > length:
            arr = arr[:length]
        else:
            padded = np.zeros(length, dtype=dtype)
            padded[: arr.size] = arr
            arr = padded
    return np.ascontiguousarray(arr, dtype=dtype)


def _valid_quat(value: Any, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    quat = _as_1d_array(value, dtype=dtype, length=4)
    if not np.all(np.isfinite(quat)) or np.linalg.norm(quat) < 1e-6:
        return IDENTITY_QUAT_F64.astype(dtype) if dtype == np.float64 else IDENTITY_QUAT_F32.copy()
    return quat


def _vr_orientation_to_rot6d(value: Any) -> np.ndarray:
    arr = _as_1d_array(value, dtype=np.float32)
    if arr.size == 18:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.size == 12:
        return np.ascontiguousarray(quat_to_rot6d(arr), dtype=np.float32)
    return np.zeros(18, dtype=np.float32)


def _body_state_joint_indices(robot_model, joint_names: list[str]) -> list[int]:
    """Map full robot-model DOF indices into the exported 29-DOF body state."""
    body_state_index = {
        full_dof_index: state_index
        for state_index, full_dof_index in enumerate(
            robot_model.get_body_actuated_joint_indices()
        )
    }
    return [body_state_index[robot_model.dof_index(name)] for name in joint_names]


def unpack_pose_message(packed_data: bytes, topic: str) -> dict:
    """Unpack pico_manager_thread_server packed topic messages."""
    header_size = 1280
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic {topic!r}")

    offset = len(topic_bytes)
    if len(packed_data) < offset + header_size:
        raise ValueError(f"Packed data too small: {len(packed_data)}")

    header_bytes = packed_data[offset : offset + header_size]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])
    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + header_size
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


def poll_robot_config_ros(service_name: str, timeout_sec: float) -> dict:
    """Read the decoupled control-loop config from its ROS Trigger service."""
    mnp.patch()
    node = rclpy.create_node("decoupled_vla_robot_config_client")
    client = node.create_client(Trigger, service_name)
    deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None

    try:
        while not client.wait_for_service(timeout_sec=1.0):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No robot config service {service_name!r} within {timeout_sec}s"
                )

        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for robot config response from {service_name!r}"
                )
            rclpy.spin_once(node, timeout_sec=0.1)

        result = future.result()
        if result is None:
            raise RuntimeError(f"Robot config service {service_name!r} returned no result")
        if not result.success:
            raise RuntimeError(f"Robot config service failed: {result.message}")

        decoded = base64.b64decode(result.message.encode("ascii"))
        config = msgpack.unpackb(decoded, object_hook=mnp.decode, raw=False)
        return config
    finally:
        node.destroy_node()


class DecoupledVLADataCollector:
    """Collect decoupled WBC frames into the Sonic VLA schema."""

    def __init__(
        self,
        *,
        node,
        config: DecoupledVLADataExporterConfig,
        data_exporter: Gr00tDataExporter,
        robot_model,
        text_to_speech: TextToSpeech | None = None,
        audio_cue: AudioCue | None = None,
    ):
        self.node = node
        self.config = config
        self.data_exporter = data_exporter
        self.robot_model = robot_model
        self.text_to_speech = text_to_speech
        self.audio_cue = audio_cue
        self.frequency = config.data_collection_frequency
        self.loop_period = 1.0 / self.frequency

        self._episode_state = EpisodeState()
        self._keyboard_listener = KeyboardListenerSubscriber()
        self._state_subscriber = ROSMsgSubscriber(config.state_topic_name)
        self._image_subscriber = ComposedCameraClientSensor(
            server_ip=config.camera_host,
            port=config.camera_port,
        )

        self.latest_proprio_msg: dict | None = None
        self.latest_image_msg: dict | None = None
        self.latest_sonic_msg: dict | None = None
        self.latest_planner_msg: dict | None = None
        self.current_stream_mode = 0
        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        self._sonic_zmq_ctx = zmq.Context()
        self._sonic_zmq_socket = self._sonic_zmq_ctx.socket(zmq.SUB)
        self._sonic_zmq_socket.connect(
            f"tcp://{config.sonic_zmq_host}:{config.sonic_zmq_port}"
        )
        self._sonic_zmq_socket.setsockopt(zmq.RCVTIMEO, 0)
        self._sonic_zmq_socket.setsockopt(zmq.CONFLATE, 0)
        self._sonic_zmq_socket.setsockopt(zmq.RCVHWM, 20)
        self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
        self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
        self._sonic_zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")

        self.telemetry = Telemetry(window_size=100)
        self.sonic_timing_monitor = TimingThresholdMonitor(
            max_failures=3,
            reset_timeout_sec=5.0,
            time_delta=config.sonic_pose_max_age,
        )
        self._last_latency_log_time = 0.0
        self._last_profile_log_time = time.monotonic()
        self._initial_yaw: float | None = None
        self._episode_init_base_quat: np.ndarray | None = None

        self._left_wrist_indices = _body_state_joint_indices(
            self.robot_model,
            [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
        )
        self._right_wrist_indices = _body_state_joint_indices(
            self.robot_model,
            [
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        )

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True, blocking: bool = False) -> None:
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say, blocking=blocking)
        else:
            print(message)

    def _play_audio_cue(self, cue: str) -> None:
        if self.audio_cue is None:
            return
        volume = self.config.discard_audio_cue_volume if cue == "discard" else None
        self.audio_cue.play(cue, volume=volume)

    def _poll_state_ros(self) -> None:
        msg = self._state_subscriber.get_msg()
        if msg is None:
            return
        if "timestamps" not in msg:
            msg["timestamps"] = {"proprio": time.time()}
        elif "proprio" not in msg["timestamps"]:
            msg["timestamps"]["proprio"] = time.time()
        self.latest_proprio_msg = msg

    def _poll_sonic_zmq_messages(self) -> None:
        max_polls = 20
        for _ in range(max_polls):
            try:
                raw = self._sonic_zmq_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            if raw.startswith(b"manager_state"):
                self._handle_manager_state(raw)
            elif raw.startswith(b"planner"):
                self._handle_planner_message(raw)
            elif raw.startswith(b"pose"):
                self._handle_pose_message(raw)

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, "manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            self.current_stream_mode = int(data["stream_mode"].flat[0])
        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    def _handle_planner_message(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, "planner")
        except Exception:
            return

        self.latest_planner_msg = {
            "planner_mode": int(data["mode"].flat[0]) if "mode" in data else 0,
            "planner_movement": (
                _as_1d_array(data.get("movement"), dtype=np.float32, length=3)
                if "movement" in data
                else np.zeros(3, dtype=np.float32)
            ),
            "planner_facing": (
                _as_1d_array(data.get("facing"), dtype=np.float32, length=3)
                if "facing" in data
                else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ),
            "planner_speed": float(data["speed"].flat[0]) if "speed" in data else -1.0,
            "planner_height": float(data["height"].flat[0]) if "height" in data else -1.0,
            "vr_3pt_position": (
                _as_1d_array(data.get("vr_position"), dtype=np.float32, length=9)
                if "vr_position" in data
                else None
            ),
            "vr_3pt_orientation": (
                _as_1d_array(data.get("vr_orientation"), dtype=np.float32)
                if "vr_orientation" in data
                else None
            ),
            "left_hand_joints": self._extract_hand_joints(data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(data, "right_hand_joints"),
            "receive_timestamp": time.time(),
        }

    def _handle_pose_message(self, raw: bytes) -> None:
        try:
            pose_data = unpack_pose_message(raw, "pose")
        except Exception as exc:
            print(f"[PICO] Error unpacking pose message: {exc}")
            return

        if self._extract_bool(pose_data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(pose_data, "toggle_data_abort"):
            self._manager_toggle_da = True

        has_smpl = "smpl_joints" in pose_data
        has_vr_3pt = "vr_position" in pose_data or "vr_orientation" in pose_data
        if not has_smpl and not has_vr_3pt:
            return

        smpl_joints = np.zeros((24, 3), dtype=np.float32)
        if has_smpl:
            smpl_joints = pose_data["smpl_joints"]
            if smpl_joints.ndim == 3:
                smpl_joints = smpl_joints[0]

        smpl_pose = np.zeros(63, dtype=np.float32)
        if "smpl_pose" in pose_data:
            raw_pose = pose_data["smpl_pose"]
            if raw_pose.ndim == 3:
                smpl_pose = raw_pose[0].reshape(-1).astype(np.float32)
            else:
                smpl_pose = raw_pose.reshape(-1).astype(np.float32)
            smpl_pose = _as_1d_array(smpl_pose, dtype=np.float32, length=63)

        frame_index = np.array([0], dtype=np.int64)
        if "frame_index" in pose_data:
            frame_index = np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)

        body_quat_w = None
        if "body_quat_w" in pose_data:
            body_quat_w = pose_data["body_quat_w"]
            if body_quat_w.ndim > 1:
                body_quat_w = body_quat_w[0]
            body_quat_w = _valid_quat(body_quat_w, dtype=np.float32)

        vr_3pt_position = None
        if "vr_position" in pose_data and pose_data["vr_position"].size == 9:
            vr_3pt_position = pose_data["vr_position"].reshape(-1).astype(np.float32)
        vr_3pt_orientation = None
        if "vr_orientation" in pose_data:
            vr_3pt_orientation = pose_data["vr_orientation"].reshape(-1).astype(np.float32)

        heading_increment = 0.0
        if "heading_increment" in pose_data:
            heading_increment = float(pose_data["heading_increment"].flat[0])

        self.latest_sonic_msg = {
            "has_smpl": has_smpl,
            "smpl_joints": np.asarray(smpl_joints, dtype=np.float32),
            "smpl_pose": smpl_pose,
            "body_quat_w": body_quat_w,
            "left_hand_joints": self._extract_hand_joints(pose_data, "left_hand_joints"),
            "right_hand_joints": self._extract_hand_joints(pose_data, "right_hand_joints"),
            "left_wrist_joints": self._extract_wrist_joints_from_pose_msg(
                pose_data,
                side="left",
            ),
            "right_wrist_joints": self._extract_wrist_joints_from_pose_msg(
                pose_data,
                side="right",
            ),
            "vr_3pt_position": vr_3pt_position,
            "vr_3pt_orientation": vr_3pt_orientation,
            "frame_index": frame_index,
            "heading_increment": heading_increment,
            "receive_timestamp": time.time(),
        }

    @staticmethod
    def _extract_bool(data: dict, key: str) -> bool:
        val = data.get(key)
        if val is None:
            return False
        if isinstance(val, np.ndarray):
            return bool(val.flat[0])
        return bool(val)

    @staticmethod
    def _extract_hand_joints(data: dict, key: str) -> np.ndarray:
        arr = data.get(key)
        if arr is None:
            return np.zeros(6, dtype=np.float32)
        if isinstance(arr, np.ndarray) and arr.ndim > 1:
            arr = arr[0]
        return _as_1d_array(arr, dtype=np.float32, length=6)

    def _extract_wrist_joints_from_pose_msg(self, data: dict, *, side: str) -> np.ndarray | None:
        if "joint_pos" not in data:
            return None
        joint_pos = np.asarray(data["joint_pos"])
        if joint_pos.ndim == 2:
            joint_pos = joint_pos[0]
        if joint_pos.size < 29:
            return None

        if side == "left":
            return np.array([joint_pos[23], joint_pos[25], joint_pos[27]], dtype=np.float32)
        return np.array([joint_pos[24], joint_pos[26], joint_pos[28]], dtype=np.float32)

    def _check_recording_commands(self) -> None:
        key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._initial_yaw = None
                self._episode_init_base_quat = None
                self._play_audio_cue("start")
                self._print_and_say(f"Started recording {self.current_episode_index}")
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._play_audio_cue("stop")
                self._print_and_say("Stopping recording, preparing to save")
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                pass
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._initial_yaw = None
                self._episode_init_base_quat = None
                self._play_audio_cue("discard")
                self._print_and_say("Discarded current recording")

    def _normalise_full_q(
        self,
        value: Any,
        *,
        default: np.ndarray | None = None,
    ) -> np.ndarray:
        num_joints = self.robot_model.num_joints
        arr = _as_1d_array(value, dtype=np.float64)
        if arr.size == num_joints:
            return arr

        body_indices = self.robot_model.get_body_actuated_joint_indices()
        if arr.size == len(body_indices):
            return self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=arr
            ).astype(np.float64)

        if default is not None:
            return _as_1d_array(default, dtype=np.float64, length=num_joints)
        return _as_1d_array(self.robot_model.q_zero, dtype=np.float64, length=num_joints)

    def _get_kinematic_state(self, proprio: dict) -> np.ndarray:
        """Return the legacy 43-DOF state used only for wrist FK."""
        return self._normalise_full_q(proprio.get("q"))

    @staticmethod
    def _get_inspire_hand_state(proprio: dict, side: str) -> np.ndarray:
        key = f"{side}_hand_inspire_q"
        if key not in proprio:
            raise RuntimeError(
                f"Missing {key!r} in the control-state message. "
                "The collector requires the native six-DOF Inspire hand state."
            )
        return _as_1d_array(proprio[key], dtype=np.float64, length=6)

    def _get_observation_state(self, proprio: dict, kinematic_state: np.ndarray) -> np.ndarray:
        body_indices = self.robot_model.get_body_actuated_joint_indices()
        if not self.config.with_hands:
            return np.ascontiguousarray(kinematic_state[body_indices], dtype=np.float64)
        return np.concatenate(
            [
                kinematic_state[body_indices],
                self._get_inspire_hand_state(proprio, "left"),
                self._get_inspire_hand_state(proprio, "right"),
            ]
        ).astype(np.float64, copy=False)

    def _get_action_wbc(self, proprio: dict, observation_state: np.ndarray) -> np.ndarray:
        hand_task = os.environ.get("SONIC_HAND_TASK", DEFAULT_HAND_TASK)
        for key in ("action", "last_action", "q_des", "target_q"):
            if key in proprio:
                legacy_action = self._normalise_full_q(proprio[key])
                body_indices = self.robot_model.get_body_actuated_joint_indices()
                if not self.config.with_hands:
                    return np.ascontiguousarray(
                        legacy_action[body_indices], dtype=np.float64
                    )
                left_indices = self.robot_model.get_hand_actuated_joint_indices("left")
                right_indices = self.robot_model.get_hand_actuated_joint_indices("right")
                return np.concatenate(
                    [
                        legacy_action[body_indices],
                        self._legacy_hand_action_to_inspire(legacy_action[left_indices], hand_task),
                        self._legacy_hand_action_to_inspire(legacy_action[right_indices], hand_task),
                    ]
                ).astype(np.float64, copy=False)
        return observation_state.copy()

    @staticmethod
    def _legacy_hand_action_to_inspire(
        legacy_action: np.ndarray,
        hand_task: str,
    ) -> np.ndarray:
        """Match the production 7-DOF-to-Inspire command conversion."""
        from decoupled_wbc.control.envs.g1.utils.command_sender import InspireHandCommandSender

        return InspireHandCommandSender.legacy_dex3_to_inspire(
            np.asarray(legacy_action, dtype=np.float64),
            hand_task=hand_task,
        )

    def _get_eef_state(self, proprio: dict, q: np.ndarray) -> np.ndarray:
        wrist_pose = proprio.get("wrist_pose")
        if wrist_pose is not None:
            wrist_pose = _as_1d_array(wrist_pose, dtype=np.float64, length=14)
            if wrist_pose.size == 14 and np.all(np.isfinite(wrist_pose)):
                return wrist_pose

        self.robot_model.cache_forward_kinematics(q)
        eef_parts = []
        for side in ["left", "right"]:
            placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            pos = placement.translation[:3]
            quat = R.from_matrix(placement.rotation).as_quat(scalar_first=True)
            eef_parts.append(np.concatenate([pos, quat]))
        return np.ascontiguousarray(np.concatenate(eef_parts), dtype=np.float64)

    def _extract_base_quat(self, proprio: dict) -> np.ndarray:
        if "base_quat" in proprio:
            return _valid_quat(proprio["base_quat"], dtype=np.float64)
        if "floating_base_pose" in proprio:
            floating_base_pose = _as_1d_array(proprio["floating_base_pose"], dtype=np.float64)
            if floating_base_pose.size >= 7:
                return _valid_quat(floating_base_pose[3:7], dtype=np.float64)
        if "torso_quat" in proprio:
            return _valid_quat(proprio["torso_quat"], dtype=np.float64)
        return IDENTITY_QUAT_F64.copy()

    def _get_pose_age(self, msg: dict | None) -> float | None:
        if msg is None:
            return None
        receive_ts = msg.get("receive_timestamp")
        if receive_ts is None:
            return 0.0
        return max(0.0, time.time() - float(receive_ts))

    def _fresh_sonic_pose(self) -> tuple[dict | None, float | None, bool]:
        msg = self.latest_sonic_msg
        age = self._get_pose_age(msg)
        if msg is None or age is None:
            return None, None, False
        is_fresh = age <= self.config.sonic_pose_max_age
        if is_fresh:
            return msg, age, True
        return msg, age, False

    def _fresh_planner_msg(self) -> tuple[dict | None, float | None, bool]:
        msg = self.latest_planner_msg
        age = self._get_pose_age(msg)
        if msg is None or age is None:
            return None, None, False
        is_fresh = age <= self.config.planner_max_age
        if is_fresh:
            return msg, age, True
        return msg, age, False

    def _add_robot_state_features(self, frame_data: dict, proprio: dict, base_quat: np.ndarray):
        frame_data["observation.root_orientation"] = base_quat
        try:
            frame_data["observation.projected_gravity"] = compute_projected_gravity(
                base_quat
            ).astype(np.float64)
        except Exception:
            frame_data["observation.projected_gravity"] = DEFAULT_PROJECTED_GRAVITY.copy()

        if "init_ref_data_root_rot_array" in proprio:
            frame_data["observation.cpp_rotation_offset"] = _valid_quat(
                proprio["init_ref_data_root_rot_array"],
                dtype=np.float64,
            )
        elif "cpp_rotation_offset" in proprio:
            frame_data["observation.cpp_rotation_offset"] = _valid_quat(
                proprio["cpp_rotation_offset"],
                dtype=np.float64,
            )
        else:
            frame_data["observation.cpp_rotation_offset"] = IDENTITY_QUAT_F64.copy()

        if self._episode_init_base_quat is None:
            self._episode_init_base_quat = base_quat.copy()
        frame_data["observation.init_base_quat"] = self._episode_init_base_quat.copy()

        frame_data["teleop.delta_heading"] = np.array(
            [self._extract_delta_heading(proprio)],
            dtype=np.float64,
        )

        if "token_state" in proprio:
            frame_data["action.motion_token"] = _as_1d_array(
                proprio["token_state"],
                dtype=np.float64,
                length=64,
            )
        else:
            frame_data["action.motion_token"] = np.zeros(64, dtype=np.float64)

    def _extract_delta_heading(self, proprio: dict) -> float:
        smpl_msg, _, smpl_fresh = self._fresh_sonic_pose()
        if smpl_fresh and smpl_msg is not None and "heading_increment" in smpl_msg:
            return float(smpl_msg["heading_increment"])

        if "delta_heading" in proprio:
            arr = _as_1d_array(proprio["delta_heading"], dtype=np.float64, length=1)
            return float(arr[0])

        nav = proprio.get("navigate_command", proprio.get("navigate_cmd"))
        if nav is not None:
            arr = _as_1d_array(nav, dtype=np.float64)
            if arr.size >= 3:
                return float(arr[2])
        return 0.0

    def _add_teleop_features(self, frame_data: dict, q: np.ndarray) -> float | None:
        smpl_msg, smpl_age, smpl_fresh = self._fresh_sonic_pose()
        planner_msg, planner_age, planner_fresh = self._fresh_planner_msg()

        stream_mode = self.current_stream_mode
        if (
            stream_mode == 0
            and smpl_fresh
            and self.config.use_sonic_pose_when_stream_off
        ):
            stream_mode = self.config.default_stream_mode_when_pose_available
        frame_data["teleop.stream_mode"] = np.array([stream_mode], dtype=np.int32)

        use_smpl = (
            smpl_fresh
            and smpl_msg is not None
            and bool(smpl_msg.get("has_smpl", True))
            and (
                stream_mode in (1, 4)
                or self.config.use_sonic_pose_when_stream_off
            )
        )
        use_planner = planner_fresh and stream_mode == 5

        if use_smpl and smpl_msg is not None:
            joints = np.asarray(smpl_msg.get("smpl_joints", np.zeros((24, 3))), dtype=np.float32)
            frame_data["teleop.smpl_joints"] = _as_1d_array(
                joints,
                dtype=np.float32,
                length=72,
            )
            frame_data["teleop.smpl_pose"] = _as_1d_array(
                smpl_msg.get("smpl_pose"),
                dtype=np.float32,
                length=63,
            )
            body_quat_w = _valid_quat(smpl_msg.get("body_quat_w"), dtype=np.float32)
            frame_data["teleop.body_quat_w"] = body_quat_w
            frame_data["teleop.target_body_orientation"] = self._compute_target_body_orientation(
                body_quat_w,
                frame_data,
            )
            frame_data["teleop.smpl_frame_index"] = _as_1d_array(
                smpl_msg.get("frame_index"),
                dtype=np.int64,
                length=1,
            )
        else:
            frame_data["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)
            frame_data["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)
            frame_data["teleop.body_quat_w"] = IDENTITY_QUAT_F32.copy()
            frame_data["teleop.target_body_orientation"] = quat_to_rot6d(IDENTITY_QUAT_F32)
            frame_data["teleop.smpl_frame_index"] = np.array([0], dtype=np.int64)

        frame_data["teleop.left_wrist_joints"] = self._get_wrist_joints(
            q,
            smpl_msg,
            use_smpl,
            side="left",
        )
        frame_data["teleop.right_wrist_joints"] = self._get_wrist_joints(
            q,
            smpl_msg,
            use_smpl,
            side="right",
        )

        if self.config.with_hands:
            frame_data["teleop.left_hand_joints"] = q[29:35].astype(np.float32)
            frame_data["teleop.right_hand_joints"] = q[35:41].astype(np.float32)

        frame_data["teleop.planner_mode"] = np.array(
            [planner_msg["planner_mode"]] if use_planner and planner_msg else [0],
            dtype=np.int32,
        )
        frame_data["teleop.planner_movement"] = (
            planner_msg["planner_movement"].copy()
            if use_planner and planner_msg is not None
            else np.zeros(3, dtype=np.float32)
        )
        frame_data["teleop.planner_facing"] = (
            planner_msg["planner_facing"].copy()
            if use_planner and planner_msg is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        frame_data["teleop.planner_speed"] = np.array(
            [planner_msg["planner_speed"]] if use_planner and planner_msg is not None else [-1.0],
            dtype=np.float32,
        )
        frame_data["teleop.planner_height"] = np.array(
            [planner_msg["planner_height"]] if use_planner and planner_msg is not None else [-1.0],
            dtype=np.float32,
        )

        vr_position = None
        vr_orientation = None
        if use_planner and planner_msg is not None:
            vr_position = planner_msg.get("vr_3pt_position")
            vr_orientation = planner_msg.get("vr_3pt_orientation")
        elif smpl_fresh and smpl_msg is not None:
            vr_position = smpl_msg.get("vr_3pt_position")
            vr_orientation = smpl_msg.get("vr_3pt_orientation")

        frame_data["teleop.vr_3pt_position"] = (
            _as_1d_array(vr_position, dtype=np.float32, length=9)
            if vr_position is not None
            else np.zeros(9, dtype=np.float32)
        )
        frame_data["teleop.vr_3pt_orientation"] = _vr_orientation_to_rot6d(vr_orientation)

        if smpl_fresh and smpl_age is not None:
            self.sonic_timing_monitor.log_time_delta(smpl_age)
            return smpl_age * 1000.0
        if use_planner and planner_age is not None:
            return planner_age * 1000.0
        return None

    def _get_wrist_joints(
        self,
        q: np.ndarray,
        smpl_msg: dict | None,
        use_smpl: bool,
        *,
        side: str,
    ) -> np.ndarray:
        key = f"{side}_wrist_joints"
        if use_smpl and smpl_msg is not None and smpl_msg.get(key) is not None:
            return _as_1d_array(smpl_msg[key], dtype=np.float32, length=3)
        indices = self._left_wrist_indices if side == "left" else self._right_wrist_indices
        return np.ascontiguousarray(q[indices].astype(np.float32), dtype=np.float32)

    def _compute_target_body_orientation(
        self,
        body_quat_w: np.ndarray,
        frame_data: dict,
    ) -> np.ndarray:
        delta_heading = float(frame_data.get("teleop.delta_heading", [0.0])[0])
        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot
        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = (
            R.from_euler("ZYX", normalised_euler, degrees=False)
            .as_quat(scalar_first=True)
            .astype(np.float32)
        )
        return np.ascontiguousarray(quat_to_rot6d(target_quat), dtype=np.float32)

    def _add_images_to_frame_data(self, frame_data: dict) -> None:
        if self.latest_image_msg is None:
            return
        images = self.latest_image_msg["images"]
        for feature_name, feature_info in self.data_exporter.features.items():
            if feature_info.get("dtype") in ["image", "video"]:
                image_key = feature_name.split(".")[-1]
                if image_key not in images:
                    raise ValueError(
                        f"Required image {image_key!r} for feature {feature_name!r} "
                        f"not found. Available images: {list(images.keys())}"
                    )
                frame_data[feature_name] = images[image_key]

    def _finalize_frame(self, t_start: float) -> bool:
        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
                self.sonic_timing_monitor.reset()
                self._initial_yaw = None
                self._episode_init_base_quat = None
            self._episode_state.change_state()
        return True

    def _add_data_frame(self) -> bool:
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            return False

        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return self._finalize_frame(t_start)

        if self.config.require_sonic_pose:
            _, _, smpl_fresh = self._fresh_sonic_pose()
            if not smpl_fresh:
                return False

        proprio = self.latest_proprio_msg
        kinematic_state = self._get_kinematic_state(proprio)
        observation_state = self._get_observation_state(proprio, kinematic_state)
        action_wbc = self._get_action_wbc(proprio, observation_state)
        observation_eef_state = self._get_eef_state(proprio, kinematic_state)
        base_quat = self._extract_base_quat(proprio)

        frame_data: dict = {
            "observation.state": observation_state,
            "observation.eef_state": observation_eef_state,
            "action.wbc": action_wbc,
        }

        self._add_robot_state_features(frame_data, proprio, base_quat)
        self._add_teleop_features(frame_data, observation_state)
        self._add_images_to_frame_data(frame_data)

        self.data_exporter.add_frame(frame_data)
        return self._finalize_frame(t_start)

    def save_and_cleanup(self) -> None:
        try:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
        except Exception as exc:
            self._print_and_say(f"Error saving episode: {exc}", blocking=True)

        try:
            self._sonic_zmq_socket.close()
            self._sonic_zmq_ctx.term()
        except Exception:
            pass

    def run(self) -> None:
        try:
            while rclpy.ok():
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    with self.telemetry.timer("poll_state"):
                        self._poll_state_ros()

                    with self.telemetry.timer("poll_pico"):
                        self._poll_sonic_zmq_messages()

                    with self.telemetry.timer("poll_image"):
                        img_msg = self._image_subscriber.read()
                        if img_msg is not None:
                            self.latest_image_msg = img_msg

                    with self.telemetry.timer("check_recording_commands"):
                        self._check_recording_commands()

                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                sleep_time = self.loop_period - (time.monotonic() - t_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()
        finally:
            self.save_and_cleanup()


def main(config: DecoupledVLADataExporterConfig) -> None:
    if config.dataset_name is None:
        config.dataset_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    rclpy.init(args=None)
    robot_config = poll_robot_config_ros(
        config.robot_config_service_name,
        config.robot_config_timeout,
    )

    node = rclpy.create_node("decoupled_vla_data_exporter")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    time.sleep(0.2)

    try:
        robot_model = get_g1_robot_model()
        if config.with_hands:
            dataset_features = get_features_sonic_inspire6(robot_model)
            modality_config = get_modality_config_sonic_inspire6(robot_model)
            schema_compatibility = "g1_inspire_41dof"
        else:
            dataset_features = get_features_sonic_body29(robot_model)
            modality_config = get_modality_config_sonic_body29(robot_model)
            schema_compatibility = "g1_body_29dof"

        if config.record_wrist_cameras:
            dataset_features.update(get_wrist_camera_features())
            wrist_modality = get_wrist_camera_modality_config()
            for key, value in wrist_modality.items():
                modality_config.setdefault(key, {}).update(value)

        text_to_speech = TextToSpeech() if config.text_to_speech else None
        audio_cue = AudioCue(volume=config.audio_cue_volume) if config.audio_cues else None
        script_config = {
            **robot_config,
            "decoupled_vla_exporter": asdict(config),
            "data_source": {
                "robot_state": "decoupled_ros",
                "pico_pose": "optional_sonic_zmq",
                "camera": "gear_sonic_composed_camera",
            },
            "schema_compatibility": schema_compatibility,
        }

        data_exporter = Gr00tDataExporter.create(
            save_root=f"{config.root_output_dir}/{config.dataset_name}",
            fps=config.data_collection_frequency,
            features=dataset_features,
            modality_config=modality_config,
            task=config.task_prompt,
            script_config=script_config,
            overwrite_existing=config.overwrite_existing_dataset,
        )

        collector = DecoupledVLADataCollector(
            node=node,
            config=config,
            data_exporter=data_exporter,
            robot_model=robot_model,
            text_to_speech=text_to_speech,
            audio_cue=audio_cue,
        )
        collector.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main(tyro.cli(DecoupledVLADataExporterConfig))
