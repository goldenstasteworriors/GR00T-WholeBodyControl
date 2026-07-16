from collections import defaultdict, deque
import time
from types import SimpleNamespace

import numpy as np

from gear_sonic.scripts.pico_manager_thread_server import PoseStreamer, _raw_xr_3pt_pose
from gear_sonic.scripts.run_decoupled_vla_data_exporter import (
    DecoupledVLADataCollector,
    _body_state_joint_indices,
    unpack_pose_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


class _TimingMonitor:
    def log_time_delta(self, _value):
        pass


class _Socket:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class _ThreePoint:
    def process_raw_vr_pose(self, pose):
        return pose


class _YawAccumulator:
    def update(self, _axis, _dt):
        pass

    def yaw_angle_change(self):
        return 0.0


class _RobotModel:
    _dof_indices = {
        "left_wrist_roll_joint": 19,
        "left_wrist_pitch_joint": 20,
        "left_wrist_yaw_joint": 21,
        "right_wrist_roll_joint": 33,
        "right_wrist_pitch_joint": 34,
        "right_wrist_yaw_joint": 35,
    }

    def get_body_actuated_joint_indices(self):
        return list(range(22)) + list(range(29, 36))

    def dof_index(self, name):
        return self._dof_indices[name]


def test_raw_xr_3pt_pose_is_head_relative_and_z_up():
    identity = [0.0, 0.0, 0.0, 1.0]
    left = np.array([1.0, 0.0, 0.0, *identity])
    right = np.array([0.0, 0.0, 1.0, *identity])
    head = np.array([0.0, 0.0, 0.0, *identity])

    pose = _raw_xr_3pt_pose(left, right, head)

    assert pose is not None
    np.testing.assert_allclose(pose[:, :3], [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(pose[:, 3:], [[1.0, 0.0, 0.0, 0.0]] * 3)


def test_right_wrist_indices_are_mapped_into_29dof_body_state():
    indices = _body_state_joint_indices(
        _RobotModel(),
        [
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )

    assert indices == [26, 27, 28]


def test_vr3pt_only_stream_message_does_not_synthesize_smpl(monkeypatch):
    monkeypatch.setattr(
        "gear_sonic.scripts.pico_manager_thread_server.get_controller_axes",
        lambda _reader: (0.0, 0.0, 0.0, 0.0),
    )
    streamer = PoseStreamer.__new__(PoseStreamer)
    streamer.socket = _Socket()
    streamer.reader = object()
    streamer.three_point = _ThreePoint()
    streamer.num_frames_to_send = 1
    streamer.target_fps = 50
    streamer.record_dir = ""
    streamer.frame_buffer = defaultdict(lambda: deque(maxlen=1))
    streamer.buffer_cleared = True
    streamer.prev_stamp_ns = None
    streamer.next_target_ns = None
    streamer.step = 0
    streamer.frame_time = 0.02
    streamer.yaw_accumulator = _YawAccumulator()
    streamer.fps_counter = 0
    streamer.last_fps_report = time.time()
    streamer.log_prefix = "test"

    vr_pose = np.array(
        [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]] * 3,
        dtype=np.float32,
    )
    streamer._run_vr3pt_only(
        sample={
            "vr_3pt_pose_raw": vr_pose,
            "timestamp_ns": 1_000_000_000,
            "timestamp_realtime": 1.0,
            "timestamp_monotonic": 1.0,
            "dt": 0.02,
            "fps": 50.0,
        },
        left_trigger=0.1,
        right_trigger=0.2,
        left_grip=0.3,
        right_grip=0.4,
        left_hand_joints=np.zeros((1, 7), dtype=np.float32),
        right_hand_joints=np.zeros((1, 7), dtype=np.float32),
        toggle_data_collection=False,
        toggle_data_abort=False,
    )

    assert len(streamer.socket.messages) == 1
    message = unpack_pose_message(streamer.socket.messages[0], "pose")
    assert "smpl_joints" not in message
    assert "smpl_pose" not in message
    np.testing.assert_allclose(message["vr_position"], vr_pose[:, :3].reshape(-1))


def test_exporter_keeps_vr3pt_when_pose_message_has_no_smpl():
    collector = DecoupledVLADataCollector.__new__(DecoupledVLADataCollector)
    collector._manager_toggle_dc = False
    collector._manager_toggle_da = False
    collector.latest_sonic_msg = None
    collector.latest_planner_msg = None
    collector.current_stream_mode = 0
    collector.config = SimpleNamespace(
        sonic_pose_max_age=1.0,
        planner_max_age=1.0,
        use_sonic_pose_when_stream_off=True,
        default_stream_mode_when_pose_available=1,
    )
    collector._left_wrist_indices = [22, 24, 26]
    collector._right_wrist_indices = [23, 25, 27]
    collector.sonic_timing_monitor = _TimingMonitor()

    vr_position = np.arange(9, dtype=np.float32) / 10.0
    vr_orientation = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        3,
    )
    raw = pack_pose_message(
        {
            "vr_position": vr_position,
            "vr_orientation": vr_orientation,
            "frame_index": np.array([7], dtype=np.int64),
        },
        topic="pose",
    )

    collector._handle_pose_message(raw)
    assert collector.latest_sonic_msg is not None
    assert collector.latest_sonic_msg["has_smpl"] is False

    frame_data = {}
    collector._add_teleop_features(frame_data, np.arange(41, dtype=np.float32))

    np.testing.assert_allclose(frame_data["teleop.vr_3pt_position"], vr_position)
    assert np.any(frame_data["teleop.vr_3pt_orientation"])
    np.testing.assert_array_equal(frame_data["teleop.smpl_joints"], np.zeros(72))
    np.testing.assert_array_equal(frame_data["teleop.smpl_pose"], np.zeros(63))
    np.testing.assert_array_equal(frame_data["teleop.smpl_frame_index"], [0])
    np.testing.assert_array_equal(frame_data["teleop.stream_mode"], [1])
