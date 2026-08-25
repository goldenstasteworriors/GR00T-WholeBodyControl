"""Local interactive viewer for pure-FK onboard camera tuning.

Run from the project root with::

    MUJOCO_GL=egl PYTHONPATH=. .venv_sim/bin/python -m gear_sonic.tools.fk_camera_tuner.app

The server is intentionally local-only.  It applies qpos and ``mj_forward`` to
the visual-mesh model; it never calls ``mj_step`` or controls a robot.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.render_onboard_fk_mask import (
    CAMERA_NAME,
    DEFAULT_MODEL_PATH,
    add_ego_camera,
    arm_geom_ids,
    build_joint_mapping,
    build_mimic_mapping,
    hand_geom_ids,
    load_episode,
    overlay_mask,
    render_masks,
    render_mesh_rgb,
    resolve_path,
    set_pose_and_forward,
    side_arm_geom_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ONBOARD_ROOT = REPO_ROOT / "outputs/onboard"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
PRESET_ROOT = REPO_ROOT / "outputs/onboard_fk_mask_experiment/tuner_presets"
DEFAULT_CAMERA = {
    # Global left-arm-only silhouette fit with the measured intrinsics fixed;
    # see outputs/onboard_fk_mask_experiment/fixed_intrinsics_offset0_left_arm_global_search.
    "camera_body": "torso_link",
    "px": 0.1010968824,
    "py": 0.0292756905,
    "pz": 0.4801855567,
    "rx": 0.6636426687,
    "ry": -0.6435363327,
    "rz": -1.1650041528,
    "fx": 607.061,
    "fy": 607.014,
    "cx": 331.952,
    "cy": 246.989,
    "state_offset": 0,
    "mask_kind": "left_arm",
    "view_mode": "raw",
    "orbit_azimuth": -2.2,
    "orbit_elevation": 0.4,
    "orbit_distance": 0.55,
    "orbit_roll": 0.0,
}
COLOR_BY_MASK = {
    "robot": (0, 0, 255),
    "arms": (0, 220, 0),
    "arm_body": (0, 180, 255),
    "hands": (0, 255, 0),
    "left_arm": (0, 255, 0),
}
VALID_MASKS = frozenset(COLOR_BY_MASK)


def as_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, 92))
    if not ok:
        raise RuntimeError("Could not encode render image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def parse_float(query: dict[str, list[str]], name: str, default: float) -> float:
    try:
        value = float(query.get(name, [str(default)])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def parse_int(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(query.get(name, [str(default)])[0])
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def safe_dataset_path(relative: str) -> Path:
    candidate = (ONBOARD_ROOT / relative).resolve()
    if ONBOARD_ROOT.resolve() not in candidate.parents or not (candidate / "meta/info.json").is_file():
        raise ValueError("Unknown onboard dataset")
    return candidate


def discover_datasets() -> list[str]:
    results = []
    for metadata in ONBOARD_ROOT.rglob("meta/info.json"):
        dataset = metadata.parent.parent
        results.append(str(dataset.relative_to(ONBOARD_ROOT)))
    return sorted(set(results))


def discover_episodes(dataset_path: Path) -> list[int]:
    pattern = re.compile(r"episode_(\d+)\.parquet$")
    episodes = []
    for path in dataset_path.glob("data/chunk-*/episode_*.parquet"):
        match = pattern.search(path.name)
        if match:
            episodes.append(int(match.group(1)))
    return sorted(set(episodes))


def euler_to_quaternion(euler: tuple[float, float, float]) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("XYZ", euler).as_quat()
    return np.asarray((w, x, y, z), dtype=np.float64)


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return Rotation.from_quat((x, y, z, w)).as_euler("XYZ")


def look_at_quaternion(camera_pos: np.ndarray, target_pos: np.ndarray, roll: float) -> np.ndarray:
    """Return a MuJoCo camera quaternion looking from camera_pos to target_pos."""
    direction = target_pos - camera_pos
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        raise ValueError("Orbit distance is too small")
    direction /= distance
    camera_z = -direction
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_x = np.cross(direction, up)
    if float(np.linalg.norm(camera_x)) < 1e-5:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        camera_x = np.cross(direction, up)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    rolled_x = math.cos(roll) * camera_x + math.sin(roll) * camera_y
    rolled_y = -math.sin(roll) * camera_x + math.cos(roll) * camera_y
    camera_to_body = np.column_stack((rolled_x, rolled_y, camera_z))
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, camera_to_body.reshape(-1))
    return quaternion


def orbit_camera(
    target_pos: np.ndarray,
    azimuth: float,
    elevation: float,
    distance: float,
    roll: float,
) -> tuple[np.ndarray, np.ndarray]:
    if distance <= 0.0:
        raise ValueError("Orbit distance must be positive")
    horizontal = distance * math.cos(elevation)
    offset = np.array(
        (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            distance * math.sin(elevation),
        ),
        dtype=np.float64,
    )
    camera_pos = target_pos + offset
    return camera_pos, look_at_quaternion(camera_pos, target_pos, roll)


@dataclass
class RenderContext:
    dataset_path: Path
    episode_index: int
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    camera_id: int
    episode: Any
    mapping: Any
    mimic_mapping: Any
    arm_ids: set[int]
    hand_ids: set[int]
    left_arm_ids: set[int]
    left_hand_ids: set[int]
    scene_option: mujoco.MjvOption
    frame_count: int
    width: int
    height: int

    @classmethod
    def create(cls, dataset_path: Path, episode_index: int) -> "RenderContext":
        episode = load_episode(dataset_path, episode_index)
        capture = cv2.VideoCapture(str(episode.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {episode.video_path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if (width, height) != (640, 480):
            raise ValueError(f"Viewer currently expects 640x480 video, got {width}x{height}")
        model = add_ego_camera(
            resolve_path(DEFAULT_MODEL_PATH),
            DEFAULT_CAMERA["camera_body"],
            (DEFAULT_CAMERA["px"], DEFAULT_CAMERA["py"], DEFAULT_CAMERA["pz"]),
            (DEFAULT_CAMERA["rx"], DEFAULT_CAMERA["ry"], DEFAULT_CAMERA["rz"]),
            DEFAULT_CAMERA["fx"],
            DEFAULT_CAMERA["fy"],
            DEFAULT_CAMERA["cx"],
            DEFAULT_CAMERA["cy"],
            width,
            height,
        )
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if camera_id < 0:
            raise RuntimeError("Injected camera was not found in the compiled model")
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[1] = 1
        arm_ids = arm_geom_ids(model)
        hand_ids = hand_geom_ids(model, arm_ids)
        left_arm_ids = side_arm_geom_ids(model, "left")
        return cls(
            dataset_path=dataset_path,
            episode_index=episode_index,
            model=model,
            data=mujoco.MjData(model),
            renderer=mujoco.Renderer(model, height=height, width=width),
            camera_id=camera_id,
            episode=episode,
            mapping=build_joint_mapping(model, episode.joint_names),
            mimic_mapping=build_mimic_mapping(model),
            arm_ids=arm_ids,
            hand_ids=hand_ids,
            left_arm_ids=left_arm_ids,
            left_hand_ids=hand_ids & left_arm_ids,
            scene_option=scene_option,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    def close(self) -> None:
        self.renderer.close()

    def left_hand_target_in_torso(self) -> np.ndarray:
        """Return the visual-hand center in torso-local coordinates."""
        if not self.left_hand_ids:
            raise RuntimeError("No left-hand visual geometry found")
        target_world = np.mean(self.data.geom_xpos[list(self.left_hand_ids)], axis=0)
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DEFAULT_CAMERA["camera_body"])
        torso_to_world = self.data.xmat[torso_id].reshape(3, 3)
        return torso_to_world.T @ (target_world - self.data.xpos[torso_id])

    def render(self, parameters: dict[str, Any]) -> dict[str, Any]:
        frame = int(parameters["frame"])
        state_index = frame + int(parameters["state_offset"])
        if not 0 <= frame < self.frame_count:
            raise ValueError(f"frame must lie in [0, {self.frame_count - 1}]")
        if not 0 <= state_index < len(self.episode.table):
            raise ValueError("frame + state offset lies outside the recorded state sequence")
        mask_kind = str(parameters["mask_kind"])
        if mask_kind not in VALID_MASKS:
            raise ValueError(f"Unknown mask kind: {mask_kind}")
        self.model.cam_intrinsic[self.camera_id] = (
            parameters["fx"],
            parameters["fy"],
            parameters["cx"],
            parameters["cy"],
        )
        self.model.cam_fovy[self.camera_id] = math.degrees(2.0 * math.atan(self.height / (2.0 * parameters["fy"])))
        state = np.asarray(self.episode.table.iloc[state_index]["observation.state"], dtype=np.float64).reshape(-1)
        set_pose_and_forward(self.model, self.data, state, self.mapping, self.mimic_mapping)
        hand_target = self.left_hand_target_in_torso()
        if parameters["view_mode"] == "orbit":
            camera_pos, camera_quaternion = orbit_camera(
                hand_target,
                parameters["orbit_azimuth"],
                parameters["orbit_elevation"],
                parameters["orbit_distance"],
                parameters["orbit_roll"],
            )
        else:
            camera_pos = np.asarray((parameters["px"], parameters["py"], parameters["pz"]), dtype=np.float64)
            camera_quaternion = euler_to_quaternion(
                (parameters["rx"], parameters["ry"], parameters["rz"])
            )
        self.model.cam_pos[self.camera_id] = camera_pos
        self.model.cam_quat[self.camera_id] = camera_quaternion
        # Camera parameters changed after the first FK pass used to locate the
        # hand target, so refresh derived camera transforms without stepping.
        mujoco.mj_forward(self.model, self.data)
        robot_mask, arms_mask, arm_body_mask, hands_mask, left_arm_mask = render_masks(
            self.renderer,
            self.data,
            self.scene_option,
            self.arm_ids,
            self.hand_ids,
            self.left_arm_ids,
        )
        mesh_rgb = render_mesh_rgb(self.renderer, self.data, self.scene_option)
        capture = cv2.VideoCapture(str(self.episode.video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
        finally:
            capture.release()
        if not ok:
            raise RuntimeError(f"Could not decode video frame {frame}")
        masks = {
            "robot": robot_mask,
            "arms": arms_mask,
            "arm_body": arm_body_mask,
            "hands": hands_mask,
            "left_arm": left_arm_mask,
        }
        overlay = overlay_mask(bgr, masks[mask_kind], COLOR_BY_MASK[mask_kind])
        resolved_euler = quaternion_to_euler(camera_quaternion)
        orbit_offset = camera_pos - hand_target
        orbit_distance = float(np.linalg.norm(orbit_offset))
        orbit_equivalent = {
            "azimuth": math.atan2(float(orbit_offset[1]), float(orbit_offset[0])),
            "elevation": math.atan2(
                float(orbit_offset[2]),
                math.hypot(float(orbit_offset[0]), float(orbit_offset[1])),
            ),
            "distance": orbit_distance,
        }
        return {
            "frame": frame,
            "state_index": state_index,
            "mask_kind": mask_kind,
            "raw": as_data_url(bgr),
            "overlay": as_data_url(overlay),
            "mesh": as_data_url(cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)),
            "mask_pixels": int(masks[mask_kind].sum()),
            "frame_count": self.frame_count,
            "view_mode": parameters["view_mode"],
            "hand_target_torso": hand_target.tolist(),
            "orbit_equivalent": orbit_equivalent,
            "resolved_camera": {
                "position": camera_pos.tolist(),
                "quaternion_wxyz": camera_quaternion.tolist(),
                "euler_intrinsic_xyz_rad": resolved_euler.tolist(),
            },
        }


class TunerService:
    def __init__(self) -> None:
        self._context: RenderContext | None = None
        self._key: tuple[Path, int] | None = None
        self._lock = threading.Lock()

    def render(self, dataset_path: Path, episode_index: int, parameters: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            key = (dataset_path, episode_index)
            if self._key != key:
                if self._context is not None:
                    self._context.close()
                self._context = RenderContext.create(dataset_path, episode_index)
                self._key = key
            assert self._context is not None
            return self._context.render(parameters)


SERVICE = TunerService()


class TunerHandler(SimpleHTTPRequestHandler):
    server_version = "FKCameraTuner/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[fk-camera-tuner] {format % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 16384:
            raise ValueError("Invalid request body length")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON request must be an object")
        return data

    def request_parameters(self, query: dict[str, list[str]]) -> tuple[Path, int, dict[str, Any]]:
        dataset_value = query.get("dataset", [""])[0]
        dataset_path = safe_dataset_path(dataset_value)
        episode_index = parse_int(query, "episode", 0)
        parameters: dict[str, Any] = {
            "frame": parse_int(query, "frame", 0),
            # Keep video and recorded joint state on the same index: this tuner
            # is for camera-pose fitting, not temporal-alignment compensation.
            "state_offset": 0,
            "mask_kind": query.get("mask_kind", [str(DEFAULT_CAMERA["mask_kind"])])[0],
            "view_mode": query.get("view_mode", [str(DEFAULT_CAMERA["view_mode"])])[0],
        }
        if parameters["view_mode"] not in {"raw", "orbit"}:
            raise ValueError("view_mode must be raw or orbit")
        for key in (
            "px", "py", "pz", "rx", "ry", "rz", "fx", "fy", "cx", "cy",
            "orbit_azimuth", "orbit_elevation", "orbit_distance", "orbit_roll",
        ):
            parameters[key] = parse_float(query, key, float(DEFAULT_CAMERA[key]))
        if parameters["fx"] <= 0 or parameters["fy"] <= 0:
            raise ValueError("fx and fy must be positive")
        if parameters["orbit_distance"] <= 0:
            raise ValueError("orbit_distance must be positive")
        return dataset_path, episode_index, parameters

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/datasets":
                self.send_json({"datasets": discover_datasets(), "defaults": DEFAULT_CAMERA})
                return
            if parsed.path == "/api/episodes":
                dataset_path = safe_dataset_path(query.get("dataset", [""])[0])
                episodes = discover_episodes(dataset_path)
                self.send_json({"episodes": episodes})
                return
            if parsed.path == "/api/info":
                dataset_path = safe_dataset_path(query.get("dataset", [""])[0])
                episode_index = parse_int(query, "episode", 0)
                episode = load_episode(dataset_path, episode_index)
                capture = cv2.VideoCapture(str(episode.video_path))
                try:
                    info = {
                        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    }
                finally:
                    capture.release()
                self.send_json(info)
                return
            if parsed.path == "/api/render":
                dataset_path, episode_index, parameters = self.request_parameters(query)
                started = time.perf_counter()
                payload = SERVICE.render(dataset_path, episode_index, parameters)
                payload["render_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
                self.send_json(payload)
                return
            return super().do_GET()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/api/presets":
                self.send_json({"error": "Unknown API path"}, HTTPStatus.NOT_FOUND)
                return
            payload = self.read_json()
            name = str(payload.get("name", "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
                raise ValueError("Preset name must contain only letters, digits, _ or -")
            dataset_path = safe_dataset_path(str(payload.get("dataset", "")))
            saved = {
                "dataset": str(dataset_path.relative_to(ONBOARD_ROOT)),
                "episode": int(payload.get("episode", 0)),
                "parameters": payload.get("parameters", {}),
                "created_at_unix": time.time(),
                "method": "interactive qpos -> mj_forward -> visual-mesh rasterization",
                "calls_mj_step": False,
            }
            PRESET_ROOT.mkdir(parents=True, exist_ok=True)
            destination = PRESET_ROOT / f"{name}.json"
            with destination.open("w", encoding="utf-8") as file:
                json.dump(saved, file, ensure_ascii=False, indent=2)
            self.send_json({"saved": str(destination.relative_to(REPO_ROOT))}, HTTPStatus.CREATED)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local FK camera tuner.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must lie in [1, 65535]")
    server = ThreadingHTTPServer((args.host, args.port), TunerHandler)
    print(f"FK camera tuner: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping FK camera tuner")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
