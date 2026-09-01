"""Render ego-view robot masks from recorded joint positions without physics.

The script loads one LeRobot episode, writes the recorded scalar joint positions
directly into MuJoCo ``qpos``, runs ``mj_forward`` (forward kinematics only), and
rasterizes the visual meshes from a camera rigidly attached to ``torso_link``.
It never calls ``mj_step`` and does not instantiate a task environment.

The output contains both the complete visible robot silhouette and an
arms-and-hands-only silhouette.  Visibility between robot links is resolved by
the rasterizer's z-buffer.  Occlusion by real scene objects is intentionally not
inferred from RGB; it requires aligned observed depth or an external estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import mujoco
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/onboard_fk_mask_experiment"
CAMERA_NAME = "onboard_fk_ego_camera"

OFFICIAL_FTP_JOINT_ALIASES = {
    f"{side}_hand_{source}": f"{side}_{target}_joint"
    for side in ("left", "right")
    for source, target in (
        ("little", "little_1"),
        ("ring", "ring_1"),
        ("middle", "middle_1"),
        ("index", "index_1"),
        ("thumb_bend", "thumb_2"),
        ("thumb_rotate", "thumb_1"),
    )
}

OFFICIAL_FTP_PRIMARY_JOINT_RANGES = {
    f"{side}_{joint}_joint": upper
    for side in ("left", "right")
    for joint, upper in (
        ("little_1", 1.4381),
        ("ring_1", 1.4381),
        ("middle_1", 1.4381),
        ("index_1", 1.4381),
        ("thumb_2", 0.5864),
        ("thumb_1", 1.1641),
    )
}

OFFICIAL_FTP_MIMIC_RELATIONS = tuple(
    relation
    for side in ("left", "right")
    for relation in (
        (f"{side}_thumb_2_joint", f"{side}_thumb_3_joint", 0.8024),
        (f"{side}_thumb_3_joint", f"{side}_thumb_4_joint", 0.9487),
        (f"{side}_index_1_joint", f"{side}_index_2_joint", 1.0843),
        (f"{side}_middle_1_joint", f"{side}_middle_2_joint", 1.0843),
        (f"{side}_ring_1_joint", f"{side}_ring_2_joint", 1.0843),
        (f"{side}_little_1_joint", f"{side}_little_2_joint", 1.0843),
    )
)

ARM_ROOT_BODIES = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")
DEBUG_BODY_ORIGINS = (
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class Episode:
    dataset_path: Path
    table: pd.DataFrame
    joint_names: tuple[str, ...]
    video_path: Path
    fps: float


@dataclass(frozen=True)
class JointMapping:
    source_index: int
    source_name: str
    target_name: str
    qpos_address: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure-FK ego-view mask rendering for onboard LeRobot data.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        default="0",
        help="Comma-separated video frame indices. Use 'all' with --stride for a sequence.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--state-frame-offset",
        type=int,
        default=0,
        help="State row = video frame + offset; useful for synchronization sweeps.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--left-hand-mount-yaml",
        type=Path,
        help="Optional fitted left wrist-local Inspire-hand translation; recorded finger state is preserved.",
    )
    parser.add_argument("--camera-body", default="torso_link")
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=(0.06, 0.0, 0.45),
        metavar=("X", "Y", "Z"),
        help="Camera position in camera-body coordinates (metres).",
    )
    parser.add_argument(
        "--camera-euler",
        type=float,
        nargs=3,
        default=(0.0, -0.8, -1.57),
        metavar=("RX", "RY", "RZ"),
        help="MuJoCo intrinsic xyz Euler angles in radians.",
    )
    parser.add_argument("--fx", type=float, default=450.0)
    parser.add_argument("--fy", type=float, default=450.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument(
        "--mujoco-principal-pixel-offset",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help=(
            "Explicit MuJoCo principal-pixel offset. Use this only to reproduce legacy camera fits; "
            "otherwise OpenCV cx/cy are converted to center-relative offsets."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--visual-geom-group",
        type=int,
        default=1,
        help="MuJoCo geom group containing visual meshes in the official model.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-mesh-rgb",
        action="store_true",
        help="Skip RGB mesh render; segmentation masks and overlays are still generated.",
    )
    parser.add_argument(
        "--annotate-body-origins",
        action="store_true",
        help="Draw projected elbow/wrist body origins for camera-calibration diagnostics.",
    )
    parser.add_argument(
        "--color-arm-links",
        action="store_true",
        help="Color arm links by kinematic segment in the diagnostic mesh render.",
    )
    parser.add_argument(
        "--panel-mask-kind",
        choices=("arms", "arm_body", "hands"),
        default="arms",
        help="Mask shown in the lower-right audit panel; all mask files are always saved.",
    )
    parser.add_argument(
        "--mask-side",
        choices=("all", "left", "right"),
        default="all",
        help="Restrict arm/hand masks to one kinematic side; the full-robot mask is unchanged.",
    )
    parser.add_argument(
        "--contact-columns",
        type=int,
        default=1,
        help="Number of columns in the generated contact sheet.",
    )
    parser.add_argument(
        "--write-panel-video",
        action="store_true",
        help="Encode the selected four-panel audit frames as an MP4.",
    )
    parser.add_argument(
        "--panel-video-fps",
        type=float,
        default=None,
        help="Audit-video frame rate; defaults to source FPS divided by --stride.",
    )
    return parser


def resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = REPO_ROOT / expanded
    return expanded.resolve()


def load_episode(dataset_path: Path, episode_index: int) -> Episode:
    dataset_path = resolve_path(dataset_path)
    info_path = dataset_path / "meta/info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    chunk_size = int(info.get("chunks_size", 1000))
    fields = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
    }
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    table_path = dataset_path / data_pattern.format(**fields)
    video_path = dataset_path / video_pattern.format(**fields, video_key="observation.images.ego_view")
    if not table_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {table_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Ego-view video not found: {video_path}")

    table = pd.read_parquet(table_path)
    feature = info.get("features", {}).get("observation.state", {})
    joint_names = feature.get("names", [])
    if not isinstance(joint_names, list) or not joint_names:
        raise ValueError(f"No observation.state names in {info_path}")
    if "observation.state" not in table:
        raise ValueError(f"No observation.state column in {table_path}")

    return Episode(
        dataset_path=dataset_path,
        table=table,
        joint_names=tuple(joint_names),
        video_path=video_path,
        fps=float(info.get("fps", 50.0)),
    )


def parse_frame_indices(frames: str, frame_count: int, stride: int, max_frames: int | None) -> list[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if frames.strip().lower() == "all":
        indices = list(range(0, frame_count, stride))
    else:
        indices = [int(value.strip()) for value in frames.split(",") if value.strip()]
    if not indices:
        raise ValueError("No frame indices selected")
    invalid = [index for index in indices if index < 0 or index >= frame_count]
    if invalid:
        raise ValueError(f"Frame indices outside [0, {frame_count - 1}]: {invalid}")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        indices = indices[:max_frames]
    return indices


def add_ego_camera(
    model_path: Path,
    camera_body: str,
    camera_pos: Sequence[float],
    camera_euler: Sequence[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    principal_pixel_offset: Sequence[float] | None = None,
) -> mujoco.MjModel:
    if min(fx, fy, width, height) <= 0:
        raise ValueError("Camera focal lengths and image size must be positive")
    spec = mujoco.MjSpec.from_file(str(model_path))
    body = spec.body(camera_body)
    if body is None:
        raise ValueError(f"Camera attachment body not found: {camera_body}")
    mujoco_principal = (
        tuple(principal_pixel_offset)
        if principal_pixel_offset is not None
        else (cx - width / 2.0, cy - height / 2.0)
    )
    body.add_camera(
        name=CAMERA_NAME,
        pos=camera_pos,
        euler=camera_euler,
        sensor_size=(float(width), float(height)),
        resolution=(width, height),
        focal_pixel=(fx, fy),
        # MuJoCo's principalpixel is an offset from the image center, while
        # OpenCV/RealSense intrinsics store an absolute top-left pixel coordinate.
        principal_pixel=mujoco_principal,
    )
    return spec.compile()


def joint_candidates(source_name: str) -> tuple[str, ...]:
    normalized = source_name.removesuffix("_joint")
    candidates = (
        (source_name,)
        if source_name.endswith("_joint")
        else (
            source_name,
            f"{source_name}_joint",
        )
    )
    alias = OFFICIAL_FTP_JOINT_ALIASES.get(normalized)
    return (*candidates, alias) if alias else candidates


def build_joint_mapping(model: mujoco.MjModel, joint_names: Sequence[str]) -> list[JointMapping]:
    mapping: list[JointMapping] = []
    missing: list[str] = []
    for source_index, source_name in enumerate(joint_names):
        target_name = ""
        joint_id = -1
        for candidate in joint_candidates(source_name):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate)
            if joint_id >= 0:
                target_name = candidate
                break
        if joint_id < 0:
            missing.append(source_name)
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"Recorded scalar joint maps to non-scalar joint: {target_name}")
        mapping.append(
            JointMapping(
                source_index=source_index,
                source_name=source_name,
                target_name=target_name,
                qpos_address=int(model.jnt_qposadr[joint_id]),
            )
        )
    if missing:
        raise ValueError(f"Dataset joints missing from model: {missing}")
    return mapping


def convert_joint_value(item: JointMapping, value: float) -> float:
    official_upper = OFFICIAL_FTP_PRIMARY_JOINT_RANGES.get(item.target_name)
    if official_upper is not None:
        return official_upper * (1.0 - float(np.clip(value, 0.0, 1.0)))
    normalized_name = item.source_name.removesuffix("_joint")
    inspire_value = float(np.clip(value, 0.0, 1.0))
    if normalized_name.endswith(("_hand_little", "_hand_ring", "_hand_middle", "_hand_index")):
        return 1.7 * (1.0 - inspire_value)
    if normalized_name.endswith("_hand_thumb_bend"):
        return 0.5 * (1.0 - inspire_value)
    if normalized_name.endswith("_hand_thumb_rotate"):
        return 1.3 - 1.4 * inspire_value
    return float(value)


def build_mimic_mapping(model: mujoco.MjModel) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    for driver_name, follower_name, multiplier in OFFICIAL_FTP_MIMIC_RELATIONS:
        driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
        follower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, follower_name)
        if driver_id < 0 or follower_id < 0:
            continue
        result.append(
            (
                int(model.jnt_qposadr[driver_id]),
                int(model.jnt_qposadr[follower_id]),
                multiplier,
            )
        )
    return result


def set_pose_and_forward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
    mapping: Sequence[JointMapping],
    mimic_mapping: Sequence[tuple[int, int, float]],
) -> None:
    if state.shape != (len(mapping),):
        raise ValueError(f"State shape {state.shape} does not match {len(mapping)} joints")
    if not np.all(np.isfinite(state)):
        raise ValueError("State contains non-finite values")
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    for item in mapping:
        data.qpos[item.qpos_address] = convert_joint_value(item, float(state[item.source_index]))
    for driver_address, follower_address, multiplier in mimic_mapping:
        data.qpos[follower_address] = data.qpos[driver_address] * multiplier
    # FK and derived transforms only.  Do not replace this with mj_step.
    mujoco.mj_forward(model, data)


def body_descends_from(model: mujoco.MjModel, body_id: int, root_body_ids: set[int]) -> bool:
    current = body_id
    while current > 0:
        if current in root_body_ids:
            return True
        current = int(model.body_parentid[current])
    return False


def arm_geom_ids(model: mujoco.MjModel) -> set[int]:
    root_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ARM_ROOT_BODIES}
    if -1 in root_ids:
        raise ValueError(f"Model does not contain arm roots {ARM_ROOT_BODIES}")
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if body_descends_from(model, int(model.geom_bodyid[geom_id]), root_ids)
    }


def side_arm_geom_ids(model: mujoco.MjModel, side: str) -> set[int]:
    """Return every visual geometry in one arm, including wrist and hand meshes."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown arm side: {side}")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_link")
    if root < 0:
        raise ValueError(f"{side}_shoulder_pitch_link not found")
    return {
        geom_id
        for geom_id in arm_geom_ids(model)
        if body_descends_from(model, int(model.geom_bodyid[geom_id]), {root})
    }


def hand_geom_ids(model: mujoco.MjModel, all_arm_ids: set[int]) -> set[int]:
    hand_tokens = (
        "hand_",
        "thumb",
        "index",
        "middle",
        "ring",
        "little",
        "palm",
        "base_link",
    )
    result: set[int] = set()
    for geom_id in all_arm_ids:
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        mesh_name = ""
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        descriptor = f"{body_name} {mesh_name}"
        if any(token in descriptor for token in hand_tokens):
            result.add(geom_id)
    return result


def apply_fitted_left_hand_translation(
    model: mujoco.MjModel,
    left_hand_ids: set[int],
    parameters: dict[str, Any],
) -> np.ndarray:
    translation = np.asarray(
        parameters["left_wrist_yaw_link_T_inspire_hand_translation_m"], dtype=np.float64
    )
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("Left hand-mount translation must contain three finite values")

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    root_names = ("left_thumb_1", "left_index_1", "left_middle_1", "left_ring_1", "left_little_1")
    root_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in root_names],
        dtype=np.int32,
    )
    if wrist_id < 0 or np.any(root_ids < 0):
        raise ValueError("Official FTP left wrist or hand root bodies are missing")
    model.body_pos[root_ids] += translation

    direct_wrist_hand_ids = np.asarray(
        [geom_id for geom_id in left_hand_ids if int(model.geom_bodyid[geom_id]) == wrist_id],
        dtype=np.int32,
    )
    model.geom_pos[direct_wrist_hand_ids] += translation
    return translation


def color_arm_links(model: mujoco.MjModel) -> None:
    colors = {
        "shoulder": np.array([0.15, 0.45, 1.0, 1.0]),
        "elbow": np.array([1.0, 0.2, 0.15, 1.0]),
        "wrist_roll": np.array([1.0, 0.75, 0.1, 1.0]),
        "wrist_pitch": np.array([0.1, 0.9, 0.9, 1.0]),
        "wrist_yaw": np.array([0.95, 0.15, 0.9, 1.0]),
        "hand": np.array([0.25, 1.0, 0.25, 1.0]),
    }
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        lineage: list[str] = []
        while body_id > 0:
            lineage.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
            body_id = int(model.body_parentid[body_id])
        joined = " ".join(lineage)
        if not ("left_" in joined or "right_" in joined):
            continue
        if any(name.endswith("wrist_yaw_link") for name in lineage):
            segment = "wrist_yaw"
        elif any(name.endswith("wrist_pitch_link") for name in lineage):
            segment = "wrist_pitch"
        elif any(name.endswith("wrist_roll_link") for name in lineage):
            segment = "wrist_roll"
        elif any(name.endswith("elbow_link") for name in lineage):
            segment = "elbow"
        elif any(name.endswith("shoulder_pitch_link") for name in lineage):
            segment = "shoulder"
        else:
            continue
        direct_body = lineage[0]
        if any(token in direct_body for token in ("thumb", "index", "middle", "ring", "little", "palm", "base")):
            segment = "hand"
        model.geom_matid[geom_id] = -1
        if "right_" in joined:
            model.geom_rgba[geom_id] = np.array([0.15, 0.35, 1.0, 1.0])
        else:
            model.geom_rgba[geom_id] = colors[segment]


def render_masks(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption,
    arm_ids: set[int],
    hand_ids: set[int],
    extra_geom_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME, scene_option=scene_option)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()

    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise RuntimeError(f"Unexpected MuJoCo segmentation shape: {segmentation.shape}")
    object_ids = segmentation[..., 0]
    object_types = segmentation[..., 1]
    geom_visible = object_types == int(mujoco.mjtObj.mjOBJ_GEOM)
    robot_mask = geom_visible
    arms_mask = geom_visible & np.isin(object_ids, np.fromiter(arm_ids, dtype=np.int32))
    hands_mask = geom_visible & np.isin(object_ids, np.fromiter(hand_ids, dtype=np.int32))
    arm_body_mask = arms_mask & ~hands_mask
    extra_mask = (
        None
        if extra_geom_ids is None
        else geom_visible & np.isin(object_ids, np.fromiter(extra_geom_ids, dtype=np.int32))
    )
    return robot_mask, arms_mask, arm_body_mask, hands_mask, extra_mask


def render_mesh_rgb(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    renderer.update_scene(data, camera=CAMERA_NAME, scene_option=scene_option)
    return renderer.render().copy()


def read_video_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode video frame {frame_index}")
    return bgr


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = bgr.copy()
    tint = np.empty_like(result)
    tint[:] = color
    result[mask] = cv2.addWeighted(result, 0.45, tint, 0.55, 0.0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (16, 16, 16), -1)
    cv2.putText(
        panel,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def project_body_origins(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> list[tuple[str, tuple[int, int]]]:
    camera_position = data.cam_xpos[camera_id]
    camera_to_world = data.cam_xmat[camera_id].reshape(3, 3)
    world_to_camera = camera_to_world.T
    projected: list[tuple[str, tuple[int, int]]] = []
    for name in DEBUG_BODY_ORIGINS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        point_camera = world_to_camera @ (data.xpos[body_id] - camera_position)
        depth = -float(point_camera[2])
        if depth <= 1e-6:
            continue
        u = int(round(cx + fx * float(point_camera[0]) / depth))
        v = int(round(cy - fy * float(point_camera[1]) / depth))
        projected.append((name, (u, v)))
    return projected


def draw_body_origins(image: np.ndarray, projected: Sequence[tuple[str, tuple[int, int]]]) -> np.ndarray:
    result = image.copy()
    for name, point in projected:
        color = (0, 220, 255) if name.startswith("left_") else (255, 170, 0)
        cv2.circle(result, point, 5, color, -1, cv2.LINE_AA)
        short_name = name.replace("_link", "").replace("left_", "L:").replace("right_", "R:")
        cv2.putText(
            result,
            short_name,
            (point[0] + 7, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def save_frame_outputs(
    output_dir: Path,
    frame_index: int,
    state_index: int,
    bgr: np.ndarray,
    robot_mask: np.ndarray,
    arms_mask: np.ndarray,
    arm_body_mask: np.ndarray,
    hands_mask: np.ndarray,
    mesh_rgb: np.ndarray | None,
    panel_mask_kind: str = "arms",
    projected_origins: Sequence[tuple[str, tuple[int, int]]] = (),
) -> Path:
    stem = f"video_{frame_index:06d}_state_{state_index:06d}"
    mask_dir = output_dir / "masks"
    panel_dir = output_dir / "panels"
    mask_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(mask_dir / f"{stem}_robot.png"), robot_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_dir / f"{stem}_arms.png"), arms_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_dir / f"{stem}_arm_body.png"), arm_body_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_dir / f"{stem}_hands.png"), hands_mask.astype(np.uint8) * 255)

    robot_overlay = overlay_mask(bgr, robot_mask, (0, 0, 255))
    panel_masks = {
        "arms": (arms_mask, "arms + hands mask (green)"),
        "arm_body": (arm_body_mask, "arm body mask, hands excluded (green)"),
        "hands": (hands_mask, "hand mesh mask (green)"),
    }
    panel_mask, panel_label = panel_masks[panel_mask_kind]
    arms_overlay = overlay_mask(bgr, panel_mask, (0, 255, 0))
    if mesh_rgb is None:
        mesh_bgr = np.zeros_like(bgr)
    else:
        mesh_bgr = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR)
    top = np.concatenate((label_panel(bgr, "real RGB"), label_panel(mesh_bgr, "FK mesh render")), axis=1)
    bottom = np.concatenate(
        (
            label_panel(robot_overlay, "full robot mask (red)"),
            label_panel(arms_overlay, panel_label),
        ),
        axis=1,
    )
    panel = np.concatenate((top, bottom), axis=0)
    panel_path = panel_dir / f"{stem}.jpg"
    cv2.imwrite(str(panel_path), panel)
    if projected_origins:
        debug_real = draw_body_origins(bgr, projected_origins)
        debug_mesh = draw_body_origins(mesh_bgr, projected_origins)
        debug = np.concatenate(
            (
                label_panel(debug_real, "projected FK body origins on real RGB"),
                label_panel(debug_mesh, "same origins on mesh render"),
            ),
            axis=1,
        )
        cv2.imwrite(str(panel_dir / f"{stem}_body_origins.jpg"), debug)
    return panel_path


def make_contact_sheet(paths: Iterable[Path], output_path: Path, columns: int = 1) -> None:
    if columns <= 0:
        raise ValueError("Contact-sheet column count must be positive")
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_width = 640
    thumbs = [
        cv2.resize(image, (thumb_width, int(image.shape[0] * thumb_width / image.shape[1]))) for image in images
    ]
    blank = np.zeros_like(thumbs[0])
    rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start : start + columns]
        row.extend([blank] * (columns - len(row)))
        rows.append(np.concatenate(row, axis=1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(rows, axis=0))


def write_panel_video(paths: Sequence[Path], output_path: Path, fps: float) -> None:
    if fps <= 0:
        raise ValueError("Panel-video FPS must be positive")
    if not paths:
        raise ValueError("No panel frames available for video")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Could not read first panel: {paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open panel video writer: {output_path}")
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Invalid panel frame: {path}")
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = build_parser().parse_args()
    model_path = resolve_path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Robot model not found: {model_path}")
    episode = load_episode(args.dataset_path, args.episode_index)

    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {episode.video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (video_width, video_height) != (args.width, args.height):
        raise ValueError(f"Video is {video_width}x{video_height}, camera is {args.width}x{args.height}")
    if frame_count != len(episode.table):
        raise ValueError(f"Video/state length mismatch: {frame_count} frames vs {len(episode.table)} rows")
    frame_indices = parse_frame_indices(args.frames, frame_count, args.stride, args.max_frames)
    state_indices = [index + args.state_frame_offset for index in frame_indices]
    invalid_state = [index for index in state_indices if index < 0 or index >= len(episode.table)]
    if invalid_state:
        raise ValueError(f"State frame indices out of range: {invalid_state}")

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / f"episode_{args.episode_index:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = add_ego_camera(
        model_path=model_path,
        camera_body=args.camera_body,
        camera_pos=args.camera_pos,
        camera_euler=args.camera_euler,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
        principal_pixel_offset=args.mujoco_principal_pixel_offset,
    )
    data = mujoco.MjData(model)
    mapping = build_joint_mapping(model, episode.joint_names)
    mimic_mapping = build_mimic_mapping(model)
    all_arm_ids = arm_geom_ids(model)
    arm_ids = all_arm_ids if args.mask_side == "all" else side_arm_geom_ids(model, args.mask_side)
    hand_ids = hand_geom_ids(model, arm_ids)
    hand_mount_path = resolve_path(args.left_hand_mount_yaml) if args.left_hand_mount_yaml else None
    hand_mount_parameters = None
    fitted_left_translation = None
    if hand_mount_path is not None:
        if args.mask_side == "right":
            raise ValueError("--left-hand-mount-yaml cannot be used with --mask-side right")
        with hand_mount_path.open(encoding="utf-8") as file:
            hand_mount_parameters = yaml.safe_load(file)
        left_ids = hand_geom_ids(model, side_arm_geom_ids(model, "left"))
        fitted_left_translation = apply_fitted_left_hand_translation(model, left_ids, hand_mount_parameters)
    if args.color_arm_links:
        color_arm_links(model)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    if not 0 <= args.visual_geom_group < len(scene_option.geomgroup):
        raise ValueError(f"Invalid geom group: {args.visual_geom_group}")
    scene_option.geomgroup[args.visual_geom_group] = 1

    report: dict[str, Any] = {
        "method": "qpos -> mj_forward -> visual-mesh rasterization",
        "calls_mj_step": False,
        "dataset_path": str(episode.dataset_path),
        "episode_index": args.episode_index,
        "video_path": str(episode.video_path),
        "model_path": str(model_path),
        "frame_count": frame_count,
        "selected_video_frames": frame_indices,
        "selected_state_frames": state_indices,
        "state_frame_offset": args.state_frame_offset,
        "joint_mapping_count": len(mapping),
        "mimic_mapping_count": len(mimic_mapping),
        "fitted_left_hand_mount_applied": fitted_left_translation is not None,
        "left_hand_mount_yaml": str(hand_mount_path) if hand_mount_path else None,
        "left_wrist_local_hand_translation_m": (
            fitted_left_translation.tolist() if fitted_left_translation is not None else None
        ),
        "recorded_ftp_finger_state_preserved": True,
        "camera": {
            "body": args.camera_body,
            "position_m": list(args.camera_pos),
            "euler_rad": list(args.camera_euler),
            "resolution": [args.width, args.height],
            "requested_intrinsics": {
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
            },
            "mujoco_principal_pixel_offset": (
                list(args.mujoco_principal_pixel_offset)
                if args.mujoco_principal_pixel_offset is not None
                else [args.cx - args.width / 2.0, args.cy - args.height / 2.0]
            ),
            "compiled_intrinsic": model.cam_intrinsic[camera_id].tolist(),
            "compiled_fovy_deg": float(model.cam_fovy[camera_id]),
        },
        "masks": {
            "robot": "all visible visual robot geoms",
            "side": args.mask_side,
            "arms": (
                f"descendants of {ARM_ROOT_BODIES}"
                if args.mask_side == "all"
                else f"descendants of {args.mask_side}_shoulder_pitch_link"
            ),
            "arm_body": "arms excluding hand meshes",
            "hands": "hand meshes only",
            "self_occlusion": "MuJoCo z-buffer",
            "real_scene_occlusion": "not applied (RGB-only source dataset)",
        },
        "visualization": {
            "panel_mask_kind": args.panel_mask_kind,
            "contact_columns": args.contact_columns,
        },
        "frames": [],
    }

    panel_paths: list[Path] = []
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        for video_index, state_index in zip(frame_indices, state_indices, strict=True):
            state = np.asarray(episode.table.iloc[state_index]["observation.state"], dtype=np.float64).reshape(-1)
            set_pose_and_forward(model, data, state, mapping, mimic_mapping)
            robot_mask, arms_mask, arm_body_mask, hands_mask, _ = render_masks(
                renderer, data, scene_option, arm_ids, hand_ids
            )
            mesh_rgb = None
            if not args.no_mesh_rgb:
                mesh_rgb = render_mesh_rgb(renderer, data, scene_option)
            bgr = read_video_frame(capture, video_index)
            projected_origins: list[tuple[str, tuple[int, int]]] = []
            if args.annotate_body_origins:
                projected_origins = project_body_origins(
                    model, data, camera_id, args.fx, args.fy, args.cx, args.cy
                )
            panel_path = save_frame_outputs(
                output_dir,
                video_index,
                state_index,
                bgr,
                robot_mask,
                arms_mask,
                arm_body_mask,
                hands_mask,
                mesh_rgb,
                args.panel_mask_kind,
                projected_origins,
            )
            panel_paths.append(panel_path)
            report["frames"].append(
                {
                    "video_frame": video_index,
                    "state_frame": state_index,
                    "robot_mask_pixels": int(robot_mask.sum()),
                    "arms_mask_pixels": int(arms_mask.sum()),
                    "arm_body_mask_pixels": int(arm_body_mask.sum()),
                    "hands_mask_pixels": int(hands_mask.sum()),
                    "panel": str(panel_path),
                }
            )
    finally:
        capture.release()
        renderer.close()

    contact_sheet = output_dir / "contact_sheet.jpg"
    make_contact_sheet(panel_paths, contact_sheet, args.contact_columns)
    report["contact_sheet"] = str(contact_sheet)
    panel_video_path = None
    if args.write_panel_video:
        panel_video_fps = args.panel_video_fps or episode.fps / args.stride
        panel_video_path = output_dir / "fk_mask_overlay.mp4"
        write_panel_video(panel_paths, panel_video_path, panel_video_fps)
        report["panel_video"] = str(panel_video_path)
        report["panel_video_fps"] = panel_video_fps
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    vertical_fov = math.degrees(2.0 * math.atan(args.height / (2.0 * args.fy)))
    print(f"Rendered {len(frame_indices)} frames with FK only (no mj_step).")
    print(f"Requested pinhole vertical FOV: {vertical_fov:.3f} deg")
    print(f"Contact sheet: {contact_sheet}")
    if panel_video_path is not None:
        print(f"Panel video: {panel_video_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
