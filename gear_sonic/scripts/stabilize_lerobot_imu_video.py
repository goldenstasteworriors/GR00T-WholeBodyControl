"""Generate an IMU-guided stabilization preview for a LeRobot episode.

The script uses ``observation.root_orientation`` together with the optional
G1 waist joint angles to approximate the camera orientation.  It intentionally
does not modify the source dataset.  The generated videos are diagnostics for
choosing a future-target construction method, not calibrated training data.

Example:
    .venv_data_collection/bin/python \
        gear_sonic/scripts/stabilize_lerobot_imu_video.py \
        --dataset outputs/onboard/8_18/8_18_use_cylinder_1 \
        --episode-index 11 \
        --camera-delay-frames 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


OUTPUT_FILENAMES = (
    "imu_stabilized_full.mp4",
    "imu_stabilized_safe_crop.mp4",
    "original_vs_imu_stabilized.mp4",
    "common_valid_mask.png",
    "metrics.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--camera-delay-frames",
        type=int,
        default=0,
        help="Exporter rows by which the camera trails the stored orientation.",
    )
    parser.add_argument("--focal-px", type=float, default=450.0)
    parser.add_argument("--camera-down-pitch-deg", type=float, default=45.0)
    parser.add_argument("--smooth-sigma-seconds", type=float, default=0.35)
    parser.add_argument("--new-camera-frame-mad", type=float, default=0.15)
    parser.add_argument(
        "--use-waist-kinematics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_episode_paths(
    dataset: Path,
    episode_index: int,
) -> tuple[Path, Path, dict]:
    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_chunk = episode_index // 1000
    format_args = {
        "episode_chunk": episode_chunk,
        "episode_index": episode_index,
    }
    data_pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    video_pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    parquet_path = dataset / data_pattern.format(**format_args)
    video_path = dataset / video_pattern.format(
        **format_args,
        video_key="observation.images.ego_view",
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Episode video not found: {video_path}")
    return parquet_path, video_path, info


def decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return frames, fps


def detect_camera_updates(
    frames: list[np.ndarray],
    difference_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    differences = np.zeros(len(frames), dtype=np.float64)
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for index, frame in enumerate(frames[1:], start=1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        differences[index] = float(np.mean(cv2.absdiff(gray, previous)))
        previous = gray

    is_update = differences > difference_threshold
    is_update[0] = True
    starts = np.flatnonzero(is_update)
    group_for_row = np.empty(len(frames), dtype=np.int64)
    for group_index, start in enumerate(starts):
        end = starts[group_index + 1] if group_index + 1 < len(starts) else len(frames)
        group_for_row[start:end] = group_index
    return starts, group_for_row


def approximate_torso_orientations(
    frame_table: pd.DataFrame,
    use_waist_kinematics: bool,
) -> np.ndarray:
    quaternion_wxyz = np.stack(
        frame_table["observation.root_orientation"].to_numpy()
    )
    base_orientation = Rotation.from_quat(quaternion_wxyz, scalar_first=True)
    if not use_waist_kinematics:
        return base_orientation.as_matrix()

    state = np.stack(frame_table["observation.state"].to_numpy())
    waist_orientation = (
        Rotation.from_euler("z", state[:, 12])
        * Rotation.from_euler("x", state[:, 13])
        * Rotation.from_euler("y", state[:, 14])
    )
    return (base_orientation * waist_orientation).as_matrix()


def torso_from_camera_rotation(camera_down_pitch_deg: float) -> np.ndarray:
    # Camera: x right, y down, z forward. Torso: x forward, y left, z up.
    torso_from_level_camera = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    downward_pitch = Rotation.from_euler(
        "x",
        -camera_down_pitch_deg,
        degrees=True,
    ).as_matrix()
    return torso_from_level_camera @ downward_pitch


def smooth_orientations(
    orientations: np.ndarray,
    sigma_frames: float,
) -> np.ndarray:
    reference = orientations[0]
    relative = np.einsum("ij,njk->nik", reference.T, orientations)
    tangent = Rotation.from_matrix(relative).as_rotvec()
    smooth_tangent = gaussian_filter1d(
        tangent,
        sigma=sigma_frames,
        axis=0,
        mode="nearest",
    )
    return np.einsum(
        "ij,njk->nik",
        reference,
        Rotation.from_rotvec(smooth_tangent).as_matrix(),
    )


def build_homographies(
    torso_world: np.ndarray,
    starts: np.ndarray,
    group_for_row: np.ndarray,
    fps: float,
    width: int,
    height: int,
    camera_delay_frames: int,
    focal_px: float,
    camera_down_pitch_deg: float,
    smooth_sigma_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    capture_rows = np.clip(
        starts - camera_delay_frames,
        0,
        len(torso_world) - 1,
    )
    raw_torso = torso_world[capture_rows]
    duration_seconds = len(group_for_row) / fps
    camera_update_fps = len(starts) / duration_seconds
    virtual_torso = smooth_orientations(
        raw_torso,
        sigma_frames=smooth_sigma_seconds * camera_update_fps,
    )

    torso_from_camera = torso_from_camera_rotation(camera_down_pitch_deg)
    raw_camera_world = np.einsum(
        "nij,jk->nik",
        raw_torso,
        torso_from_camera,
    )
    virtual_camera_world = np.einsum(
        "nij,jk->nik",
        virtual_torso,
        torso_from_camera,
    )
    virtual_from_raw = np.einsum(
        "nji,njk->nik",
        virtual_camera_world,
        raw_camera_world,
    )

    intrinsic = np.array(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    group_homographies = np.einsum(
        "ij,njk,kl->nil",
        intrinsic,
        virtual_from_raw,
        np.linalg.inv(intrinsic),
    )
    return group_homographies[group_for_row], group_homographies


def compute_common_valid_mask(
    homographies: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.full((height, width), 255, dtype=np.uint8)
    common = source.copy()
    for homography in homographies:
        valid = cv2.warpPerspective(
            source,
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        common = cv2.bitwise_and(common, valid)
    return cv2.erode(
        common,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )


def centered_safe_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = mask.shape
    aspect = width / height
    for scale in np.linspace(1.0, 0.70, 301):
        crop_width = int(width * scale) // 2 * 2
        crop_height = int(crop_width / aspect) // 2 * 2
        x0 = (width - crop_width) // 2
        y0 = (height - crop_height) // 2
        x1 = x0 + crop_width
        y1 = y0 + crop_height
        if np.all(mask[y0:y1, x0:x1] > 0):
            return x0, y0, x1, y1
    raise RuntimeError("No fully valid centered crop remains above 70% scale")


def stabilize_frames(
    frames: list[np.ndarray],
    homographies: np.ndarray,
    crop: tuple[int, int, int, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    height, width = frames[0].shape[:2]
    x0, y0, x1, y1 = crop
    full_frames: list[np.ndarray] = []
    cropped_frames: list[np.ndarray] = []
    for frame, homography in zip(frames, homographies, strict=True):
        full = cv2.warpPerspective(
            frame,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cropped = cv2.resize(
            full[y0:y1, x0:x1],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        full_frames.append(full)
        cropped_frames.append(cropped)
    return full_frames, cropped_frames


def global_motion_statistics(
    frames: list[np.ndarray],
    starts: np.ndarray,
) -> dict[str, float]:
    translations: list[float] = []
    rotations: list[float] = []
    for left_index, right_index in zip(starts[:-1], starts[1:], strict=True):
        left = cv2.cvtColor(frames[left_index], cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(frames[right_index], cv2.COLOR_BGR2GRAY)
        height, _ = left.shape
        background_mask = np.zeros_like(left)
        background_mask[: int(height * 0.62), :] = 255
        points = cv2.goodFeaturesToTrack(
            left,
            maxCorners=250,
            qualityLevel=0.01,
            minDistance=8,
            mask=background_mask,
        )
        if points is None or len(points) < 12:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(left, right, points, None)
        if tracked is None or status is None:
            continue
        valid = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[valid]
        destination = tracked.reshape(-1, 2)[valid]
        if len(source) < 12:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 8:
            continue
        translations.append(float(np.linalg.norm(affine[:, 2])))
        rotations.append(
            float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
        )

    translation = np.asarray(translations)
    rotation = np.abs(np.asarray(rotations))
    if len(translation) == 0:
        return {"pairs": 0.0}
    return {
        "pairs": float(len(translation)),
        "translation_median_px": float(np.median(translation)),
        "translation_p90_px": float(np.percentile(translation, 90)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90)),
    }


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def ensure_output_is_safe(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs; pass --overwrite if intended:\n"
            f"{paths}"
        )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    parquet_path, video_path, _ = resolve_episode_paths(
        dataset,
        args.episode_index,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset / f"preview_episode_{args.episode_index:06d}_imu_stabilization_sync"
    )
    ensure_output_is_safe(output_dir, args.overwrite)

    frame_table = pd.read_parquet(parquet_path)
    frames, fps = decode_video(video_path)
    if len(frames) != len(frame_table):
        raise RuntimeError(
            f"Video/parquet length mismatch: {len(frames)} vs {len(frame_table)}"
        )

    starts, group_for_row = detect_camera_updates(
        frames,
        args.new_camera_frame_mad,
    )
    torso_world = approximate_torso_orientations(
        frame_table,
        args.use_waist_kinematics,
    )
    height, width = frames[0].shape[:2]
    row_homographies, group_homographies = build_homographies(
        torso_world=torso_world,
        starts=starts,
        group_for_row=group_for_row,
        fps=fps,
        width=width,
        height=height,
        camera_delay_frames=args.camera_delay_frames,
        focal_px=args.focal_px,
        camera_down_pitch_deg=args.camera_down_pitch_deg,
        smooth_sigma_seconds=args.smooth_sigma_seconds,
    )
    common_mask = compute_common_valid_mask(
        group_homographies,
        width,
        height,
    )
    crop = centered_safe_crop(common_mask)
    full_stable, cropped_stable = stabilize_frames(
        frames,
        row_homographies,
        crop,
    )

    comparison: list[np.ndarray] = []
    for original, stable in zip(frames, cropped_stable, strict=True):
        left = original.copy()
        right = stable.copy()
        cv2.putText(
            left,
            "original",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            right,
            "synchronized IMU preview",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        comparison.append(np.hstack([left, right]))

    write_video(output_dir / "imu_stabilized_full.mp4", full_stable, fps)
    write_video(
        output_dir / "imu_stabilized_safe_crop.mp4",
        cropped_stable,
        fps,
    )
    write_video(
        output_dir / "original_vs_imu_stabilized.mp4",
        comparison,
        fps,
    )
    cv2.imwrite(str(output_dir / "common_valid_mask.png"), common_mask)

    x0, y0, x1, y1 = crop
    metrics = {
        "status": "preview_only_missing_camera_calibration",
        "source_parquet": str(parquet_path),
        "source_video": str(video_path),
        "episode_frames": len(frames),
        "export_fps": fps,
        "detected_camera_updates": int(len(starts)),
        "estimated_camera_update_fps": float(len(starts) / (len(frames) / fps)),
        "camera_delay_export_frames": args.camera_delay_frames,
        "assume_camera_imu_synchronized": args.camera_delay_frames == 0,
        "focal_px_assumed": args.focal_px,
        "camera_down_pitch_deg_assumed": args.camera_down_pitch_deg,
        "smooth_sigma_seconds": args.smooth_sigma_seconds,
        "use_waist_kinematics": args.use_waist_kinematics,
        "common_valid_fraction": float(np.mean(common_mask > 0)),
        "safe_crop_xyxy": [x0, y0, x1, y1],
        "safe_crop_fraction": float(
            ((x1 - x0) * (y1 - y0)) / (width * height)
        ),
        "original_global_motion": global_motion_statistics(frames, starts),
        "stabilized_global_motion": global_motion_statistics(full_stable, starts),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
