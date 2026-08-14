#!/usr/bin/env python3
"""Render synchronized onboard RGB and left RH56DFTP tactile videos.

The left half of every output frame is the recorded RGB stream.  The right
half arranges all 17 tactile matrices in an approximate left-hand layout and
uses a fixed color scale so contact strength remains comparable over time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

# Prefer the checkout that contains this script over a possibly stale editable
# installation when the script is invoked directly with ``python path/to/...``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.data_collection.inspire_tactile import (
    TACTILE_FORCE_COUNT,
    TACTILE_REGION_COUNT,
    TACTILE_REGIONS,
    TACTILE_TAXEL_COUNT,
)


TACTILE_VALUES_KEY = "observation.tactile.left_values"
TACTILE_VALID_KEY = "observation.tactile.left_valid"
TACTILE_AGE_KEY = "observation.tactile.left_age_ms"
TACTILE_FORCE_KEY = "observation.tactile.left_force_act_g"
TACTILE_FORCE_VALID_KEY = "observation.tactile.left_force_valid"
TIMESTAMP_KEY = "timestamp"
PANEL_WIDTH = 640
PANEL_HEIGHT = 480


@dataclass(frozen=True)
class RegionBox:
    name: str
    label: str
    x: int
    y: int
    width: int
    height: int


# Approximate palmar view of the left hand.  Matrices retain the row-major
# ordering used by record_rh56dftp_tactile.py and the dataset schema.
REGION_BOXES = (
    RegionBox("little_end", "L END", 23, 102, 60, 34),
    RegionBox("little_tip", "L TIP", 13, 150, 80, 92),
    RegionBox("little_pad", "L PAD", 13, 257, 80, 76),
    RegionBox("ring_end", "R END", 119, 62, 60, 34),
    RegionBox("ring_tip", "R TIP", 109, 110, 80, 92),
    RegionBox("ring_pad", "R PAD", 109, 217, 80, 76),
    RegionBox("middle_end", "M END", 215, 42, 60, 34),
    RegionBox("middle_tip", "M TIP", 205, 90, 80, 92),
    RegionBox("middle_pad", "M PAD", 205, 197, 80, 76),
    RegionBox("index_end", "I END", 311, 72, 60, 34),
    RegionBox("index_tip", "I TIP", 301, 120, 80, 92),
    RegionBox("index_pad", "I PAD", 301, 227, 80, 76),
    RegionBox("thumb_end", "T END", 447, 105, 60, 34),
    RegionBox("thumb_tip", "T TIP", 437, 153, 80, 92),
    RegionBox("thumb_mid", "T MID", 447, 260, 60, 34),
    RegionBox("thumb_pad", "T PAD", 437, 309, 80, 112),
    RegionBox("palm", "PALM", 104, 334, 278, 124),
)

FORCE_LABELS = ("little", "ring", "middle", "index", "thumb-b", "thumb-r")


def _fixed_list_to_numpy(table, key: str, dtype) -> np.ndarray:
    column = table[key].combine_chunks()
    if not hasattr(column.type, "list_size"):
        raise ValueError(f"{key} is not a fixed-size list column: {column.type}")
    return column.values.to_numpy(zero_copy_only=False).reshape(
        len(column), column.type.list_size
    ).astype(dtype, copy=False)


def load_episode_tactile(parquet_path: Path) -> dict[str, np.ndarray]:
    required = (
        TACTILE_VALUES_KEY,
        TACTILE_VALID_KEY,
        TACTILE_AGE_KEY,
        TACTILE_FORCE_KEY,
        TACTILE_FORCE_VALID_KEY,
        TIMESTAMP_KEY,
    )
    schema = pq.read_schema(parquet_path)
    missing = [key for key in required if key not in schema.names]
    if missing:
        raise ValueError(f"episode does not contain tactile fields: {', '.join(missing)}")
    table = pq.read_table(parquet_path, columns=list(required))

    values = _fixed_list_to_numpy(table, TACTILE_VALUES_KEY, np.uint16)
    valid = _fixed_list_to_numpy(table, TACTILE_VALID_KEY, np.bool_)
    age_ms = _fixed_list_to_numpy(table, TACTILE_AGE_KEY, np.float32)
    force = _fixed_list_to_numpy(table, TACTILE_FORCE_KEY, np.int16)
    force_valid = np.asarray(table[TACTILE_FORCE_VALID_KEY].to_numpy(), dtype=np.bool_)
    timestamps = np.asarray(table[TIMESTAMP_KEY].to_numpy(), dtype=np.float32)

    expected_shapes = {
        TACTILE_VALUES_KEY: (len(table), TACTILE_TAXEL_COUNT),
        TACTILE_VALID_KEY: (len(table), TACTILE_REGION_COUNT),
        TACTILE_AGE_KEY: (len(table), TACTILE_REGION_COUNT),
        TACTILE_FORCE_KEY: (len(table), TACTILE_FORCE_COUNT),
    }
    arrays = {
        TACTILE_VALUES_KEY: values,
        TACTILE_VALID_KEY: valid,
        TACTILE_AGE_KEY: age_ms,
        TACTILE_FORCE_KEY: force,
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {arrays[key].shape}")

    return {
        "values": values,
        "valid": valid,
        "age_ms": age_ms,
        "force": force,
        "force_valid": force_valid,
        "timestamps": timestamps,
    }


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.42,
    color: tuple[int, int, int] = (220, 230, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _contact_colormap(values: np.ndarray, max_value: float, gamma: float) -> np.ndarray:
    normalized = np.clip(values.astype(np.float32) / max_value, 0.0, 1.0)
    normalized = np.power(normalized, gamma)
    intensity = np.rint(normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)


def _draw_taxel_grid(
    panel: np.ndarray,
    matrix: np.ndarray,
    box: RegionBox,
    *,
    valid: bool,
    max_value: float,
    gamma: float,
) -> None:
    x0, y0 = box.x, box.y
    x1, y1 = x0 + box.width, y0 + box.height
    if valid:
        colored = _contact_colormap(matrix, max_value, gamma)
        panel[y0:y1, x0:x1] = cv2.resize(
            colored, (box.width, box.height), interpolation=cv2.INTER_NEAREST
        )
        border = (95, 105, 112)
    else:
        panel[y0:y1, x0:x1] = (48, 48, 52)
        cv2.line(panel, (x0, y0), (x1 - 1, y1 - 1), (80, 80, 170), 2)
        cv2.line(panel, (x1 - 1, y0), (x0, y1 - 1), (80, 80, 170), 2)
        border = (80, 80, 180)

    rows, cols = matrix.shape
    if min(box.width / cols, box.height / rows) >= 3.0:
        for col in range(1, cols):
            x = x0 + round(col * box.width / cols)
            cv2.line(panel, (x, y0), (x, y1 - 1), (28, 31, 36), 1)
        for row in range(1, rows):
            y = y0 + round(row * box.height / rows)
            cv2.line(panel, (x0, y), (x1 - 1, y), (28, 31, 36), 1)
    cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), border, 1)
    _draw_text(panel, box.label, (x0, y0 - 5), scale=0.34, color=(170, 182, 188))


def _draw_scale(panel: np.ndarray, max_value: float, gamma: float) -> None:
    ramp = np.linspace(0.0, max_value, 180, dtype=np.float32)[None, :]
    colored = _contact_colormap(ramp, max_value, gamma)
    colored = cv2.resize(colored, (180, 12), interpolation=cv2.INTER_LINEAR)
    panel[15:27, 434:614] = colored
    cv2.rectangle(panel, (434, 15), (613, 26), (100, 110, 116), 1)
    _draw_text(panel, "0", (434, 40), scale=0.32, color=(155, 166, 172))
    _draw_text(panel, f"{max_value:g}", (574, 40), scale=0.32, color=(155, 166, 172))


def render_tactile_panel(
    values: np.ndarray,
    valid: np.ndarray,
    age_ms: np.ndarray,
    force: np.ndarray,
    force_valid: bool,
    *,
    max_value: float,
    gamma: float,
    episode_index: int,
    frame_index: int,
    timestamp: float,
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (23, 29, 34), dtype=np.uint8)
    _draw_text(panel, "LEFT HAND TACTILE", (16, 27), scale=0.58, color=(238, 243, 245), thickness=1)
    _draw_scale(panel, max_value, gamma)

    offset = 0
    region_index = {region.name: index for index, region in enumerate(TACTILE_REGIONS)}
    region_values = {}
    for region in TACTILE_REGIONS:
        region_values[region.name] = values[offset : offset + region.size].reshape(
            region.rows, region.cols
        )
        offset += region.size

    for box in REGION_BOXES:
        index = region_index[box.name]
        _draw_taxel_grid(
            panel,
            region_values[box.name],
            box,
            valid=bool(valid[index]),
            max_value=max_value,
            gamma=gamma,
        )

    valid_ages = age_ms[valid]
    max_age = float(valid_ages.max()) if valid_ages.size else -1.0
    _draw_text(
        panel,
        f"ep {episode_index:03d}  frame {frame_index:05d}  t={timestamp:7.2f}s",
        (395, 448),
        scale=0.34,
        color=(164, 176, 183),
    )
    _draw_text(
        panel,
        f"valid {int(valid.sum())}/{len(valid)}  max age {max_age:.0f} ms",
        (395, 465),
        scale=0.34,
        color=(164, 176, 183),
    )

    force_x, force_y = 531, 74
    _draw_text(panel, "FORCE_ACT (g)", (force_x, force_y - 11), scale=0.34, color=(170, 182, 188))
    for index, (label, value) in enumerate(zip(FORCE_LABELS, force, strict=True)):
        y = force_y + index * 24
        color = (205, 215, 220) if force_valid else (100, 105, 110)
        _draw_text(panel, f"{label:7s} {int(value):5d}", (force_x, y), scale=0.32, color=color)
    return panel


def _fit_rgb(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    y = (PANEL_HEIGHT - resized.shape[0]) // 2
    x = (PANEL_WIDTH - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (168, 32), (8, 12, 16), -1)
    _draw_text(canvas, "RGB / EGO VIEW", (12, 23), scale=0.52, color=(238, 243, 245))
    return canvas


def _find_episode_data(dataset_root: Path, episode_index: int) -> Path:
    matches = sorted(
        (dataset_root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet file for episode {episode_index}, found {len(matches)}"
        )
    return matches[0]


def _find_episode_video(dataset_root: Path, video_key: str, episode_index: int) -> Path:
    matches = sorted(
        (dataset_root / "videos").glob(
            f"chunk-*/{video_key}/episode_{episode_index:06d}.mp4"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {video_key!r} video for episode {episode_index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _episode_indices(dataset_root: Path, selection: str) -> list[int]:
    episode_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    available = [int(path.stem.split("_")[-1]) for path in episode_files]
    if not available:
        raise FileNotFoundError(f"no episode parquet files under {dataset_root / 'data'}")
    if selection == "all":
        return available
    try:
        selected = int(selection)
    except ValueError as exc:
        raise ValueError("--episode-index must be a nonnegative integer or 'all'") from exc
    if selected not in available:
        raise ValueError(f"episode {selected} not found; available range is {available[0]}..{available[-1]}")
    return [selected]


def _output_path(
    dataset_root: Path,
    output: Path | None,
    video_key: str,
    episode_index: int,
    multiple: bool,
) -> Path:
    filename = f"episode_{episode_index:06d}_rgb_tactile.mp4"
    if output is None:
        return dataset_root / "visualizations" / video_key / filename
    if multiple or output.suffix.lower() != ".mp4":
        return output / filename
    return output


def render_episode(
    dataset_root: Path,
    episode_index: int,
    output_path: Path,
    *,
    video_key: str,
    fps: float,
    max_value: float,
    gamma: float,
    start_frame: int,
    max_frames: int,
    overwrite: bool,
) -> None:
    parquet_path = _find_episode_data(dataset_root, episode_index)
    video_path = _find_episode_video(dataset_root, video_key, episode_index)

    tactile = load_episode_tactile(parquet_path)
    total_rows = tactile["values"].shape[0]
    if not 0 <= start_frame < total_rows:
        raise ValueError(f"--start-frame must be in [0, {total_rows - 1}]")
    stop_frame = total_rows if max_frames <= 0 else min(total_rows, start_frame + max_frames)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial_path.exists():
        partial_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open RGB video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if source_frames < stop_frame:
        capture.release()
        raise ValueError(
            f"RGB video has {source_frames} frames but parquet requires {stop_frame}"
        )

    writer = cv2.VideoWriter(
        str(partial_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (PANEL_WIDTH * 2, PANEL_HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to create output video: {partial_path}")

    written = 0
    try:
        for frame_index in range(start_frame, stop_frame):
            ok, rgb = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decode failed at frame {frame_index}")
            tactile_panel = render_tactile_panel(
                tactile["values"][frame_index],
                tactile["valid"][frame_index],
                tactile["age_ms"][frame_index],
                tactile["force"][frame_index],
                bool(tactile["force_valid"][frame_index]),
                max_value=max_value,
                gamma=gamma,
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=float(tactile["timestamps"][frame_index]),
            )
            writer.write(np.hstack((_fit_rgb(rgb), tactile_panel)))
            written += 1
            if written % max(1, round(fps * 5)) == 0:
                print(f"  rendered {written}/{stop_frame - start_frame} frames", flush=True)
    finally:
        capture.release()
        writer.release()

    if written != stop_frame - start_frame:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"expected {stop_frame - start_frame} frames, wrote {written}")
    partial_path.replace(output_path)
    print(f"Wrote {written} frames: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB (left) plus RH56DFTP tactile contact (right) from an onboard dataset."
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root, e.g. outputs/onboard/name")
    parser.add_argument(
        "--episode-index",
        default="0",
        help="Episode number or 'all' (default: 0).",
    )
    parser.add_argument("--video-key", default="observation.images.ego_view")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mp4 for one episode, or output directory for --episode-index all.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; defaults to meta/info.json.")
    parser.add_argument("--max-value", type=float, default=4095.0, help="Tactile color-scale maximum.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Color gamma; below 1 reveals weak contact.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 renders the complete episode.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"dataset metadata not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    fps = float(info.get("fps", 0.0) if args.fps is None else args.fps)
    if fps <= 0.0:
        raise SystemExit("output FPS must be positive")
    if args.max_value <= 0.0:
        raise SystemExit("--max-value must be positive")
    if args.gamma <= 0.0:
        raise SystemExit("--gamma must be positive")
    if args.start_frame < 0 or args.max_frames < 0:
        raise SystemExit("--start-frame and --max-frames must be nonnegative")

    try:
        episodes = _episode_indices(dataset_root, args.episode_index)
        for episode_index in episodes:
            output_path = _output_path(
                dataset_root,
                args.output.expanduser().resolve() if args.output else None,
                args.video_key,
                episode_index,
                multiple=len(episodes) > 1,
            )
            render_episode(
                dataset_root,
                episode_index,
                output_path,
                video_key=args.video_key,
                fps=fps,
                max_value=args.max_value,
                gamma=args.gamma,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
