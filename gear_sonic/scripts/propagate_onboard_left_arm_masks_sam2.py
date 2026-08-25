"""Propagate audited left-arm masks through an onboard video with SAM 2.

The conditioning masks must be created from RGB images and audited by a human.
This script deliberately does not import the FK renderer, robot model, recorded
joint states, or camera calibration.  The resulting supervision is therefore
independent of the extrinsics that it will later be used to estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Propagate audited left-arm masks with SAM 2 video tracking.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seed-mask-dir", type=Path, required=True)
    parser.add_argument("--seed-frames", default="20,235,470,705")
    parser.add_argument("--model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-stride", type=int, default=25)
    parser.add_argument("--min-temporal-iou", type=float, default=0.55)
    parser.add_argument("--min-area-ratio", type=float, default=0.75)
    parser.add_argument("--max-area-ratio", type=float, default=1.35)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def load_seed_masks(mask_dir: Path, frames: list[int], shape: tuple[int, int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for frame in frames:
        path = mask_dir / f"video_{frame:06d}_left.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.shape != shape:
            raise ValueError(f"Seed mask {path} has shape {mask.shape}, expected {shape}")
        masks[frame] = mask > 127
    return masks


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.logical_and(first, second).sum() / max(np.logical_or(first, second).sum(), 1))


def overlay(image: np.ndarray, mask: np.ndarray, frame: int, suspicious: bool, is_seed: bool) -> np.ndarray:
    result = image.copy()
    color = (0, 180, 255) if suspicious else (40, 235, 60)
    tint = np.full_like(result, color)
    result[mask] = cv2.addWeighted(result, 0.38, tint, 0.62, 0.0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    label = "SEED" if is_seed else ("SUSPICIOUS" if suspicious else "PROPAGATED")
    cv2.rectangle(result, (0, 0), (result.shape[1], 31), (10, 10, 10), -1)
    cv2.putText(
        result,
        f"SAM2 LEFT ARM  frame {frame:06d}  {label}",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.audit_stride <= 0:
        raise ValueError("--audit-stride must be positive")
    if not 0.0 <= args.min_temporal_iou <= 1.0:
        raise ValueError("--min-temporal-iou must lie in [0, 1]")
    if not 0.0 < args.min_area_ratio <= args.max_area_ratio:
        raise ValueError("Invalid area-ratio interval")

    video = args.video.expanduser().resolve()
    seed_dir = args.seed_mask_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mask_dir = output_dir / "masks"
    audit_dir = output_dir / "audit_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seed_frames = sorted({int(value) for value in args.seed_frames.split(",") if value.strip()})
    if not seed_frames or seed_frames[0] < 0 or seed_frames[-1] >= frame_count:
        raise ValueError(f"Invalid seed frames for {frame_count}-frame video: {seed_frames}")
    seeds = load_seed_masks(seed_dir, seed_frames, (height, width))

    predictor = SAM2VideoPredictor.from_pretrained(args.model_id, device=args.device)
    state = predictor.init_state(
        video_path=str(video),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=True,
    )
    for frame, mask in seeds.items():
        predictor.add_new_mask(state, frame_idx=frame, obj_id=1, mask=mask)

    predicted: dict[int, np.ndarray] = {}
    autocast_enabled = args.device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for frame, object_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=seed_frames[0],
            reverse=False,
        ):
            object_index = list(object_ids).index(1)
            predicted[int(frame)] = (logits[object_index, 0] > 0.0).cpu().numpy()
        if seed_frames[0] > 0:
            for frame, object_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=seed_frames[0],
                max_frame_num_to_track=seed_frames[0] + 1,
                reverse=True,
            ):
                object_index = list(object_ids).index(1)
                predicted[int(frame)] = (logits[object_index, 0] > 0.0).cpu().numpy()

    missing = sorted(set(range(frame_count)) - set(predicted))
    if missing:
        raise RuntimeError(f"SAM 2 did not return masks for frames: {missing[:20]}")

    records: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    try:
        for frame in range(frame_count):
            mask = predicted[frame]
            cv2.imwrite(str(mask_dir / f"video_{frame:06d}_left.png"), mask.astype(np.uint8) * 255)
            area = int(mask.sum())
            temporal_iou = None if previous is None else mask_iou(previous, mask)
            previous_area = area if previous is None else int(previous.sum())
            area_ratio = None if previous is None else area / max(previous_area, 1)
            suspicious = bool(
                area < 100
                or (
                    temporal_iou is not None
                    and temporal_iou < args.min_temporal_iou
                    and area_ratio is not None
                    and not args.min_area_ratio <= area_ratio <= args.max_area_ratio
                )
            )
            is_seed = frame in seeds
            seed_iou = mask_iou(mask, seeds[frame]) if is_seed else None
            records.append(
                {
                    "frame": frame,
                    "area_px": area,
                    "temporal_iou_prev": temporal_iou,
                    "area_ratio_prev": area_ratio,
                    "suspicious": suspicious,
                    "is_seed": is_seed,
                    "seed_iou": seed_iou,
                }
            )
            if frame % args.audit_stride == 0 or suspicious or is_seed:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame}")
                cv2.imwrite(str(audit_dir / f"video_{frame:06d}.jpg"), overlay(bgr, mask, frame, suspicious, is_seed))
            previous = mask
    finally:
        capture.release()

    report = {
        "method": "SAM 2 video propagation from human-audited RGB masks",
        "independent_of_fk_and_camera_extrinsics": True,
        "video": str(video),
        "model_id": args.model_id,
        "device": args.device,
        "frame_count": frame_count,
        "resolution": [width, height],
        "seed_mask_dir": str(seed_dir),
        "seed_frames": seed_frames,
        "suspicious_frame_count": sum(record["suspicious"] for record in records),
        "frames": records,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(
        f"Wrote {frame_count} independent SAM 2 masks; "
        f"{report['suspicious_frame_count']} temporal outliers: {output_dir}"
    )


if __name__ == "__main__":
    main()
