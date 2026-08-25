"""Generate auditable dense left-arm SAM pseudo masks using FK only as prompts.

Each recorded state is assigned to ``qpos`` and passed through
``mujoco.mj_forward``.  Its rendered complete left-arm silhouette supplies a
box plus positive points to SAM; it is *not* copied into the output mask.
This permits SAM to follow the RGB evidence while retaining the left/right
identity needed for camera calibration.  Every frame is saved, and the report
marks frames that should be excluded from fitting because their SAM/FK
agreement is too weak or their mask has an implausible area.

No physics step is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import mujoco
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from gear_sonic.scripts.calibrate_onboard_fk_camera import physical_angles_to_quaternion, quaternion_to_mjcf_euler
from gear_sonic.scripts.calibrate_onboard_fk_camera_from_masks import side_arm_geom_ids
from gear_sonic.scripts.render_onboard_fk_mask import (
    CAMERA_NAME,
    DEFAULT_MODEL_PATH,
    add_ego_camera,
    build_joint_mapping,
    build_mimic_mapping,
    load_episode,
    resolve_path,
    set_pose_and_forward,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/onboard_fk_mask_experiment/checkpoints/sam_vit_b_01ec64.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate dense, auditable left-arm SAM pseudo masks guided by FK.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--camera-body", default="torso_link")
    parser.add_argument("--camera-pos", type=float, nargs=3, required=True)
    parser.add_argument("--camera-pitch-yaw-roll-deg", type=float, nargs=3, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--box-padding-px", type=int, default=28)
    parser.add_argument("--min-sam-score", type=float, default=0.72)
    parser.add_argument("--min-fk-iou", type=float, default=0.16)
    parser.add_argument("--min-area-ratio", type=float, default=0.35)
    parser.add_argument("--max-area-ratio", type=float, default=2.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def prompt_points(mask: np.ndarray, count: int = 8) -> np.ndarray:
    """Pick well-spaced interior pixels, preserving disconnected hand pieces."""
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    candidates = np.argwhere(distance >= max(2.0, np.percentile(distance[mask], 65)))
    if len(candidates) == 0:
        candidates = np.argwhere(mask)
    selected: list[np.ndarray] = []
    for y, x in candidates[np.argsort(distance[candidates[:, 0], candidates[:, 1]])[::-1]]:
        candidate = np.array([x, y], dtype=np.float32)
        if not selected or min(np.linalg.norm(candidate - item) for item in selected) >= 22.0:
            selected.append(candidate)
        if len(selected) >= count:
            break
    return np.asarray(selected, dtype=np.float32)


def retain_prompt_components(mask: np.ndarray, positive: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    selected: set[int] = set()
    height, width = mask.shape
    for x, y in positive:
        label = int(labels[int(np.clip(round(y), 0, height - 1)), int(np.clip(round(x), 0, width - 1))])
        if label:
            selected.add(label)
    return np.isin(labels, list(selected)) if selected else np.zeros_like(mask, dtype=bool)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.logical_and(first, second).sum() / max(np.logical_or(first, second).sum(), 1))


def overlay(image: np.ndarray, fk: np.ndarray, sam: np.ndarray, frame: int, usable: bool) -> np.ndarray:
    result = image.copy()
    for mask, color in ((fk, (0, 0, 255)), (sam, (0, 255, 0))):
        tint = np.full_like(result, color)
        result[mask] = cv2.addWeighted(result, 0.55, tint, 0.45, 0)[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 1, cv2.LINE_AA)
    status = "USABLE" if usable else "REJECTED (audit only)"
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (12, 12, 12), -1)
    cv2.putText(result, f"frame {frame:06d}  green SAM target / red FK prompt  {status}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_stride <= 0 or args.box_padding_px < 0:
        raise ValueError("--frame-stride must be positive and --box-padding-px must be non-negative")
    if not 0.0 <= args.min_sam_score <= 1.0 or not 0.0 <= args.min_fk_iou <= 1.0:
        raise ValueError("Mask score thresholds must lie in [0, 1]")
    if not 0.0 < args.min_area_ratio <= args.max_area_ratio:
        raise ValueError("Invalid area-ratio interval")

    episode = load_episode(args.dataset_path, args.episode_index)
    output_dir = resolve_path(args.output_dir)
    mask_dir, audit_dir = output_dir / "masks", output_dir / "audit_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    pitch_yaw_roll = np.radians(args.camera_pitch_yaw_roll_deg)
    model = add_ego_camera(
        resolve_path(args.model_path), args.camera_body, args.camera_pos,
        quaternion_to_mjcf_euler(physical_angles_to_quaternion(*pitch_yaw_roll)),
        args.fx, args.fy, args.cx, args.cy, 640, 480,
    )
    mapping, mimic_mapping = build_joint_mapping(model, episode.joint_names), build_mimic_mapping(model)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1
    renderer = mujoco.Renderer(model, height=480, width=640)
    left_ids = np.fromiter(side_arm_geom_ids(model, "left"), dtype=np.int32)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {episode.video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records: list[dict[str, Any]] = []
    try:
        for frame in range(0, frame_count, args.frame_stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame}")
            data = mujoco.MjData(model)
            state = np.asarray(episode.table.iloc[frame]["observation.state"], dtype=np.float64).reshape(-1)
            set_pose_and_forward(model, data, state, mapping, mimic_mapping)
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=CAMERA_NAME, scene_option=scene_option)
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            fk_mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[..., 0], left_ids)
            ys, xs = np.nonzero(fk_mask)
            if len(xs) < 20:
                records.append({"frame": frame, "usable": False, "reason": "FK arm not visible"})
                continue
            padding = args.box_padding_px
            box = np.array([max(0, xs.min() - padding), max(0, ys.min() - padding), min(639, xs.max() + padding), min(479, ys.max() + padding)], dtype=np.float32)
            positive = prompt_points(fk_mask)
            # Negative corner prompts prevent the padded box from absorbing background.
            negative = np.array([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]], dtype=np.float32)
            predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(
                point_coords=np.concatenate((positive, negative)),
                point_labels=np.concatenate((np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32))),
                box=box[None, :], multimask_output=True,
            )
            masks = np.stack([retain_prompt_components(mask, positive) for mask in masks])
            overlaps = [iou(mask, fk_mask) for mask in masks]
            # SAM confidence remains the dominant term; FK only resolves candidate identity.
            selected = int(np.argmax(np.asarray(scores) + 0.18 * np.asarray(overlaps)))
            target = masks[selected].astype(bool)
            fk_area, target_area = int(fk_mask.sum()), int(target.sum())
            ratio = target_area / max(fk_area, 1)
            overlap = overlaps[selected]
            usable = bool(scores[selected] >= args.min_sam_score and overlap >= args.min_fk_iou and args.min_area_ratio <= ratio <= args.max_area_ratio)
            reason = "usable" if usable else "low SAM/FK agreement or implausible area"
            cv2.imwrite(str(mask_dir / f"video_{frame:06d}_left.png"), target.astype(np.uint8) * 255)
            if frame % max(1, round(episode.fps)) == 0 or not usable:
                cv2.imwrite(str(audit_dir / f"video_{frame:06d}.jpg"), overlay(bgr, fk_mask, target, frame, usable))
            records.append({"frame": frame, "usable": usable, "reason": reason, "sam_predicted_iou": float(scores[selected]), "fk_iou": overlap, "area_ratio_sam_over_fk": ratio, "candidate_index": selected, "box_xyxy": box.tolist()})
            if frame % 50 == 0:
                print(f"frame {frame:06d}/{frame_count - 1}: {'usable' if usable else 'rejected'}; SAM={scores[selected]:.3f}, FK-IoU={overlap:.3f}", flush=True)
    finally:
        capture.release()
        renderer.close()
    usable_frames = [row["frame"] for row in records if row.get("usable")]
    report = {"method": "SAM with FK-derived prompt box/points; FK silhouette never copied into target", "is_ground_truth": False, "calls_mj_step": False, "state_frame_offset": 0, "left_only": True, "camera": {"body": args.camera_body, "pos": args.camera_pos, "physical_pitch_yaw_roll_deg": args.camera_pitch_yaw_roll_deg, "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy}, "selection": {"min_sam_score": args.min_sam_score, "min_fk_iou": args.min_fk_iou, "area_ratio": [args.min_area_ratio, args.max_area_ratio]}, "frame_count": frame_count, "frame_stride": args.frame_stride, "usable_frame_count": len(usable_frames), "usable_frames": usable_frames, "frames": records}
    with (output_dir / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (output_dir / "usable_frames.txt").open("w", encoding="utf-8") as file:
        file.write(",".join(str(frame) for frame in usable_frames) + "\n")
    print(f"Generated {len(records)} masks; {len(usable_frames)} passed automatic audit: {output_dir}")


if __name__ == "__main__":
    main()
