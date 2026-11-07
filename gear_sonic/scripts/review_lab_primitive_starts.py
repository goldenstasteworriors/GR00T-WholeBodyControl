"""Audit bilateral ready poses and render numbered one-second start comparisons.

Candidates are review suggestions, not automatically approved training boundaries.
Every v2 episode keeps its ID, including episodes without a valid earlier ready pose.
All robot state and video files are read-only; this tool never connects to a robot.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter

from segment_lab_primitives import episode_path, read_json, video_keys, write_json


TITLES = {
    "TAKE_DROPPER_AND_ASPIRATE": "拿滴管并取液",
    "TAKE_TUBE": "取试管",
    "DISPENSE_TO_HELD_TUBE": "向持有试管加液",
    "RETURN_DROPPER": "放回滴管",
    "RETURN_TUBE_TO_RACK": "放回试管架",
    "PLACE_TUBE_IN_BATH": "停步后放入烧杯",
    "WALK_BACKWARD": "后退",
    "WALK_LEFT_HOLDING_TUBE": "持试管向左走",
    "WALK_FORWARD_TO_BATH": "向前走至烧杯前停步",
}


def task_grip_mask(action, task):
    left, right = action[:, -2], action[:, -1]
    held_left = (left > .25) & (left < 4)
    held_right = (right > .25) & (right < 4)
    if task == "TAKE_TUBE":
        return right > 4
    if task == "TAKE_DROPPER_AND_ASPIRATE":
        return left > 4
    if task == "DISPENSE_TO_HELD_TUBE":
        return held_left & held_right
    if task == "RETURN_DROPPER":
        return held_left
    return held_right


def audit(root, dataset):
    annotation = read_json(dataset / "meta/annotations.json")
    provenance = read_json(dataset / "meta/segment_provenance.json")
    cache, counters, records = {}, Counter(), []
    for segment in provenance:
        name, episode, task = segment["dataset"], segment["episode"], segment["task"]
        key = (name, episode)
        if key not in cache:
            info = read_json(root / name / "meta/info.json")
            df = pd.read_parquet(episode_path(root / name, info, episode))
            eef = median_filter(np.stack(df["observation.eef_state"])[:, [2, 9]], size=(5, 1))
            action = np.stack(df["action.wbc"])
            # Reference follows each episode's calibration/body height. A joint
            # upper plateau is used, rather than either arm's separate maximum.
            reference = float(np.quantile(np.min(eef, axis=1), .90))
            threshold = max(.18, reference - .015)
            high = np.min(eef, axis=1) >= threshold
            stable = np.convolve(high.astype(int), np.ones(6, dtype=int), mode="full")[5:len(high)+5] == 6
            cache[key] = (info, eef, action, reference, threshold, stable)
        info, eef, action, reference, threshold, high = cache[key]
        fps = info["fps"]
        old = segment["start_frame"]
        valid = high & task_grip_mask(action, task)
        proposed, status = old, "KEEP_CANDIDATE"
        reason = "Bilateral height and task gripper-state candidate at original start; visual confirmation required."
        if not valid[old]:
            # Locomotion stages and stationary placement cannot borrow frames
            # across direction changes or before the final stop.
            if task.startswith("WALK_") or task == "PLACE_TUBE_IN_BATH":
                lo = old
            else:
                lo = max(0, old - 6 * fps)
            hi = min(segment["end_frame"] - fps, old + 2 * fps)
            # Never skip an opening/closing/squeeze event just to find a pose.
            grip = task_grip_mask(action, task)
            previous_bad = np.flatnonzero(~grip[lo:old])
            if len(previous_bad):
                lo += int(previous_bad[-1]) + 1
            bad = np.flatnonzero(~grip[old:hi+1])
            if len(bad):
                hi = min(hi, old + int(bad[0]) - 1)
            candidates = np.flatnonzero(valid[lo:hi+1]) + lo if hi >= lo else np.array([], dtype=int)
            earlier = candidates[candidates <= old]
            later = candidates[candidates > old]
            if len(earlier):
                proposed, status = int(earlier[-1]), "EARLIER_CANDIDATE"
                reason = "Earlier bilateral upper-pose candidate with required object/gripper state."
            elif len(later):
                proposed, status = int(later[0]), "LATER_CANDIDATE"
                reason = "Upper-pose candidate reached shortly after old boundary, before the next gripper event."
            else:
                status = "NO_READY_POSE_IN_WINDOW"
                reason = "No suitable bilateral upper pose before task event within search window; keep visible for manual review."
        counters[task] += 1
        records.append({
            **segment, "review_id": f"{task}-{counters[task]:03d}",
            "class_number": counters[task], "old_start_frame": old,
            "proposed_start_frame": proposed, "status": status, "reason": reason,
            "old_wrist_z": eef[old].tolist(), "proposed_wrist_z": eef[proposed].tolist(),
            "bilateral_reference_z": reference, "candidate_height_threshold": threshold,
            "reference_is_weak": reference < .19,
            "fps": fps, "requires_visual_confirmation": True,
        })
    proposal = deepcopy(annotation)
    for s, r in zip(proposal["segments"], records):
        s["start_frame"] = r["proposed_start_frame"]
        s["boundary_review_id"] = r["review_id"]
        s["boundary_review_status"] = r["status"]
    proposal["version"] = "bilateral_start_candidates_NOT_APPROVED"
    proposal["review_warning"] = "Review candidates only. Exclusions must be recomputed after human-approved boundaries; v2 dataset has not been overwritten."
    return records, proposal


def frames_at(path, start, count, fps):
    pictures = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(int(Fraction(start, fps) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            index = round(float(frame.time) * fps)
            if index < start:
                continue
            if index != start + len(pictures):
                raise ValueError(f"Unexpected frame {index} in {path}")
            pictures.append(frame.reformat(width=320, height=240, format="rgb24").to_image())
            if len(pictures) == count:
                break
    if len(pictures) != count:
        raise ValueError(f"Insufficient frames in {path}")
    return pictures


def render_class(job):
    root, output, task, records = job
    fps = records[0]["fps"]
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    colors = {"KEEP_CANDIDATE": "#244d39", "EARLIER_CANDIDATE": "#164c75",
              "LATER_CANDIDATE": "#164c75", "NO_READY_POSE_IN_WINDOW": "#854619"}
    target = output / f"{task}.mp4"
    with av.open(str(target), "w") as container:
        stream = container.add_stream("libx264", rate=fps, options={"crf": "20", "preset": "fast"})
        stream.width, stream.height, stream.pix_fmt = 960, 640, "yuv420p"
        stream.thread_count = 1
        frame_number = 0
        for record in records:
            source = root / record["dataset"]
            info = read_json(source / "meta/info.json")
            views = video_keys(info)
            starts = [record["old_start_frame"], record["proposed_start_frame"]]
            sequences = []
            for start in starts:
                sequences.append([frames_at(episode_path(source, info, record["episode"], view), start, fps, fps) for view in views])
            base = Image.new("RGB", (960, 640), "#111820")
            draw = ImageDraw.Draw(base)
            draw.rectangle((0, 0, 960, 72), fill=colors[record["status"]])
            draw.text((10, 4), f"{record['review_id']} / {len(records):03d}   |   v2 EP{record['output_episode']:06d}   {TITLES[task]}", font=font, fill="white")
            draw.text((10, 34), f"来源 {record['dataset']}  原始 EP{record['episode']:06d}", font=small, fill="white")
            for row, start in enumerate(starts):
                y = 74 + row * 283
                title = "原起点" if row == 0 else "建议起点（待确认）"
                draw.text((10, y), f"{title}: frame {start} / {start/fps:.2f}s   |   {record['status'] if row else 'V2'}", font=small, fill="white")
                for j, label in enumerate(["头部", "左腕", "右腕"]):
                    draw.text((j * 320 + 140, 340 + row * 283), label, font=small, fill="white")
            for k in range(fps):
                picture = base.copy()
                for row in range(2):
                    for j in range(3):
                        picture.paste(sequences[row][j][k], (j * 320, 100 + row * 283))
                frame = av.VideoFrame.from_image(picture)
                frame.pts, frame.time_base = frame_number, Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
                frame_number += 1
            print(f"Rendered {record['review_id']}", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    return {"task": task, "episodes": len(records), "frames": frame_number,
            "seconds": frame_number / fps, "video": target.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Raw dataset parent")
    parser.add_argument("--dataset", type=Path, required=True, help="Previous processed dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, proposal = audit(args.root, args.dataset)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)
    summary = {"episodes": len(records), "status_counts": dict(Counter(r["status"] for r in records)),
               "weak_reference_episodes": sum(r["reference_is_weak"] for r in records),
               "method": "Per-source episode bilateral wrist-height upper plateau and task gripper state; 6 consecutive high frames. Candidates only: neither height nor gripper target proves a visually correct ready pose.",
               "search": "At most 6s earlier / 2s later; do not cross next gripper event or borrow earlier locomotion/placement frames.",
               "dataset_unchanged": True, "review_records": records}
    write_json(args.output / "boundary_review.json", summary)
    write_json(args.output / "candidate_annotations_NOT_APPROVED.json", proposal)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        videos = list(pool.map(render_class, [(args.root, args.output, task, rows) for task, rows in grouped.items()]))
    write_json(args.output / "video_index.json", videos)
    print(json.dumps({k: v for k, v in summary.items() if k != "review_records"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
