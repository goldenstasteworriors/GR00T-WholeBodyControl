"""Build two-task NaVILA SFT data with explicit command-integral weak labels.

Terminal task success must be confirmed by the demonstrator via the CLI flag.
Translation/yaw quantities remain weak supervision, never measured odometry.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def split_episodes(episodes, seed):
    mapping = {}
    rng = random.Random(seed)
    for category in (1, 3):
        group = sorted((e for e in episodes if e["category"] == category), key=lambda e: e["id"])
        rng.shuffle(group)
        validation = max(1, int(len(group) * .1 + .5))
        test = max(1, len(group) // 10)
        for i, episode in enumerate(group):
            mapping[episode["id"]] = "validation" if i < validation else "test" if i < validation + test else "train"
    return mapping


def motion_chunks(episode, df):
    commands = np.stack(df["navigation.policy_velocity"])
    times = np.asarray(df["navigation.camera_time"].tolist()).reshape(-1)
    dt = np.r_[np.diff(times), 0.]
    for segment in episode["segments"]:
        action = segment["action"]
        if action not in ("forward", "turn_left", "turn_right"):
            continue
        axis = 0 if action == "forward" else 2
        target = .25 if axis == 0 else math.radians(15)
        minimum = .05 if axis == 0 else math.radians(5)
        start, amount = segment["start_frame"], 0.
        for frame in range(start, segment["end_frame"]):
            amount += abs(commands[frame, axis]) * dt[frame]
            if amount >= target or frame == segment["end_frame"] - 1:
                end = frame + 1
                if amount >= minimum:
                    value = round(amount * 100) if axis == 0 else round(math.degrees(amount))
                    answer = f"Move forward {value} cm." if axis == 0 else f"Turn {'left' if action == 'turn_left' else 'right'} {value} degrees."
                    yield {"start_frame": start, "end_frame": end, "action": action,
                           "weak_amount_m_or_rad": float(amount), "a": answer,
                           "quantity_source": "policy_velocity_left_endpoint_integral",
                           "quantity_is_measured": False}
                start, amount = end, 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-terminal-success", action="store_true")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    if not args.confirm_terminal_success:
        parser.error("Demonstrator confirmation required: --confirm-terminal-success")
    review, output = args.review_root.resolve(), args.output.resolve()
    episodes = [e for e in json.loads((review / "episodes.json").read_text()) if e["category"] in (1, 3)]
    mapping = split_episodes(episodes, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    samples = {key: [] for key in ("train", "validation", "test")}
    provenance, omitted = [], Counter()
    for episode in episodes:
        df = pd.read_parquet(review / episode["source_parquet"])
        if len(df) != episode["frames"]:
            raise ValueError("Episode length changed")
        motions = list(motion_chunks(episode, df))
        stop = episode["terminal"]["start_frame"]
        if stop is None or not episode["terminal"]["stop_candidate"]:
            raise ValueError(f"No terminal stationary candidate: {episode['id']}")
        motions.append({"start_frame": stop, "end_frame": len(df), "action": "stop", "a": "Stop.",
                        "success": True, "success_source": "demonstrator_confirmed_2026-11-09",
                        "quantity_is_measured": False})
        modes = np.asarray(df["navigation.control_mode"].tolist()).reshape(-1)
        needed = set()
        for index, motion in enumerate(motions):
            start, end = motion["start_frame"], motion["end_frame"]
            if not np.all(modes[start:end] == 3):
                raise ValueError("Non-running action interval")
            # No future images. The official loader handles histories shorter than eight.
            frames = np.unique(np.linspace(0, start, min(start + 1, 8), dtype=int)).tolist()
            needed.update(frames)
            paths = [f"frames/{episode['id']}/{frame:06d}.jpg" for frame in frames]
            sample = {"id": f"{episode['id']}_{index:03d}", "video_id": episode["id"],
                      "frames": paths, "q": episode["instruction"], "a": motion["a"],
                      "conversations": [{"from": "human", "value": episode["instruction"]},
                                        {"from": "gpt", "value": motion["a"]}],
                      "label_quality": "weak_command_integral_with_user_confirmed_terminal_success"}
            samples[mapping[episode["id"]]].append(sample)
            provenance.append({"sample_id": sample["id"], "episode_id": episode["id"],
                               "source_episode_index": episode["source_episode_index"],
                               "category": episode["category"], "split": mapping[episode["id"]],
                               "history_frames": frames, **motion})
        for segment in episode["segments"]:
            if segment["action"] not in ("forward", "turn_left", "turn_right", "idle"):
                omitted[segment["action"]] += segment["end_frame"] - segment["start_frame"]
        directory = output / "frames" / episode["id"]
        directory.mkdir(parents=True)
        video = review / episode["source_videos"][0]
        capture = cv2.VideoCapture(str(video))
        try:
            for frame in range(max(needed) + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Missing source frame: {video} {frame}")
                if frame in needed and not cv2.imwrite(str(directory / f"{frame:06d}.jpg"), image):
                    raise OSError("Image write failed")
        finally:
            capture.release()
    for split, rows in samples.items():
        write_json(output / f"{split}.json", rows)
    write_json(output / "provenance.json", provenance)
    report = {"tasks": ["between_reagents_and_rack", "stop_before_beaker"],
              "episodes": len(episodes), "seed": args.seed,
              "episode_splits": dict(Counter(mapping.values())),
              "samples": {split: len(rows) for split, rows in samples.items()},
              "actions": {split: dict(Counter(p["action"] for p in provenance if p["split"] == split))
                          for split in samples},
              "excluded_unsupported_frames": dict(omitted),
              "distance_supervision": "WEAK: commanded velocity integration, not measured odometry",
              "stop_supervision": "User explicitly confirmed terminal stops successful on 2026-11-09",
              "history": "up to 8 ego images at or before action start; no future frame leakage",
              "split_unit": "complete source episode; no cross-split episode/frame reuse"}
    write_json(output / "dataset_report.json", report)
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "sha256.json", hashes)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
