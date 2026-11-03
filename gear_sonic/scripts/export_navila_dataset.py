"""Review PICO demonstrations and export official NaVILA vlnce annotations.

Draft quantities are command integrals only. Export requires explicit human
review of action amounts/completion; no synthetic odometry or success labels.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_episode(root, info, episode):
    return pd.read_parquet(root / info["data_path"].format(
        episode_index=episode, episode_chunk=episode // info["chunks_size"]))


def draft_segments(df, episode):
    commands = np.stack(df["navigation.policy_velocity"].to_numpy())
    # LeRobot serializes shape-(1,) features as scalar parquet columns.
    times = np.asarray(df["navigation.camera_time"].tolist(), dtype=float).reshape(-1)
    modes = np.asarray(df["navigation.control_mode"].tolist(), dtype=int).reshape(-1)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-increasing camera times in episode {episode}")
    categories = []
    for (vx, vy, wz), mode in zip(commands, modes):
        if mode != 3:
            action = "inactive"
        elif abs(vy) > .02 or vx < -.02 or (abs(vx) > .02 and abs(wz) > .05):
            action = "unsupported"
        elif abs(wz) > .05:
            action = "left" if wz > 0 else "right"
        elif vx > .02:
            action = "forward"
        else:
            action = "idle"
        categories.append(action)
    segments = []
    boundaries = [0] + [i for i in range(1, len(df)) if categories[i] != categories[i - 1]] + [len(df)]
    for start, end in zip(boundaries, boundaries[1:]):
        action = categories[start]
        if action in ("inactive", "idle"):
            continue
        axis = 0 if action == "forward" else 2
        # No extrapolated final interval beyond the last recorded observation.
        amount = float(np.sum(commands[start:end - 1, axis] * np.diff(times[start:end])))
        if action in ("left", "right"):
            amount = abs(math.degrees(amount))
        segments.append({"episode_index": episode, "start_frame": start, "end_frame": end,
                         "action": action, "amount": None, "reviewed": False,
                         "command_integral_hint": amount if action != "unsupported" else None})
    # A terminal stop is a review candidate, never assumed successful.
    segments.append({"episode_index": episode, "start_frame": len(df) - 1,
                     "end_frame": len(df), "action": "stop", "success": False, "reviewed": False})
    return segments


def validate_segment(segment, df):
    start, end = segment["start_frame"], segment["end_frame"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(df):
        raise ValueError(f"Invalid frame interval: {segment}")
    if segment.get("reviewed") is not True:
        raise ValueError("Every exported segment needs reviewed=true; exclude unused segments")
    action = segment["action"]
    if action not in ("forward", "left", "right", "stop"):
        raise ValueError("NaVILA standard actions: forward, left, right, stop; exclude lateral/backward/mixed segments")
    modes = np.asarray(df["navigation.control_mode"].iloc[start:end].tolist()).reshape(-1)
    if np.any(modes != 3):
        raise ValueError("Training action interval must be in running mode")
    if action == "stop":
        if segment.get("success") is not True:
            raise ValueError("Stop labels require explicit success=true; pause/ESTOP is not success")
        return "Stop."
    amount = segment.get("amount")
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount <= 0:
        raise ValueError("Provide a reviewed positive amount: forward meters or turn degrees")
    # The existing NaVILA execution bridge parses forward distances in cm.
    return f"Move forward {amount * 100:g} cm." if action == "forward" else f"Turn {action} {amount:g} degrees."


def export(root, labels_path, output):
    info = json.loads((root / "meta/info.json").read_text())
    tasks = {row["task_index"]: row["task"] for row in
             map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    labels = json.loads(labels_path.read_text())
    if not labels:
        raise ValueError("No reviewed navigation segments")
    samples, needed, tables = [], {}, {}
    for index, segment in enumerate(labels):
        episode = segment["episode_index"]
        if type(episode) is not int or not 0 <= episode < info["total_episodes"]:
            raise ValueError(f"Unknown episode: {episode}")
        if episode in info.get("discarded_episode_indices", []):
            raise ValueError(f"Episode {episode} was discarded during collection")
        if episode not in tables:
            tables[episode] = read_episode(root, info, episode)
        df = tables[episode]
        answer = validate_segment(segment, df)
        current = segment["start_frame"]
        # All context is at or before the decision, never future action frames.
        history = np.unique(np.linspace(0, current, min(8, current + 1), dtype=int)).tolist()
        needed.setdefault(episode, set()).update(history)
        instruction = segment.get("instruction", tasks[int(df.iloc[current]["task_index"])])
        samples.append({"id": f"ep{episode:06d}_{index:06d}", "video_id": f"episode_{episode:06d}",
                        "frames": [f"frames/episode_{episode:06d}/{frame:06d}.jpg" for frame in history],
                        "q": instruction, "a": answer,
                        "conversations": [{"from": "human", "value": instruction}, {"from": "gpt", "value": answer}]})
    # Validate everything before creating output; never overwrite an earlier export.
    output.mkdir(parents=True, exist_ok=False)
    for episode, frames in needed.items():
        video = root / info["video_path"].format(episode_index=episode,
            episode_chunk=episode // info["chunks_size"], video_key="observation.images.ego_view")
        capture = cv2.VideoCapture(str(video))
        directory = output / "frames" / f"episode_{episode:06d}"
        directory.mkdir(parents=True)
        try:
            for frame in range(max(frames) + 1):
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {frame}: {video}")
                if frame in frames and not cv2.imwrite(str(directory / f"{frame:06d}.jpg"), image):
                    raise OSError("Failed to write navigation image")
        finally:
            capture.release()
    (output / "annotations.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    (output / "reviewed_segments.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(samples)} samples: {output / 'annotations.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--make-template", type=Path, help="Write an unreviewed label draft; no robot access")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.make_template:
        info = json.loads((args.dataset / "meta/info.json").read_text())
        draft = [segment for episode in range(info["total_episodes"])
                 if episode not in info.get("discarded_episode_indices", [])
                 for segment in draft_segments(read_episode(args.dataset, info, episode), episode)]
        with args.make_template.open("x") as stream:
            json.dump(draft, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Draft only: review amounts, remove unsupported segments and confirm success before export.")
    elif args.labels and args.output:
        export(args.dataset, args.labels, args.output)
    else:
        parser.error("Use --make-template PATH, or --labels PATH --output DIRECTORY")


if __name__ == "__main__":
    main()
