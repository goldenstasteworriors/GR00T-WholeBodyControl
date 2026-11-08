"""Assemble existing primitive cuts and the curated dispense replacement for training.

Copies the original-speed camera files without encoding. Only episode, global
frame and task indices are rewritten; source boundaries and signals stay intact.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from segment_lab_primitives import PROMPTS, episode_path, numeric_stats, read_json, video_keys, write_json


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def assemble(primitives, dispense, output):
    info = deepcopy(read_json(primitives / "meta/info.json"))
    curated_info = read_json(dispense / "meta/info.json")
    if info["features"] != curated_info["features"] or info["fps"] != curated_info["fps"]:
        raise ValueError("Dataset features or frame rates differ")
    original = read_json(primitives / "meta/segment_provenance.json")
    curated = read_json(dispense / "meta/segment_provenance.json")
    replacements = {row["previous_v2_episode"]: row for row in curated}
    if len(replacements) != len(curated):
        raise ValueError("Duplicate replacement source episode")
    counters, chosen, replaced = Counter(), [], set()
    for row in original:
        task = row["task"]
        counters[task] += 1
        if task == "DISPENSE_TO_HELD_TUBE":
            replacement = replacements.get(row["output_episode"])
            if replacement is None:
                continue
            if any(row[key] != replacement[key] for key in ("dataset", "episode", "task")):
                raise ValueError("Dispense replacement maps to the wrong original trial")
            chosen.append((dispense, curated_info, replacement))
            replaced.add(row["output_episode"])
        else:
            chosen.append((primitives, info, {**row, "review_id": f"{task}-{counters[task]:03d}"}))
    if replaced != set(replacements):
        raise ValueError("Some curated dispensing episodes have no source slot")
    tasks = [task for task in PROMPTS if any(row["task"] == task for _, _, row in chosen)]
    task_indices = {task: i for i, task in enumerate(tasks)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "meta").mkdir()
    episodes, stats, provenance, checks, total = [], [], [], [], 0
    for episode, (source, source_info, row) in enumerate(chosen):
        source_episode = row["output_episode"]
        before = pd.read_parquet(episode_path(source, source_info, source_episode))
        count = row["end_frame"] - row["start_frame"]
        if len(before) != count:
            raise ValueError(f"Source cut length mismatch: {row['review_id']}")
        frame = before.copy()
        frame["episode_index"] = episode
        frame["index"] = np.arange(total, total + count, dtype=np.int64)
        frame["task_index"] = task_indices[row["task"]]
        path = episode_path(output, info, episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        saved = pd.read_parquet(path)
        for column in before:
            if column not in {"episode_index", "index", "task_index"}:
                np.testing.assert_array_equal(np.stack(before[column]), np.stack(saved[column]))
        for column in ("episode_index", "index", "task_index"):
            np.testing.assert_array_equal(frame[column].to_numpy(), saved[column].to_numpy())
        hashes = {}
        for view in video_keys(info):
            source_video = episode_path(source, source_info, source_episode, view)
            target = episode_path(output, info, episode, view)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, target)
            expected = digest(source_video)
            if digest(target) != expected:
                raise ValueError(f"Video copy mismatch: {target}")
            hashes[view] = expected
        episodes.append({"episode_index": episode, "tasks": [PROMPTS[row["task"]]], "length": count})
        stats.append({"episode_index": episode, "stats": numeric_stats(frame)})
        provenance.append({**row, "output_episode": episode, "source_cut_dataset": str(source.resolve()),
                           "source_cut_episode": source_episode,
                           "boundary_review": "reviewed_dispense" if row["task"] == "DISPENSE_TO_HELD_TUBE" else "v2_pending_user_video_review"})
        checks.append({"review_id": row["review_id"], "episode": episode, "frames": count, "video_sha256": hashes})
        total += count
        if (episode + 1) % 25 == 0:
            print(f"Assembled and verified {episode + 1}/{len(chosen)} episodes", flush=True)
    info.update(total_episodes=len(chosen), total_frames=total, total_tasks=len(tasks),
                total_videos=len(chosen) * len(video_keys(info)),
                total_chunks=(len(chosen) + info["chunks_size"] - 1) // info["chunks_size"],
                splits={"train": f"0:{len(chosen)}"})
    info.pop("discarded_episode_indices", None)
    info["processing"] = {"method": "assemble existing original-speed cuts; replace all old dispense cuts with curated set",
                          "sources": [str(primitives.resolve()), str(dispense.resolve())],
                          "video_reencoded": False, "source_data_unchanged": True,
                          "other_class_boundaries": "v2_pending_user_video_review"}
    write_json(output / "meta/info.json", info)
    for name, records in (("episodes", episodes), ("episodes_stats", stats),
                          ("tasks", [{"task_index": task_indices[t], "task": PROMPTS[t]} for t in tasks])):
        (output / "meta" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    shutil.copy2(primitives / "meta/modality.json", output / "meta/modality.json")
    write_json(output / "meta/primitive_prompts.json", {t: PROMPTS[t] for t in tasks})
    write_json(output / "meta/segment_provenance.json", provenance)
    write_json(output / "meta/dispense_exclusions.json", read_json(dispense / "meta/annotations.json")["rejected"])
    summary = {"episodes": len(chosen), "frames": total, "fps": info["fps"], "seconds": total / info["fps"],
               "task_counts": dict(Counter(row["task"] for _, _, row in chosen))}
    write_json(output / "meta/processing_summary.json", summary)
    write_json(output / "meta/assembly_validation.json", {
        "all_retained_signals_and_camera_timestamps_exact": True,
        "all_video_copies_sha256_match": True, "episodes": checks})
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--dispense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.primitives, args.dispense, args.output)


if __name__ == "__main__":
    main()
