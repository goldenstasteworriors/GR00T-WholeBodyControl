"""Build small observed validation-frame fixtures for the downloadable deployment."""
import argparse
import json
from pathlib import Path
import os
import hashlib

import lz4.frame
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    poses, pose_entries = {}, []
    tasks = json.loads((args.training_root / "cache/complete.json").read_text())["tasks"]
    for task in sorted(tasks):
        manifest = json.loads((args.training_root / "cache" / task / "manifest.json").read_text())
        for entry in manifest["rows"]:
            with np.load(args.training_root / "cache/episodes" / str(entry["episode"]) / "observations.npz") as data:
                values = data["state"].copy()
            key = f"episode_{entry['episode']}"
            poses[key] = values
            pose_entries.append(dict(task=task, episode=entry["episode"], frames=len(values), key=key))
        row = next(r for r in manifest["rows"] if r["episode"] in manifest["validation"])
        cut = (row["length"] - 2) // 2
        folder = args.training_root / "cache/episodes" / str(row["episode"])
        with np.load(folder / "observations.npz") as archive:
            numeric = {k: archive[k] for k in archive.files}
        indices = np.unique(np.maximum(0, cut - np.array([24, 16, 8, 0])))
        offsets = np.load(folder / "rgb-offsets.npy")
        frames = []
        fd = os.open(folder / "rgb.lz4", os.O_RDONLY)
        try:
            for index in indices:
                a, b = map(int, offsets[index:index + 2])
                raw = lz4.frame.decompress(os.pread(fd, b - a, a))
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(3, 240, 240, 3).transpose(0, 3, 1, 2))
        finally:
            os.close(fd)
        future = np.minimum(cut + 1 + np.arange(32), row["length"] - 1)
        np.savez_compressed(args.output / f"{task}.npz", state=numeric["state"][cut], images=np.stack(frames),
                            frame_times=(numeric["camera_time"][indices] - numeric["timestamp"][cut]).astype(np.float32),
                            truth=numeric["state"][future])
        records.append(dict(task=task, episode=row["episode"], cut=cut, source="held-out observation only"))
    (args.output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    from contract import JOINT_NAMES
    pose_root = args.output.parent / "initial_poses"
    pose_root.mkdir(exist_ok=True)
    np.savez_compressed(pose_root / "catalog.npz", **poses)
    catalog = dict(source_dataset="lab_primitives_train_20261110_boxed_v4", source_column="observation.state",
                   source_indices=list(range(15,31)), joint_names=JOINT_NAMES, fps=30,
                   array_file="catalog.npz", sha256=hashlib.sha256((pose_root / "catalog.npz").read_bytes()).hexdigest(),
                   episodes=sorted(pose_entries, key=lambda e:e["episode"]))
    (pose_root / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
