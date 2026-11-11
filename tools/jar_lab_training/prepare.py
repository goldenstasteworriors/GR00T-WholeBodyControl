"""Prepare observation-only G1 laboratory demonstrations for JAR."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from jar.mixed_cache import atomic_json, digest_json, encode_rgb, decode_rgb


CAMERAS = ["observation.images.ego_view", "observation.images.left_wrist", "observation.images.right_wrist"]
TASKS = ["TAKE_TUBE", "TAKE_DROPPER_AND_ASPIRATE", "DISPENSE_TO_HELD_TUBE",
         "RETURN_DROPPER", "RETURN_TUBE_TO_RACK", "PLACE_TUBE_IN_BATH"]
JOINTS = [f"{side}_{joint}_joint" for side in ("left", "right")
          for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                        "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINTS += ["left_hand_dex1_motor", "right_hand_dex1_motor"]
READ_COLUMNS = ["observation.state", "timestamp", "frame_index", "episode_index", "task_index"]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_episode(job):
    source, cache, row, indices, info, task_text = job
    cv2.setNumThreads(1)
    source, cache = Path(source), Path(cache)
    episode = row["output_episode"]
    folder = cache / "episodes" / str(episode)
    folder.mkdir(parents=True, exist_ok=True)
    path = source / info["data_path"].format(episode_chunk=episode // 1000, episode_index=episode)
    columns = list(READ_COLUMNS)
    schema = pq.read_schema(path).names
    if "observation.camera_timestamps" in schema:
        columns.append("observation.camera_timestamps")
    table = pq.read_table(path, columns=columns).to_pydict()
    state = np.asarray(table["observation.state"], dtype=np.float32)[:, indices]
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    n = len(state)
    if n < 2 or state.shape != (n, 16) or not np.isfinite(state).all():
        raise ValueError(f"Invalid observation.state episode {episode}")
    if not np.allclose(timestamp, np.arange(n) / 30, atol=2e-5):
        raise ValueError(f"Unexpected 30 Hz timestamp clock: {episode}")
    if not np.array_equal(np.asarray(table["frame_index"]), np.arange(n)):
        raise ValueError(f"Frame index differs from video frame number: {episode}")
    if set(table["episode_index"]) != {episode} or len(set(table["task_index"])) != 1:
        raise ValueError("Mixed episode/task identifiers")
    # Processed videos are explicitly aligned by frame number. Source camera
    # timestamps are wall-clock timestamps, not this trimmed episode's clock.
    # Keep their relative inter-camera delays without inventing future images:
    # the latest camera in each synchronized tuple is observed at this tick.
    camera_time = np.repeat(timestamp[:, None], 3, axis=1)
    skew = 0.
    if "observation.camera_timestamps" in table:
        capture = np.asarray(table["observation.camera_timestamps"], dtype=np.float64)
        if capture.shape != (n, 3) or not np.isfinite(capture).all():
            raise ValueError("Invalid source camera timestamps")
        camera_time += capture - capture.max(axis=1, keepdims=True)
        skew = float(np.max(np.ptp(capture, axis=1)))
    if np.any(camera_time > timestamp[:, None] + 1e-7):
        raise ValueError("Future camera leak")
    with (folder / "observations.npz").open("wb") as stream:
        np.savez(stream, state=state, timestamp=timestamp, camera_time=camera_time)
    caps = []
    source_files = [{"path": str(path.relative_to(source)), "sha256": sha256(path)}]
    for camera in CAMERAS:
        video = source / info["video_path"].format(episode_chunk=episode // 1000,
                                                  episode_index=episode, video_key=camera)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 30) > .01:
            raise ValueError(f"Invalid video {video}")
        caps.append(cap)
        source_files.append({"path": str(video.relative_to(source)), "sha256": sha256(video)})
    offsets = [0]
    decoded_hash = hashlib.sha256()
    try:
        with (folder / "rgb.lz4").open("wb") as stream:
            for frame in range(n):
                pictures = []
                for cap in caps:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f"Missing video frame {episode}:{frame}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pictures.append(cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA))
                pictures = np.stack(pictures)
                payload = encode_rgb(pictures)
                if not np.array_equal(decode_rgb(payload, 240), pictures):
                    raise ValueError("RGB roundtrip failed")
                decoded_hash.update(pictures.tobytes())
                stream.write(payload)
                offsets.append(offsets[-1] + len(payload))
            if any(cap.read()[0] for cap in caps):
                raise ValueError(f"Video has extra frames: {episode}")
    finally:
        for cap in caps:
            cap.release()
    np.save(folder / "rgb-offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    receipt = dict(episode=episode, length=n, task=row["task"], instruction=task_text,
                   source_group=[row["dataset"], row["episode"]],
                   parquet_columns_read=columns, ignored_fields="all action.*, teleop.*, bbox and review annotations",
                   selected_source_indices=indices, max_camera_skew_seconds=skew,
                   rgb_bytes=offsets[-1], decoded_rgb_sha256=decoded_hash.hexdigest(),
                   observation_sha256=sha256(folder / "observations.npz"),
                   rgb_sha256=sha256(folder / "rgb.lz4"), source_files=source_files)
    atomic_json(folder / "complete.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text())
    assert info["fps"] == 30
    names = info["features"]["observation.state"]["names"]
    indices = [names.index(name) for name in JOINTS]
    assert len(set(indices)) == 16 and indices == list(range(15, 31))
    rows = [r for r in json.loads((source / "meta/segment_provenance.json").read_text()) if r["task"] in TASKS]
    metadata = {r["episode_index"]: r for r in map(json.loads, (source / "meta/episodes.jsonl").read_text().splitlines())}
    receipts, jobs = [], []
    for row in rows:
        task_text = metadata[row["output_episode"]]["tasks"]
        assert len(task_text) == 1
        receipt = output / "episodes" / str(row["output_episode"]) / "complete.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old["selected_source_indices"] == indices
            assert old["observation_sha256"] == sha256(receipt.parent / "observations.npz")
            assert old["rgb_sha256"] == sha256(receipt.parent / "rgb.lz4")
            assert all(sha256(source / f["path"]) == f["sha256"] for f in old["source_files"])
            receipts.append(old)
        else:
            jobs.append((str(source), str(output), row, indices, info, task_text[0]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for receipt in pool.map(prepare_episode, jobs):
            receipts.append(receipt)
            print(json.dumps(dict(episode=receipt["episode"], task=receipt["task"], completed=len(receipts), total=len(rows))), flush=True)
    configs = json.loads(Path(args.base_config).read_text())
    configs.update(state_dim=16, action_dim=16, gripper_indices=[14, 15], camera_keys=CAMERAS,
                   text_tokenizer_path=str(Path(args.official).resolve()))
    atomic_json(output / "config.json", configs)
    for task in TASKS:
        task_rows = sorted((r for r in receipts if r["task"] == task), key=lambda r: r["episode"])
        groups = sorted({tuple(r["source_group"]) for r in task_rows})
        order = np.random.default_rng(42).permutation(len(groups))
        validation_groups = {groups[i] for i in order[:max(1, int(np.ceil(.1 * len(groups))))]}
        validation = [r["episode"] for r in task_rows if tuple(r["source_group"]) in validation_groups]
        train = [r["episode"] for r in task_rows if r["episode"] not in validation]
        if not train or not validation:
            raise ValueError(f"Insufficient source episode groups: {task}")
        values = np.concatenate([np.load(output / "episodes" / str(i) / "observations.npz")["state"] for i in train])
        low, high = np.quantile(values, [.01, .99], axis=0)
        low[-2:], high[-2:] = values[:, -2:].min(axis=0), values[:, -2:].max(axis=0)
        manifest = dict(task=task, rows=task_rows, train=train, validation=validation,
                        normalization=dict(low=low.tolist(), high=high.tolist(), training_rows=len(values)),
                        joint_names=JOINTS, source_indices=indices, fps=30, bins=512, horizon=32,
                        target="observation.state[t+1:t+33, source_indices]; never action fields",
                        input="observation.state[t, source_indices] and causal three-view RGB history",
                        history="unique(max(0,t-[24,16,8,0]))", validation_seed=42,
                        split_unit="original (dataset,episode), all its segments kept together",
                        camera_time="trimmed video frame clock plus source per-tuple camera delays relative to latest capture",
                        inference="predict future observed positions, not actuator commands; no robot execution is implemented")
        manifest["signature"] = digest_json(manifest)
        atomic_json(output / task / "manifest.json", manifest)
        print(json.dumps(dict(task=task, train_episodes=len(train), validation_episodes=len(validation))), flush=True)
    atomic_json(output / "complete.json", dict(tasks=TASKS, episodes=len(rows), joint_names=JOINTS,
                source=str(source), action_columns_read=False, rgb_roundtrip_verified=True))


if __name__ == "__main__":
    main()
