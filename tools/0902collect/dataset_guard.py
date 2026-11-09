#!/usr/bin/env python3
"""Dataset safety checks used by the standalone 0902 collection workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def camera_urls_differ_only_by_host(stored: Any, configured: str) -> bool:
    """Allow an HTTP camera computer to move without rewriting dataset history."""
    if not isinstance(stored, str):
        return False
    try:
        old, new = urlsplit(stored), urlsplit(configured)
        return (
            old.scheme in ("http", "https")
            and bool(old.hostname)
            and bool(new.hostname)
            and old.hostname != new.hostname
            and (old.scheme, old.port, old.path, old.query, old.fragment, old.username, old.password)
            == (new.scheme, new.port, new.path, new.query, new.fragment, new.username, new.password)
        )
    except ValueError:
        return False


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSONL from {path}: {exc}")


def data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def video_keys(info: dict[str, Any]) -> list[str]:
    keys = list(info.get("video_keys", []))
    if not keys:
        keys = [
            key
            for key, feature in info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
    return keys


def video_path(
    root: Path, info: dict[str, Any], episode_index: int, video_key: str
) -> Path:
    pattern = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    chunk_size = int(info.get("chunks_size", 1000))
    return root / pattern.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def load_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    info = read_json(root / "meta" / "info.json")
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    stats_path = root / "meta" / "episodes_stats.jsonl"
    stats = read_jsonl(stats_path) if stats_path.exists() else []
    return info, episodes, stats


def validate_resume(
    root: Path,
    task: str | None = None,
    *,
    fps: int | None = None,
    hand_mode: str | None = None,
    hand_pose_config: str | None = None,
    camera_http_url: str | None = None,
    camera_mode: str | None = None,
    checkpoint: str | None = None,
    obs_config: str | None = None,
    planner: str | None = None,
    motion_data: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not root.is_dir():
        fail(f"resume dataset does not exist: {root}")
    info, episodes, stats = load_dataset(root)
    if camera_mode is not None:
        has_images = any(
            value.get("dtype") in ("image", "video")
            for value in info.get("features", {}).values()
        )
        if has_images != (camera_mode == "true"):
            fail("camera mode differs from dataset features; use a new dataset")
    indices = [int(row["episode_index"]) for row in episodes]
    expected = list(range(len(episodes)))
    if indices != expected:
        fail(f"episode indices are not contiguous: {indices}")
    if int(info.get("total_episodes", -1)) != len(episodes):
        fail(
            f"info.total_episodes={info.get('total_episodes')} but episodes.jsonl has {len(episodes)} rows"
        )
    lengths = [int(row["length"]) for row in episodes]
    if int(info.get("total_frames", -1)) != sum(lengths):
        fail(
            f"info.total_frames={info.get('total_frames')} but episode lengths sum to {sum(lengths)}"
        )
    discarded = info.get("discarded_episode_indices", [])
    if discarded:
        fail(f"discarded episodes still need cleanup: {discarded}")

    stats_indices = [int(row["episode_index"]) for row in stats]
    if stats and stats_indices != expected:
        fail(f"episode stats indices do not match episodes: {stats_indices} vs {expected}")

    if fps is not None and int(info.get("fps", -1)) != fps:
        fail(f"configured fps={fps} does not match dataset fps={info.get('fps')}")

    script_config = info.get("script_config", {})
    expected_config: dict[str, Any] = {}
    if hand_mode is not None:
        expected_config["hand_data_mode"] = hand_mode.replace("-", "_")
    if hand_pose_config is not None:
        expected_config["inspire_hand_pose_config"] = hand_pose_config
    if camera_http_url is not None:
        expected_config["camera_http_url"] = camera_http_url
    if checkpoint is not None:
        expected_config["model_path"] = f"{checkpoint}_decoder.onnx"
        expected_config["encoder_file"] = f"{checkpoint}_encoder.onnx"
    if obs_config is not None:
        expected_config["obs_config_path"] = obs_config
    if planner is not None:
        expected_config["planner_path"] = planner
    if motion_data is not None:
        expected_config["reference_motion_path"] = motion_data
    for key, configured_value in expected_config.items():
        stored_value = script_config.get(key)
        if stored_value != configured_value:
            if key == "camera_http_url" and camera_urls_differ_only_by_host(
                stored_value, configured_value
            ):
                print(
                    f"[WARN] camera host changed: {stored_value!r} -> {configured_value!r}; "
                    "allowing resume with the same stream settings"
                )
                continue
            fail(
                f"configured {key}={configured_value!r} does not match "
                f"dataset script_config value {stored_value!r}"
            )

    expected_data: set[Path] = set()
    expected_videos: set[Path] = set()
    for row in episodes:
        idx = int(row["episode_index"])
        parquet = data_path(root, info, idx)
        expected_data.add(parquet.resolve())
        if not parquet.is_file() or parquet.stat().st_size == 0:
            fail(f"missing/empty parquet for episode {idx}: {parquet}")
        for key in video_keys(info):
            video = video_path(root, info, idx, key)
            expected_videos.add(video.resolve())
            if not video.is_file() or video.stat().st_size == 0:
                fail(f"missing/empty video for episode {idx}, key {key}: {video}")

    actual_data = {path.resolve() for path in root.rglob("episode_*.parquet")}
    actual_videos = {path.resolve() for path in root.rglob("episode_*.mp4")}
    if actual_data != expected_data:
        fail("parquet file set contains missing or unfinished extra episodes")
    if actual_videos != expected_videos:
        fail("video file set contains missing or unfinished extra episodes")

    if task and episodes:
        task_rows = read_jsonl(root / "meta" / "tasks.jsonl")
        known_tasks = {str(row.get("task")) for row in task_rows}
        if task not in known_tasks:
            fail(f"configured task {task!r} is not present in dataset tasks {sorted(known_tasks)}")

    return info, episodes


def command_check(args: argparse.Namespace) -> None:
    root = Path(args.dataset_dir).resolve()
    if args.mode == "new":
        if root.exists():
            fail(f"new dataset already exists: {root}")
        print(f"[PASS] new dataset path is unused: {root}")
        return
    info, episodes = validate_resume(
        root,
        args.task,
        fps=args.fps,
        hand_mode=args.hand_mode,
        hand_pose_config=args.hand_pose_config,
        camera_http_url=args.camera_http_url,
        camera_mode=args.camera_mode,
        checkpoint=args.checkpoint,
        obs_config=args.obs_config,
        planner=args.planner,
        motion_data=args.motion_data,
    )
    print(
        f"[PASS] resumable dataset: episodes={len(episodes)} "
        f"frames={info['total_frames']} next_episode={len(episodes)}"
    )


def ffprobe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        fail(f"expected one video stream in {path}, found {len(streams)}")
    return streams[0]


def longest_exact_duplicate_run(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    previous = None
    run = 0
    maximum = 0
    count = 0
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        digest = line.rsplit(",", 1)[-1].strip()
        count += 1
        if digest == previous:
            run += 1
        else:
            previous = digest
            run = 1
        maximum = max(maximum, run)
    return count, maximum


def command_latest(args: argparse.Namespace) -> None:
    root = Path(args.dataset_dir).resolve()
    info, episodes = validate_resume(root)
    if not episodes:
        fail("dataset has no saved episodes")
    row = episodes[-1]
    idx = int(row["episode_index"])
    parquet = data_path(root, info, idx)
    keys = video_keys(info)
    if not keys:
        import pandas as pd

        parquet_frames = len(pd.read_parquet(parquet))
        if parquet_frames != int(row["length"]):
            fail(f"frame mismatch: metadata={row['length']} parquet={parquet_frames}")
        print(f"[PASS] latest episode={idx}, parquet={parquet_frames} frames, no video")
        return
    if "observation.images.ego_view" in keys:
        key = "observation.images.ego_view"
    elif keys:
        key = keys[0]
    else:
        fail("dataset has no video feature")
    video = video_path(root, info, idx, key)

    import pandas as pd

    parquet_frames = len(pd.read_parquet(parquet))
    stream = ffprobe_video(video)
    video_frames = int(stream.get("nb_read_frames", 0))
    decoded_frames, duplicate_run = longest_exact_duplicate_run(video)
    expected_frames = int(row["length"])
    if not (expected_frames == parquet_frames == video_frames == decoded_frames):
        fail(
            "frame mismatch: "
            f"metadata={expected_frames} parquet={parquet_frames} "
            f"ffprobe={video_frames} decoded={decoded_frames}"
        )

    fps_text = str(stream.get("avg_frame_rate", info.get("fps", "unknown")))
    try:
        numerator, denominator = fps_text.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = float(info.get("fps", 0))
    freeze_seconds = duplicate_run / fps if fps else 0.0
    print(f"[PASS] latest episode={idx}")
    print(f"  parquet: {parquet} ({parquet_frames} frames)")
    print(
        f"  video:   {video} ({stream.get('codec_name')} "
        f"{stream.get('width')}x{stream.get('height')} {fps_text} "
        f"duration={stream.get('duration')}s frames={video_frames})"
    )
    print(
        f"  longest exact duplicate run: {duplicate_run} frames "
        f"({freeze_seconds:.3f}s)"
    )


def command_recording(args: argparse.Namespace) -> None:
    path = Path(args.latency_log)
    value = "unknown"
    episode = "unknown"
    if path.is_file():
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = "true" if row.get("recording") else "false"
            episode = str(row.get("episode_index", "unknown"))
            break
    if args.value_only:
        print(value)
    else:
        print(f"recording={value} episode={episode} latency_log={path}")


def latest_latency_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"latency log does not exist: {path}")
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    fail(f"latency log has no valid JSON rows: {path}")


def command_health(args: argparse.Namespace) -> None:
    path = Path(args.latency_log)
    row = latest_latency_row(path)
    now = time.time()
    log_age_ms = max(0.0, (now - float(row.get("timestamp", 0.0))) * 1000.0)

    def metric(name: str) -> float:
        try:
            return float(row[name])
        except (KeyError, TypeError, ValueError):
            fail(f"latency row is missing numeric metric {name!r}")

    state_age = metric("source.state_update_age_ms")
    pose_age = float(row.get("source.sonic_pose_age_ms", float("inf")))
    planner_age = float(row.get("source.planner_command_age_ms", float("inf")))
    control_age = min(pose_age, planner_age)

    failures: list[str] = []
    limits = {
        "latency log": (log_age_ms, args.max_log_age_ms),
        "G1 state": (state_age, args.max_source_age_ms),
        "PICO pose/planner": (control_age, args.max_source_age_ms),
    }
    camera_status = "disabled"
    if not args.no_camera:
        camera_receive_age = metric("source.camera_update_age_ms")
        camera_timestamp_age = metric("source.camera_timestamp_age_ms")
        limits["camera receive"] = (camera_receive_age, args.max_source_age_ms)
        limits["camera timestamp"] = (camera_timestamp_age, args.max_camera_timestamp_age_ms)
        camera_status = f"{camera_timestamp_age:.1f}ms"
    for label, (value, limit) in limits.items():
        if value > limit:
            failures.append(f"{label} age {value:.1f} ms exceeds {limit:.1f} ms")

    print(
        "latency_health "
        f"recording={bool(row.get('recording'))} "
        f"episode={row.get('episode_index', 'unknown')} "
        f"log={log_age_ms:.1f}ms state={state_age:.1f}ms "
        f"camera={camera_status} control={control_age:.1f}ms"
    )
    if failures:
        fail("; ".join(failures))
    print("[PASS] required live data sources are fresh")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def command_prune_discarded_tail(args: argparse.Namespace) -> None:
    if args.confirm != "DELETE_DISCARDED_TAIL":
        fail("destructive confirmation token is missing")
    root = Path(args.dataset_dir).resolve()
    info, episodes, stats = load_dataset(root)
    indices = [int(row["episode_index"]) for row in episodes]
    if indices != list(range(len(indices))):
        fail(f"episode indices are not contiguous: {indices}")
    discarded = sorted({int(value) for value in info.get("discarded_episode_indices", [])})
    if not discarded:
        print("[PASS] no discarded episodes are present")
        return
    if discarded[0] > len(indices) or discarded[-1] > len(indices):
        fail(
            f"discarded indices skip beyond the next writable episode: "
            f"episodes={indices} discarded={discarded}"
        )
    expected_discarded = list(range(discarded[0], discarded[-1] + 1))
    persisted_discarded = [idx for idx in discarded if idx < len(indices)]
    if discarded != expected_discarded or indices[discarded[0] :] != persisted_discarded:
        fail(f"discarded episodes are not a contiguous tail: episodes={indices} discarded={discarded}")

    kept_episodes = [row for row in episodes if int(row["episode_index"]) not in discarded]
    kept_indices = [int(row["episode_index"]) for row in kept_episodes]
    if kept_indices != list(range(len(kept_indices))):
        fail(f"kept indices would not be contiguous: {kept_indices}")

    latest_stats: dict[int, dict[str, Any]] = {}
    for row in stats:
        idx = int(row["episode_index"])
        if idx in kept_indices:
            latest_stats[idx] = row
    if stats and sorted(latest_stats) != kept_indices:
        fail("cannot reconstruct one stats row per kept episode")

    # Complete every structural validation before the first destructive write.
    # The emergency-stop caller waits for the exporter to exit, so these files
    # are closed and represent only the explicitly flagged tail episodes.
    for idx in discarded:
        parquet = data_path(root, info, idx)
        if parquet.exists():
            parquet.unlink()
        for key in video_keys(info):
            video = video_path(root, info, idx, key)
            if video.exists():
                video.unlink()

    info["total_episodes"] = len(kept_episodes)
    info["total_frames"] = sum(int(row["length"]) for row in kept_episodes)
    info.pop("discarded_episode_indices", None)
    atomic_write_json(root / "meta" / "info.json", info)
    atomic_write_jsonl(root / "meta" / "episodes.jsonl", kept_episodes)
    if stats:
        atomic_write_jsonl(
            root / "meta" / "episodes_stats.jsonl",
            [latest_stats[idx] for idx in kept_indices],
        )
    print(f"[PASS] deleted discarded tail episodes: {discarded}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--dataset-dir", required=True)
    check.add_argument("--mode", choices=("new", "resume"), required=True)
    check.add_argument("--task")
    check.add_argument("--fps", type=int)
    check.add_argument("--hand-mode", choices=("right-only", "left-only", "both"))
    check.add_argument("--hand-pose-config")
    check.add_argument("--camera-http-url")
    check.add_argument("--camera-mode", choices=("true", "false"))
    check.add_argument("--checkpoint")
    check.add_argument("--obs-config")
    check.add_argument("--planner")
    check.add_argument("--motion-data")
    check.set_defaults(func=command_check)

    latest = subparsers.add_parser("latest")
    latest.add_argument("--dataset-dir", required=True)
    latest.set_defaults(func=command_latest)

    recording = subparsers.add_parser("recording")
    recording.add_argument("--latency-log", required=True)
    recording.add_argument("--value-only", action="store_true")
    recording.set_defaults(func=command_recording)

    health = subparsers.add_parser("health")
    health.add_argument("--latency-log", required=True)
    health.add_argument("--no-camera", action="store_true")
    health.add_argument("--max-log-age-ms", type=float, default=2000.0)
    health.add_argument("--max-source-age-ms", type=float, default=500.0)
    health.add_argument("--max-camera-timestamp-age-ms", type=float, default=1000.0)
    health.set_defaults(func=command_health)

    prune = subparsers.add_parser("prune-discarded-tail")
    prune.add_argument("--dataset-dir", required=True)
    prune.add_argument("--confirm", required=True)
    prune.set_defaults(func=command_prune_discarded_tail)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
