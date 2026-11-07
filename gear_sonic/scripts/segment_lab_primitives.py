"""Review and export manually annotated lab primitives from LeRobot v2.1 data.

All intervals use source frame indices [start, end). Source files are read-only.
No model service or robot connection is used.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROMPTS = {
    "TAKE_DROPPER_AND_ASPIRATE": "Use the left gripper to take the indicated dropper from the source bottle and aspirate reagent once. Keep holding the dropper.",
    "TAKE_TUBE": "Use the right gripper to pick up the indicated test tube and keep holding it.",
    "DISPENSE_TO_HELD_TUBE": "Use the dropper held by the left gripper to dispense into the test tube held by the right gripper. Do not aspirate again. Keep holding both objects.",
    "RETURN_DROPPER": "Use the left gripper to return the held dropper directly to the indicated source bottle and release it.",
    "RETURN_TUBE_TO_RACK": "Use the right gripper to place the held test tube into the indicated empty rack slot and release it once supported.",
    "PLACE_TUBE_IN_BATH": "Use the right gripper to place the held test tube at the indicated supported position in the water bath and release it once stable.",
    "WALK_BACKWARD": "Walk backward while keeping the test tube held by the right gripper, then stop.",
    "WALK_LEFT_HOLDING_TUBE": "Walk to the left while keeping the test tube held by the right gripper, then stop.",
    "WALK_FORWARD_TO_BATH": "Walk forward while keeping the test tube held by the right gripper. Stop in front of the indicated beaker, ready to place the tube inside. Keep holding the tube.",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def episode_path(root, info, episode, video_key=None):
    key = "video_path" if video_key else "data_path"
    return root / info[key].format(
        episode_index=episode, episode_chunk=episode // info["chunks_size"],
        video_key=video_key,
    )


def video_keys(info):
    return [k for k, v in info["features"].items() if v["dtype"] == "video"]


def review_episode(job):
    root, dest, episode, samples = job
    root, dest = Path(root), Path(dest)
    info = read_json(root / "meta/info.json")
    df = pd.read_parquet(episode_path(root, info, episode))
    n, fps = len(df), info["fps"]
    indices = np.unique(np.linspace(0, n - 1, samples).astype(int))
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    sheet = Image.new("RGB", (1280, rows * 258 + 220), "white")
    draw = ImageDraw.Draw(sheet)
    with av.open(str(episode_path(root, info, episode, "observation.images.ego_view"))) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        wanted = {int(i): j for j, i in enumerate(indices)}
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                j = wanted[i]
                image = frame.reformat(width=320, height=240, format="rgb24").to_image()
                x, y = j % cols * 320, j // cols * 258
                sheet.paste(image, (x, y))
                draw.text((x + 3, y + 240), f"ep{episode:02d} f{i} {i/fps:.1f}s", fill="black")
    eef = np.stack(df["observation.eef_state"])
    action = np.stack(df["action.wbc"])
    offset = rows * 258
    series = [("L wrist z", eef[:, 2], "red"), ("R wrist z", eef[:, 9], "blue"),
              ("L gripper target", action[:, -2], "red"),
              ("R gripper target", action[:, -1], "blue")]
    for j, (name, values, color) in enumerate(series):
        y = offset + 5 + j * 50
        lo, hi = (0, .5) if j < 2 else (0, 5.5)
        draw.text((3, y), name, fill=color)
        step = max(1, n // 1100)
        points = [(150 + i / max(1, n - 1) * 1110,
                   y + 40 - np.clip((values[i] - lo) / (hi - lo), 0, 1) * 38)
                  for i in range(0, n, step)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for t in np.arange(0, n / fps, 10):
            x = 150 + t * fps / max(1, n - 1) * 1110
            draw.line((x, y + 5, x, y + 40), fill="#cccccc")
            draw.text((x, y + 40), str(int(t)), fill="black")
    dest.mkdir(parents=True, exist_ok=True)
    sheet.save(dest / f"episode_{episode:06d}.jpg", quality=82)
    return {"episode": episode, "frames": n, "seconds": n / fps}


def numeric_stats(df):
    stats = {}
    for key in df:
        values = np.stack(df[key]).astype(np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {key}")
        stats[key] = {name: func(values, axis=0).tolist() for name, func in
                      (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))}
        stats[key]["count"] = [len(df)]
    return stats


def cut_video(job):
    source, clips, fps = job
    active = {}
    counts = Counter()
    starts = defaultdict(list)
    for clip in clips:
        starts[clip["start_frame"]].append(clip)
    last_end = max(c["end_frame"] for c in clips)
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        input_stream.thread_count = 1
        for i, frame in enumerate(input_container.decode(input_stream)):
            for clip in starts.get(i, []):
                target = Path(clip["video_path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(target), "w")
                stream = container.add_stream("libx264", rate=fps,
                                              options={"crf": "18", "preset": "fast"})
                stream.width, stream.height = frame.width, frame.height
                stream.pix_fmt = "yuv420p"
                stream.thread_count = 1
                active[clip["output_episode"]] = (container, stream, clip)
            for episode, (container, stream, clip) in list(active.items()):
                # Reformat makes an independent AVFrame for overlapping clips.
                output_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                output_frame.pts = counts[episode]
                output_frame.time_base = Fraction(1, fps)
                for packet in stream.encode(output_frame):
                    container.mux(packet)
                counts[episode] += 1
                if i + 1 == clip["end_frame"]:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                    del active[episode]
            if i + 1 >= last_end:
                break
    if active:
        for container, _, _ in active.values():
            container.close()
        raise ValueError(f"Source video shorter than requested intervals: {source}")
    for clip in clips:
        expected = clip["end_frame"] - clip["start_frame"]
        if counts[clip["output_episode"]] != expected:
            raise ValueError(f"Frame count mismatch: {source}, {clip}")
    return len(clips)


def segment_key(segment):
    return tuple(segment[k] for k in ("dataset", "episode", "start_frame", "end_frame", "task"))


def export(root, annotation_path, dest, workers, reuse=None):
    annotation = read_json(annotation_path)
    segments = annotation["segments"]
    if not segments:
        raise ValueError("No reviewed segments")
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "meta").mkdir()
    reference = root / segments[0]["dataset"]
    info = deepcopy(read_json(reference / "meta/info.json"))
    tasks = [name for name in PROMPTS if any(s["task"] == name for s in segments)]
    task_indices = {name: i for i, name in enumerate(tasks)}
    total, episode_meta, stats_meta, provenance = 0, [], [], []
    jobs = defaultdict(list)
    reused = {}
    if reuse is not None:
        reuse_info = read_json(reuse / "meta/info.json")
        if reuse_info["features"] != info["features"] or reuse_info["fps"] != info["fps"]:
            raise ValueError("Reuse dataset FPS/features differ")
        reused = {segment_key(s): s["output_episode"] for s in
                  read_json(reuse / "meta/segment_provenance.json")}
    cache_key, source_df = None, None
    for episode, segment in enumerate(segments):
        source = root / segment["dataset"]
        source_info = read_json(source / "meta/info.json")
        if source_info["fps"] != info["fps"] or source_info["features"] != info["features"]:
            raise ValueError("Source FPS/features differ")
        key = (segment["dataset"], segment["episode"])
        if key != cache_key:
            source_df = pd.read_parquet(episode_path(source, source_info, segment["episode"]))
            cache_key = key
        start, end = segment["start_frame"], segment["end_frame"]
        if not 0 <= start < end <= len(source_df):
            raise ValueError(f"Invalid interval: {segment}")
        df = source_df.iloc[start:end].copy().reset_index(drop=True)
        count = len(df)
        df["episode_index"] = episode
        df["frame_index"] = np.arange(count, dtype=np.int64)
        df["index"] = np.arange(total, total + count, dtype=np.int64)
        df["timestamp"] = np.arange(count, dtype=np.float32) / info["fps"]
        df["task_index"] = task_indices[segment["task"]]
        path = episode_path(dest, info, episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        total += count
        episode_meta.append({"episode_index": episode, "tasks": [PROMPTS[segment["task"]]], "length": count})
        stats_meta.append({"episode_index": episode, "stats": numeric_stats(df)})
        provenance.append({"output_episode": episode, **segment})
        for view in video_keys(info):
            if segment_key(segment) in reused:
                target = episode_path(dest, info, episode, view)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(episode_path(reuse, reuse_info, reused[segment_key(segment)], view), target)
                continue
            source_video = episode_path(source, source_info, segment["episode"], view)
            jobs[str(source_video)].append({**segment, "output_episode": episode,
                                           "video_path": str(episode_path(dest, info, episode, view))})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(cut_video, [(p, clips, info["fps"]) for p, clips in jobs.items()])):
            print(f"Encoded source view {i+1}/{len(jobs)}", flush=True)
    info.update(total_episodes=len(segments), total_frames=total, total_tasks=len(tasks),
                total_videos=len(segments) * len(video_keys(info)),
                total_chunks=(len(segments) + info["chunks_size"] - 1) // info["chunks_size"],
                splits={"train": f"0:{len(segments)}"})
    info.pop("discarded_episode_indices", None)
    info["processing"] = {"annotations": str(annotation_path.resolve()),
                          "intervals": "source frames [start, end)",
                          "video_alignment": "same frame interval for all three views and parquet",
                          "source_data_unchanged": True}
    write_json(dest / "meta/info.json", info)
    for name, rows in (("episodes", episode_meta), ("episodes_stats", stats_meta),
                       ("tasks", [{"task_index": task_indices[t], "task": PROMPTS[t]} for t in tasks])):
        (dest / "meta" / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    shutil.copy2(reference / "meta/modality.json", dest / "meta/modality.json")
    write_json(dest / "meta/segment_provenance.json", provenance)
    write_json(dest / "meta/primitive_prompts.json", {t: PROMPTS[t] for t in tasks})
    write_json(dest / "meta/annotations.json", annotation)
    summary = {"episodes": len(segments), "frames": total, "seconds": total / info["fps"],
               "task_counts": dict(Counter(s["task"] for s in segments))}
    write_json(dest / "meta/processing_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def start_images(root, dest):
    """Save the actual first decoded frame of every exported clip in all views."""
    info = read_json(root / "meta/info.json")
    provenance = read_json(root / "meta/segment_provenance.json")
    views = video_keys(info)
    grouped = defaultdict(list)
    records = []
    dest.mkdir(parents=True, exist_ok=False)
    for segment in provenance:
        episode, task = segment["output_episode"], segment["task"]
        directory = dest / task / f"episode_{episode:06d}"
        directory.mkdir(parents=True)
        triptych = Image.new("RGB", (960, 270), "white")
        draw = ImageDraw.Draw(triptych)
        images = {}
        for j, view in enumerate(views):
            with av.open(str(episode_path(root, info, episode, view))) as container:
                frame = next(container.decode(video=0))
                picture = frame.to_image()
            name = view.rsplit(".", 1)[-1]
            picture.save(directory / f"{name}.png")
            images[name] = str((directory / f"{name}.png").relative_to(dest))
            triptych.paste(picture.resize((320, 240)), (j * 320, 0))
            draw.text((j * 320 + 4, 241), f"ep{episode:06d} {name}", fill="black")
        draw.text((4, 256), f"{segment['dataset']} / ep{segment['episode']} / frame {segment['start_frame']}", fill="black")
        triptych.save(directory / "three_views.jpg", quality=92)
        grouped[task].append((episode, directory / "three_views.jpg"))
        records.append({**segment, "start_images": images,
                        "three_views": str((directory / "three_views.jpg").relative_to(dest))})
    overview = Image.new("RGB", (960, len(grouped) * 294), "white")
    d = ImageDraw.Draw(overview)
    galleries = {}
    for row, (task, clips) in enumerate(grouped.items()):
        pages = []
        for offset in range(0, len(clips), 6):
            page = Image.new("RGB", (960, min(6, len(clips) - offset) * 294), "white")
            pd = ImageDraw.Draw(page)
            for j, (_, path) in enumerate(clips[offset:offset+6]):
                pd.text((4, j * 294 + 3), task, fill="black")
                with Image.open(path) as picture:
                    page.paste(picture, (0, j * 294 + 24))
            path = dest / task / f"gallery_{offset//6:03d}.jpg"
            page.save(path, quality=88)
            pages.append(str(path.relative_to(dest)))
        galleries[task] = pages
        d.text((4, row * 294 + 3), f"{task} ({len(clips)} clips)", fill="black")
        with Image.open(clips[0][1]) as picture:
            overview.paste(picture, (0, row * 294 + 24))
    overview.save(dest / "overview.jpg", quality=90)
    write_json(dest / "index.json", {"dataset": str(root.resolve()), "views": views,
                                    "episodes": records, "galleries": galleries})
    print(f"Saved {len(records) * len(views)} lossless start images", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["review", "export", "starts"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse", type=Path, help="Copy unchanged videos from a previous export")
    args = parser.parse_args()
    if args.mode == "review":
        source = args.root / args.dataset
        episodes = [json.loads(s)["episode_index"] for s in (source / "meta/episodes.jsonl").read_text().splitlines()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(review_episode, [(source, args.output, ep, args.samples) for ep in episodes]))
        write_json(args.output / "episodes.json", results)
    elif args.mode == "starts":
        start_images(args.root, args.output)
    else:
        if args.annotations is None:
            parser.error("export requires --annotations")
        export(args.root, args.annotations, args.output, args.workers, args.reuse)


if __name__ == "__main__":
    main()
