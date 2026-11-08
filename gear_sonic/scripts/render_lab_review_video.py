"""Render complete, numbered three-view primitive reviews without changing data."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
from fractions import Fraction
from functools import cache
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from review_lab_primitive_starts import TITLES
from segment_lab_primitives import episode_path, read_json, video_keys, write_json


@cache
def fonts():
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(path, 17), ImageFont.truetype(path, 14)


def sampled_frames(path, count, speed):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        decoded = 0
        for i, frame in enumerate(container.decode(stream)):
            if round(float(frame.time) * 30) != i:
                raise ValueError(f"Unexpected source timestamp: {path}, frame {i}")
            if i % speed == 0:
                yield frame.reformat(width=320, height=240, format="rgb24").to_image()
            decoded += 1
        if decoded != count:
            raise ValueError(f"Expected {count} frames, decoded {decoded}: {path}")


def render_clip(job):
    dataset, output, info, row, speed = job
    frames = row["end_frame"] - row["start_frame"]
    target = output / "clips" / f"{row['review_id']}.mp4"
    font, small = fonts()
    base = Image.new("RGB", (960, 320), "#131e29")
    draw = ImageDraw.Draw(base)
    draw.text((8, 1), f"{row['review_id']}   {TITLES[row['task']]}   {speed}倍速", font=font, fill="white")
    draw.text((8, 24), f"{row['dataset']}  原始 EP{row['episode']:06d}  |  {row['start_frame']/30:.3f}–{row['end_frame']/30:.3f}s", font=small, fill="white")
    for j, label in enumerate(("头部", "左腕", "右腕")):
        draw.text((j * 320 + 140, 303), label, font=small, fill="white")
    views = video_keys(info)
    expected_views = [f"observation.images.{v}" for v in ("ego_view", "left_wrist", "right_wrist")]
    if views != expected_views:
        raise ValueError(f"Unexpected camera order: {views}")
    sources = [sampled_frames(episode_path(dataset, info, row["output_episode"], v), frames, speed) for v in views]
    count = 0
    with av.open(str(target), "w") as container:
        stream = container.add_stream("libx264", rate=30, options={"crf": "20", "preset": "fast"})
        stream.width, stream.height, stream.pix_fmt = 960, 320, "yuv420p"
        stream.thread_count = 1
        for images in zip(*sources, strict=True):
            picture = base.copy()
            for j, image in enumerate(images):
                picture.paste(image, (j * 320, 63))
            source_frame = row["start_frame"] + count * speed
            ImageDraw.Draw(picture).text((8, 44), f"source frame {source_frame} / {source_frame/30:.3f}s  |  cut EP{row['output_episode']:06d}", font=small, fill="#c8e5ff")
            frame = av.VideoFrame.from_image(picture)
            frame.pts, frame.time_base = count, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    if count != (frames + speed - 1) // speed:
        raise ValueError(f"Wrong output length: {row['review_id']}")
    print(f"Rendered {row['review_id']} ({count} frames)", flush=True)
    return {**row, "speed": speed, "video": str(target.resolve()), "video_frames": count}


def concat(rows, target):
    # Offset integer stream timestamps directly. Summing container durations
    # rounded to microseconds would slowly drift from the exact 30fps grid.
    offset, chapters, stream = 0, [], None
    with av.open(str(target), "w") as output:
        for i, row in enumerate(rows):
            chapters.append({"id": i, "start": offset, "end": offset + row["video_frames"],
                             "time_base": Fraction(1, 30), "metadata": {
                                 "title": f"{row['review_id']} / {row['dataset']} / EP{row['episode']:06d}"}})
            with av.open(row["video"]) as source:
                input_stream = source.streams.video[0]
                if stream is None:
                    stream = output.add_stream_from_template(input_stream)
                for packet in source.demux(input_stream):
                    if packet.dts is None:
                        continue
                    shift = Fraction(offset, 30) / packet.time_base
                    if shift.denominator != 1:
                        raise ValueError("Clip timestamps cannot represent exact frame offsets")
                    packet.pts += int(shift)
                    packet.dts += int(shift)
                    packet.stream = stream
                    output.mux(packet)
            offset += row["video_frames"]
        output.set_chapters(chapters)
    return offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boundary-review", type=Path)
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--speed", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-clips", action="store_true", help="Reuse completed clips from the same source intervals")
    args = parser.parse_args()
    info = read_json(args.dataset / "meta/info.json")
    if info["fps"] != 30:
        raise ValueError("This review renderer expects the 30fps collection schema")
    provenance = read_json(args.dataset / "meta/segment_provenance.json")
    audit = {}
    if args.boundary_review:
        audit = {r["output_episode"]: r for r in read_json(args.boundary_review)["review_records"]}
    counters, rows = Counter(), []
    for segment in provenance:
        task = segment["task"]
        counters[task] += 1
        if task in args.exclude_task:
            continue
        review_id = segment.get("review_id", f"{task}-{counters[task]:03d}")
        if audit:
            record = audit[segment["output_episode"]]
            for key in ("dataset", "episode", "start_frame", "end_frame", "task"):
                if record[key] != segment[key]:
                    raise ValueError(f"Review ID map mismatch: {key}, {segment}")
            review_id = record["review_id"]
        rows.append({**segment, "review_id": review_id})
    task_order = {task: i for i, task in enumerate(TITLES)}
    rows.sort(key=lambda r: (task_order[r["task"]], r["review_id"]))
    args.output.mkdir(parents=True, exist_ok=args.reuse_clips)
    (args.output / "clips").mkdir(exist_ok=args.reuse_clips)
    if args.reuse_clips:
        rendered = []
        for row in rows:
            path = args.output / "clips" / f"{row['review_id']}.mp4"
            count = (row["end_frame"] - row["start_frame"] + args.speed - 1) // args.speed
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if stream.frames != count or (stream.width, stream.height) != (960, 320):
                    raise ValueError(f"Invalid cached clip: {path}")
            rendered.append({**row, "speed": args.speed, "video": str(path.resolve()), "video_frames": count})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rendered = list(pool.map(render_clip, [(args.dataset, args.output, info, row, args.speed) for row in rows]))
    class_videos, offset = {}, 0
    for task in TITLES:
        selected = [r for r in rendered if r["task"] == task]
        if not selected:
            continue
        name = f"{task}_{args.speed}x.mp4"
        class_videos[task] = {"video": name, "segments": len(selected), "frames": concat(selected, args.output / name)}
        class_offset = 0
        for row in selected:
            row["combined_start_seconds"] = offset / 30
            row["class_start_seconds"] = class_offset / 30
            offset += row["video_frames"]
            class_offset += row["video_frames"]
            row["combined_end_seconds"] = offset / 30
    combined = args.output / f"all_primitives_{args.speed}x.mp4"
    if concat(rendered, combined) != offset:
        raise ValueError("Combined output length mismatch")
    write_json(args.output / "video_index.json", {
        "dataset": str(args.dataset.resolve()), "boundary_policy": "Existing exported intervals; no unapproved candidate boundaries applied.",
        "speed": args.speed, "fps": 30, "combined": combined.name,
        "combined_frames": offset, "segments": rendered, "class_videos": class_videos,
    })
    columns = ["review_id", "task", "dataset", "episode", "output_episode", "start_frame", "end_frame",
               "combined_start_seconds", "combined_end_seconds", "class_start_seconds"]
    with (args.output / "编号时间索引.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rendered)
    print(f"Complete: {len(rendered)} segments, {offset/30:.2f}s", flush=True)


if __name__ == "__main__":
    main()
