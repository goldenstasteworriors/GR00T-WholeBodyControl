"""Render manually reviewed, complete lab processes with persistent review IDs.

Review IDs are retained even when earlier IDs are excluded. Source and training
videos are not modified. The combined movie plays every other frame at 30 FPS.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from segment_lab_primitives import episode_path, read_json, video_keys, write_json


def writer(path, stack):
    container = stack.enter_context(av.open(str(path), "w"))
    stream = container.add_stream("libx264", rate=30, options={"crf": "20", "preset": "fast"})
    stream.width, stream.height, stream.pix_fmt = 960, 320, "yuv420p"
    stream.thread_count = 1
    return container, stream


def encode(target, picture, index):
    container, stream = target
    frame = av.VideoFrame.from_image(picture)
    frame.pts, frame.time_base = index, Fraction(1, 30)
    for packet in stream.encode(frame):
        container.mux(packet)


def finish(target):
    container, stream = target
    for packet in stream.encode():
        container.mux(packet)


def interval_frames(container, start, end):
    stream = container.streams.video[0]
    stream.thread_count = 1
    container.seek(int(Fraction(start, 30) / stream.time_base), stream=stream, backward=True)
    expected = start
    for frame in container.decode(stream):
        index = round(float(frame.time) * 30)
        if index < start:
            continue
        if index != expected:
            raise ValueError(f"Expected source frame {expected}, got {index}")
        yield frame.reformat(width=320, height=240, format="rgb24").to_image()
        expected += 1
        if expected == end:
            break
    if expected != end:
        raise ValueError("Source video ended before reviewed interval")


def render_segment(job):
    root, output, segment = job
    source = root / segment["dataset"]
    info = read_json(source / "meta/info.json")
    start, end = segment["start_frame"], segment["end_frame"]
    path = output / f"{segment['review_id']}.mp4"
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    base = Image.new("RGB", (960, 320), "#131e29")
    draw = ImageDraw.Draw(base)
    draw.text((8, 2), f"{segment['review_id']}   加液完整过程   原始 EP{segment['episode']:06d}", font=font, fill="white")
    draw.text((8, 25), f"{segment['dataset']}  |  {start/30:.3f}–{end/30:.3f}s", font=small, fill="white")
    for j, name in enumerate(["头部", "左腕", "右腕"]):
        draw.text((j * 320 + 140, 303), name, font=small, fill="white")
    with ExitStack() as stack:
        inputs = [stack.enter_context(av.open(str(episode_path(source, info, segment["episode"], view)))) for view in video_keys(info)]
        target = writer(path, stack)
        count = 0
        for frames in zip(*(interval_frames(c, start, end) for c in inputs)):
            picture = base.copy()
            for j, image in enumerate(frames):
                picture.paste(image, (j * 320, 63))
            d = ImageDraw.Draw(picture)
            d.text((8, 45), f"source frame {start+count}   source {(start+count)/30:.3f}s   clip {count/30:.3f}s", fill="white")
            encode(target, picture, count)
            count += 1
        if count != end - start:
            raise ValueError(f"Short render: {segment['review_id']}")
        finish(target)
    print(f"Rendered full process {segment['review_id']}", flush=True)
    return {"review_id": segment["review_id"], "video": str(path), "frames": count,
            "source_start_frame": start, "source_end_frame": end}


def combine(rows, output):
    full_count, starts_count, boundaries_count, index = 0, 0, 0, []
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
    with ExitStack() as stack:
        full = writer(output / "all_complete_processes_2x.mp4", stack)
        starts = writer(output / "all_starts_1s.mp4", stack)
        boundaries = writer(output / "all_starts_and_ends_1s.mp4", stack)
        for row in rows:
            offset = full_count
            with av.open(row["video"]) as container:
                container.streams.video[0].thread_count = 1
                count = 0
                for i, frame in enumerate(container.decode(video=0)):
                    if i % 2 == 0:
                        encode(full, frame.to_image(), full_count)
                        full_count += 1
                    if i < 30:
                        picture = frame.to_image()
                        ImageDraw.Draw(picture).text((750, 43), "开始 1 秒", font=font, fill="#90ee90")
                        encode(starts, picture, starts_count)
                        starts_count += 1
                        encode(boundaries, picture, boundaries_count)
                        boundaries_count += 1
                    if i >= row["frames"] - 30:
                        picture = frame.to_image()
                        ImageDraw.Draw(picture).text((750, 43), "收尾 1 秒", font=font, fill="#ffdf80")
                        encode(boundaries, picture, boundaries_count)
                        boundaries_count += 1
                    count += 1
                if count != row["frames"]:
                    raise ValueError(f"Short combined input: {row['review_id']}")
            index.append({**row, "combined_2x_start_seconds": offset / 30,
                          "combined_2x_end_seconds": full_count / 30})
        for target in [full, starts, boundaries]:
            finish(target)
    write_json(output / "process_video_index.json", {"clips": index,
               "combined_2x_frames": full_count, "starts_frames": starts_count,
               "start_end_frames": boundaries_count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    annotation = read_json(args.annotations)
    clips = args.output / "clips"
    clips.mkdir()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(render_segment, [(args.root, clips, segment) for segment in annotation["segments"]]))
    combine(rows, args.output)


if __name__ == "__main__":
    main()
