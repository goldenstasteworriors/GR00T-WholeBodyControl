"""Read-only temporal ego review for missing initial target annotations.

Render source-frame neighborhoods with recorded bilateral wrist heights.
No inferred visibility, automatic boundary changes or robot connection.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter

from segment_lab_primitives import episode_path
from review_lab_primitive_starts import task_grip_mask


OFFSETS = [-60, -30, -15, -6, 0, 6, 15, 30, 45, 60, 90, 120]


def export_overviews(output):
    data = json.loads((output / "temporal_review.json").read_text())
    rows = [r for r in data["episodes"] if r["annotation_method"] != "unresolved"]
    folder = output / "overviews"
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    lines = ["# 其他首次未出框片段：前半秒／首帧／后半秒", "", "[返回入口](README.md)", ""]
    for offset in range(0, len(rows), 7):
        page = Image.new("RGB", (960, 7 * 290), "#14202b")
        draw = ImageDraw.Draw(page)
        for j, row in enumerate(rows[offset:offset + 7]):
            for k, delta in enumerate([-15, 0, 15]):
                record = min(row["frames"], key=lambda r: abs(r["offset_frames"] - delta))
                x, y = k * 320, j * 290
                pic = Image.open(output / record["image"]).convert("RGB")
                page.paste(pic.resize((320, 240)), (x, y + 45))
                draw.text((x + 2, y), f"{row['review_id']} {record['offset_frames']:+d}f", font=font, fill="#ffdc72" if delta == 0 else "white")
                z = record["wrist_z"]
                draw.text((x + 2, y + 22), f"L/R z {z[0]:.3f}/{z[1]:.3f}", font=font, fill="white")
        name = f"overviews/page_{offset//7:02d}.jpg"
        page.save(output / name, quality=94)
        lines += [f"![{offset+1}–{min(offset+7,len(rows))}]({name})", ""]
    (output / "other_overview.md").write_text("\n".join(lines) + "\n")


def read_frames(path, wanted, fps):
    images = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(int(Fraction(min(wanted), fps) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            index = round(float(frame.time) * fps)
            if index in wanted:
                images[index] = frame.to_image()
            if index >= max(wanted):
                break
    assert set(images) == set(wanted), (path, set(wanted) - set(images))
    return images


def run(args):
    initial = args.dataset / "initial_ego_boxes"
    audit = json.loads((initial / "method_audit_v1/manifest.json").read_text())
    included = {r["review_id"]: r for r in audit["episodes"] if r["method_group"] == "unresolved"
                or r["initial_pipeline_outcome"] in ("no_point_prompt_no_sam", "sam_rejected_point_constraints")}
    provenance = json.loads((args.dataset / "meta/segment_provenance.json").read_text())
    rows = [r for r in provenance if r["review_id"] in included]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("ego_sheets", "three_views", "frames"):
        (args.output / name).mkdir()
    cache, result = {}, []
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
    for row in rows:
        rid, start = row["review_id"], row["start_frame"]
        source = args.raw_root / row["dataset"]
        key = (row["dataset"], row["episode"])
        if key not in cache:
            info = json.loads((source / "meta/info.json").read_text())
            df = pd.read_parquet(episode_path(source, info, row["episode"]))
            eef = np.stack(df["observation.eef_state"])
            action = np.stack(df["action.wbc"])
            z = median_filter(eef[:, [2, 9]], size=(5, 1))
            reference = float(np.quantile(z.min(axis=1), .9))
            threshold = max(.18, reference - .015)
            cache[key] = (info, eef, action, z, reference, threshold)
        info, eef, action, z, reference, threshold = cache[key]
        fps = info["fps"]
        indexes = sorted({start + delta for delta in OFFSETS if 0 <= start + delta < len(z)})
        images = read_frames(episode_path(source, info, row["episode"], "observation.images.ego_view"), indexes, fps)
        folder = args.output / "frames" / rid
        folder.mkdir()
        sheet = Image.new("RGB", (1280, ((len(indexes) + 3) // 4) * 310 + 60), "#14202b")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 2), f"{rid} | EP{row['output_episode']:06d} | 原始起点 f{start}", font=font, fill="white")
        draw.text((8, 27), "原首帧黄色；高度仅作相对参考；起点前帧可能属于上一动作，需核对物体状态。", font=font, fill="#ffdc72")
        records = []
        grip = task_grip_mask(action, row["task"])
        for j, index in enumerate(indexes):
            delta = index - start
            images[index].save(folder / f"f{index:06d}.jpg", quality=95)
            x, y = j % 4 * 320, 60 + j // 4 * 310
            sheet.paste(images[index].resize((320, 240)), (x, y + 60))
            color = "#ffdc72" if delta == 0 else "white"
            draw.text((x + 3, y), f"{delta:+d}帧 / {delta/fps:+.2f}s | f{index}", font=font, fill=color)
            draw.text((x + 3, y + 20), f"腕高 L {z[index,0]:.3f} / R {z[index,1]:.3f}", font=font, fill=color)
            difference = z[index] - z[start]
            draw.text((x + 3, y + 40), f"对首帧 Δ {difference[0]:+.3f}/{difference[1]:+.3f}m", font=font, fill=color)
            records.append({"source_frame": index, "offset_frames": delta, "offset_seconds": delta / fps,
                            "wrist_z": z[index].tolist(), "wrist_z_change_from_start": difference.tolist(),
                            "upper_plateau_height_proxy": bool(z[index].min() >= threshold),
                            "task_gripper_proxy": bool(grip[index]),
                            "gripper_targets": action[index, -2:].tolist(),
                            "inside_existing_segment": bool(start <= index < row["end_frame"]),
                            "image": f"frames/{rid}/f{index:06d}.jpg"})
        sheet.save(args.output / "ego_sheets" / f"{rid}.jpg", quality=94)
        # Current unresolved cases get two wrist views at the same source frames.
        if included[rid]["method_group"] == "unresolved":
            selected = sorted({max(0, start - 15), start, min(len(z) - 1, start + 15), min(len(z) - 1, start + 60)})
            keys = ["observation.images.ego_view", "observation.images.left_wrist", "observation.images.right_wrist"]
            board = Image.new("RGB", (960, 4 * 275), "#14202b")
            bd = ImageDraw.Draw(board)
            for k, view in enumerate(keys):
                pics = read_frames(episode_path(source, info, row["episode"], view), selected, fps)
                for j, index in enumerate(selected):
                    board.paste(pics[index].resize((320, 240)), (k * 320, j * 275 + 35))
                    bd.text((k * 320 + 3, j * 275), f"{view.rsplit('.',1)[-1]} f{index} ({index-start:+d})", font=font, fill="white")
            board.save(args.output / "three_views" / f"{rid}.jpg", quality=94)
        item = {**row, "annotation_method": included[rid]["method_group"],
                "initial_pipeline_outcome": included[rid]["initial_pipeline_outcome"],
                "fps": fps, "reference_height_proxy": reference, "height_threshold_proxy": threshold,
                "note": "Height is recorded eef z, not camera or world height; thresholds are review aids, not proof of raised hands.",
                "frames": records}
        result.append(item)
        print(rid, flush=True)
    (args.output / "temporal_review.json").write_text(json.dumps({"dataset_unchanged": True, "episodes": result}, ensure_ascii=False, indent=2) + "\n")
    sections = defaultdict(list)
    for r in result:
        rid = r["review_id"]
        block = [f"## {rid}", "", f"当前类别：{r['annotation_method']}；首次流程：{r['initial_pipeline_outcome']}", "",
                 f"![前后帧与双腕高度](ego_sheets/{rid}.jpg)", ""]
        if r["annotation_method"] == "unresolved":
            block += [f"[同帧头部及双腕画面](three_views/{rid}.jpg)", ""]
        sections["unresolved" if r["annotation_method"] == "unresolved" else "other_initial_failures"].extend(block)
    for name, blocks in sections.items():
        (args.output / f"{name}.md").write_text("# 前后帧可见性检查\n\n[返回入口](README.md)\n\n" + "\n".join(blocks))
    (args.output / "README.md").write_text(
        "# 无框首帧的前后帧检查\n\n未修改训练视频、起点或标注。先检查目标身份及可见性，再结合双腕高度选择帧。\n\n"
        "- [当前仍留空的 25 段](unresolved.md)\n- [其他首次未出框的 56 段](other_initial_failures.md)\n\n"
        "前后帧均用原始视频编号；首帧前的画面不属于现有切分，不能直接套用成首帧框。\n")
    export_overviews(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
