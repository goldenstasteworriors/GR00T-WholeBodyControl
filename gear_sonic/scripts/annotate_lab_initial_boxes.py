"""Offline ego-only initial target annotations and numbered visual review.

Point prompts and reviewed boxes are expressed in the actual 640x480 first
frame. SAM2 proposes object masks; rack slots use explicit reviewed regions.
No robot connection, model API, video modification or start-boundary change.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAMERA = "observation.images.ego_view"
RULES = {
    "TAKE_TUBE": (0, "test_tube"),
    "TAKE_DROPPER_AND_ASPIRATE": (0, "dropper"),
    "RETURN_DROPPER": (1, "source_bottle"),
    "RETURN_TUBE_TO_RACK": (1, "destination_rack_slot"),
    "PLACE_TUBE_IN_BATH": (1, "water_bath_beaker"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def review(dataset, previous_path, corrections_path, output, ffmpeg="ffmpeg"):
    """Apply explicit first-frame corrections without reusing SAM scores as identity confidence."""
    data = json.loads(previous_path.read_text())
    corrections = json.loads(corrections_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    for name in ("overlays", "galleries", "videos"):
        (output / name).mkdir()
    grouped = defaultdict(list)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    labels = {"TAKE_TUBE": "取试管", "TAKE_DROPPER_AND_ASPIRATE": "拿滴管并取液",
              "RETURN_DROPPER": "放回滴管", "RETURN_TUBE_TO_RACK": "放回试管槽",
              "PLACE_TUBE_IN_BATH": "放入水浴烧杯"}
    for record in data["episodes"]:
        rid, task = record["review_id"], record["task"]
        if task not in RULES:
            assert not record["targets"]
            continue
        frame = dataset / "initial_ego_boxes/frames" / f"{rid}.png"
        assert hashlib.sha256(frame.read_bytes()).hexdigest() == record["reference_image_sha256"]
        if rid in corrections.get("not_visible", []):
            record["targets"] = []
            record["status"] = "unresolved_ego_visibility_or_identity"
        elif rid in corrections["boxes"]:
            role, entity = RULES[task]
            record["targets"] = [{"camera": CAMERA, "role": role, "entity": entity,
                                  "box": corrections["boxes"][rid],
                                  "source": "assistant_current_frame_region_proposal"}]
            record["status"] = "assistant_corrected_pending_user_review"
        else:
            record["status"] = ("sam_candidate_pending_user_review" if record["targets"]
                                else "unresolved_segmentation")
        record["needs_user_review"] = True
        record["user_approved"] = False
        record["annotation_note"] = corrections.get("notes", {}).get(rid, "")
        panel = Image.new("RGB", (640, 570), "#14202b")
        panel.paste(Image.open(frame).convert("RGB"), (0, 90))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 2), f"{rid} | EP{record['episode_index']:06d}", font=font, fill="white")
        draw.text((8, 25), f"{labels[task]} · ego 首帧 · 待用户核对", font=font, fill="#ffdc72")
        note = record["annotation_note"] or ("没有可确认的首帧框" if not record["targets"] else "框坐标为原始 640×480 图像坐标")
        draw.text((8, 50), note[:43], font=font, fill="#ffdc72")
        for mark in record["targets"]:
            mark.pop("confidence", None)
            x1, y1, x2, y2 = mark["box"]
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480, (rid, mark)
            color = "#45ff86" if mark["role"] == 0 else "#ffcf40"
            draw.rectangle((x1, y1 + 90, x2 - 1, y2 + 89), outline=color, width=3)
        panel.save(output / "overlays" / f"{rid}.png")
        grouped[task].append((rid, panel))
    galleries, videos = {}, {}
    lines = ["# Ego 初始框检查", "", "本目录是待核对标注；尚未获得用户确认。全部框位于原始 ego 首帧，未移动切分起点。",
             "", "绿色：操作物。黄色：接收物／目标槽位。加液及行走按用户要求不标框。",
             "", "没有可靠首帧目标的片段留空，并列在末尾。检查时请提供类别和编号。", ""]
    for task, tiles in grouped.items():
        galleries[task] = []
        lines += [f"## {labels[task]}（{len(tiles)} 段）", "", f"[逐段视频，每段首帧停留 1 秒](videos/{task}.mp4)", ""]
        for offset in range(0, len(tiles), 12):
            page = Image.new("RGB", (1280, 4 * 380), "#14202b")
            for j, (_, tile) in enumerate(tiles[offset:offset + 12]):
                page.paste(tile.resize((426, 380)), (j % 3 * 426, j // 3 * 380))
            name = f"galleries/{task}_{offset//12:02d}.jpg"
            page.save(output / name, quality=95)
            galleries[task].append(name)
            lines += [f"![{task} {offset+1}–{min(offset+12,len(tiles))}]({name})", ""]
        playlist = output / "videos" / f"{task}.ffconcat"
        playlist.write_text("ffconcat version 1.0\n" + "".join(
            f"file '../overlays/{rid}.png'\nduration 1\n" for rid, _ in tiles)
            + f"file '../overlays/{tiles[-1][0]}.png'\n")
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-safe", "0", "-i", str(playlist),
                        "-t", str(len(tiles)), "-r", "30", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(output / "videos" / f"{task}.mp4")], check=True)
        videos[task] = f"videos/{task}.mp4"
    unresolved = [r for r in data["episodes"] if r["task"] in RULES and not r["targets"]]
    lines += ["## 无可靠首帧框的片段", ""]
    lines += [f"- [{r['review_id']}](overlays/{r['review_id']}.png)：{r['annotation_note'] or r['status']}" for r in unresolved]
    data.update(galleries=galleries, videos=videos, review_state="pending_user_review",
                status_counts=dict(Counter(r["status"] for r in data["episodes"])),
                previous_annotations=str(previous_path.resolve()),
                corrections_sha256=hashlib.sha256(corrections_path.read_bytes()).hexdigest())
    write_json(output / "annotations.json", data)
    write_json(output / "corrections_used.json", corrections)
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(data["status_counts"], ensure_ascii=False))


def run(dataset, prompts_path, output, sam_source=None, checkpoint=None, device="cpu"):
    rows = json.loads((dataset / "meta/segment_provenance.json").read_text())
    prompt_data = json.loads(prompts_path.read_text())
    prompts = {p["review_id"]: p for p in prompt_data["episodes"]}
    expected = {row["review_id"] for row in rows if row["task"] in RULES}
    if set(prompts) != expected:
        raise ValueError(f"Prompt coverage mismatch: missing={expected-set(prompts)}, extra={set(prompts)-expected}")
    predictor = None
    if any(p.get("positive_points") for p in prompts.values()):
        import sys
        import torch

        sys.path.insert(0, str(sam_source))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        torch.set_num_threads(4)
        predictor = SAM2ImagePredictor(build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml", str(checkpoint), device=device,
            apply_postprocessing=False))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("overlays", "masks", "galleries"):
        (output / name).mkdir()
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 17)
    records, grouped = [], defaultdict(list)
    frame_root = dataset / "initial_ego_boxes/frames"
    for row in rows:
        rid, task = row["review_id"], row["task"]
        record = {"episode_index": row["output_episode"], "review_id": rid, "task": task,
                  "frame_index": 0, "source_frame_index": row["start_frame"],
                  "source_dataset": row["dataset"], "source_episode": row["episode"],
                  "camera": CAMERA, "targets": []}
        if task not in RULES:
            record["status"] = "not_required_by_user"
            records.append(record)
            continue
        prompt = prompts[rid]
        path = frame_root / f"{rid}.png"
        image = Image.open(path).convert("RGB")
        record["image_size"] = list(image.size)
        record["reference_image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["prompt"] = prompt
        record["status"] = prompt["status"]
        record["needs_user_review"] = True
        candidates = []
        if prompt.get("positive_points"):
            import torch

            positive, negative = prompt["positive_points"], prompt.get("negative_points", [])
            coords = np.asarray(positive + negative, np.float32)
            labels = np.asarray([1] * len(positive) + [0] * len(negative), np.int32)
            with torch.inference_mode():
                predictor.set_image(np.array(image))
                masks, scores, _ = predictor.predict(
                    point_coords=coords, point_labels=labels,
                    box=np.asarray(prompt["seed_box"], np.float32) if prompt.get("seed_box") else None,
                    multimask_output=True)
            for i, mask in enumerate(masks):
                ys, xs = np.where(mask)
                if not len(xs):
                    continue
                valid = (all(mask[y, x] for x, y in positive)
                         and all(not mask[y, x] for x, y in negative))
                candidates.append({"candidate": i, "score": float(scores[i]),
                                   "satisfies_points": valid,
                                   "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
            record["sam_candidates"] = candidates
        box = prompt.get("reviewed_box")
        method = "current_frame_reviewed_region"
        if box is None and prompt.get("region_box") is not None:
            box = prompt["region_box"]
            method = "registered_region_candidate"
        if box is None and candidates:
            valid = [c for c in candidates if c["satisfies_points"]]
            if valid:
                best = max(valid, key=lambda c: c["score"])
                box = best["box"]
                method = "sam2_single_image_candidate"
                Image.fromarray(masks[best["candidate"]].astype(np.uint8) * 255).save(output / "masks" / f"{rid}.png")
        if box is not None:
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError(f"Invalid original-image box: {rid}: {box}")
            role, entity = RULES[task]
            record["targets"] = [{"camera": CAMERA, "role": role, "entity": entity, "box": box,
                                  "confidence": prompt.get("confidence", .5), "source": method}]
        panel = Image.new("RGB", (640, 545), "#14202b")
        panel.paste(image, (0, 65))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 3), f"{rid} | EP{row['output_episode']:06d} | ego frame 0", font=font, fill="white")
        draw.text((8, 26), f"{RULES[task][1]} | {record['status']}", font=font, fill="#ffdc72")
        for mark in record["targets"]:
            x1, y1, x2, y2 = mark["box"]
            color = "#45ff86" if mark["role"] == 0 else "#ffcf40"
            draw.rectangle((x1, y1 + 65, x2 - 1, y2 + 64), outline=color, width=3)
            draw.text((min(x1, 420), max(65, y1 + 43)), mark["entity"], fill=color)
        panel.save(output / "overlays" / f"{rid}.png")
        grouped[task].append((rid, panel))
        records.append(record)
        print(f"Annotated {rid}: {record['status']}, marks={len(record['targets'])}", flush=True)
    galleries = {}
    for task, tiles in grouped.items():
        galleries[task] = []
        for offset in range(0, len(tiles), 12):
            page = Image.new("RGB", (1280, 4 * 364), "#14202b")
            for j, (_, tile) in enumerate(tiles[offset:offset + 12]):
                page.paste(tile.resize((426, 364)), (j % 3 * 426, j // 3 * 364))
            name = f"{task}_{offset//12:02d}.jpg"
            page.save(output / "galleries" / name, quality=95)
            galleries[task].append("galleries/" + name)
    write_json(output / "annotations.json", {"dataset": str(dataset.resolve()), "camera": CAMERA,
               "box_format": "original-pixel xyxy, x2/y2 exclusive", "frame_index": 0,
               "rules": RULES, "episodes": records, "galleries": galleries,
               "status_counts": dict(Counter(r["status"] for r in records)),
               "prompt_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--review-from", type=Path)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.review_from:
        if not args.corrections:
            parser.error("--review-from requires --corrections")
        review(args.dataset, args.review_from, args.corrections, args.output, args.ffmpeg)
    else:
        if not args.prompts:
            parser.error("--prompts is required for SAM annotation")
        run(args.dataset, args.prompts, args.output, args.sam_source, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
