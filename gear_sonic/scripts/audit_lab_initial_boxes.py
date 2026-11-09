"""Classify existing ego annotations and replay rejected SAM2 prompts locally.

This audit preserves the exact original points, box prompts and first frames.
Replayed candidate masks are diagnostics, never replacement training labels.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GROUPS = {
    "sam2": "SAM2 出框（待核对对象）",
    "manual": "人工矩形框（待核对）",
    "unresolved": "当前无可靠框",
    "not_required": "用户要求不标框",
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export_sam_success_review(dataset, output):
    """Compare every originally accepted SAM box with the current annotation."""
    root = dataset / "initial_ego_boxes"
    original = json.loads((root / "sam_review_v1/annotations.json").read_text())
    current = {r["review_id"]: r for r in json.loads((root / "review_v3/annotations.json").read_text())["episodes"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 12)
    sections, records = [], []
    for row in sorted(original["episodes"], key=lambda r: (r["task"], r["review_id"])):
        if not row["targets"] or row["targets"][0]["source"] != "sam2_single_image_candidate":
            continue
        rid, prompt = row["review_id"], row["prompt"]
        selected = max((c for c in row["sam_candidates"] if c["satisfies_points"]), key=lambda c: c["score"])
        assert selected["box"] == row["targets"][0]["box"]
        now = current[rid]
        replaced = bool(now["targets"] and now["targets"][0]["source"] == "assistant_current_frame_region_proposal")
        frame = root / "frames" / f"{rid}.png"
        assert hashlib.sha256(frame.read_bytes()).hexdigest() == row["reference_image_sha256"]
        image = Image.open(frame).convert("RGB")
        panel = Image.new("RGB", (1280, 650), "#14202b")
        panel.paste(image, (0, 110))
        panel.paste(image, (640, 110))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 3), f"{rid} | EP{row['episode_index']:06d} | ego 首帧", font=font, fill="white")
        draw.text((8, 27), f"左：原 SAM 采用框（青）及原点 | 候选 {selected['candidate']} | 分数 {selected['score']:.3f}", font=font, fill="#47dcff")
        draw.text((648, 27), "右：后来人工指定的框（黄）" if replaced else "右：当前保留的 SAM 框（青）", font=font, fill="#ffcf40" if replaced else "#47dcff")
        draw.text((8, 53), "绿 P 为正点；红 N 为负点；紫框是原输入框。点由锚帧配准传播。", font=font, fill="white")
        draw.text((8, 78), "本页仅供对照：人工覆盖不代表原 SAM 框一定错误，尚未修改当前标注。", font=font, fill="#ffdc72")
        for box, dx, color in ((prompt.get("seed_box"), 0, "#d06dff"),
                               (selected["box"], 0, "#47dcff"),
                               (now["targets"][0]["box"] if now["targets"] else None, 640, "#ffcf40" if replaced else "#47dcff")):
            if box:
                x1, y1, x2, y2 = box
                draw.rectangle((x1 + dx, y1 + 110, x2 - 1 + dx, y2 + 109), outline=color, width=2)
        for prefix, field, color in (("P", "positive_points", "#37ff74"), ("N", "negative_points", "#ff4242")):
            for i, (x, y) in enumerate(prompt.get(field, []), 1):
                draw.ellipse((x - 4, y + 106, x + 4, y + 114), fill=color, outline="black", width=1)
                draw.text((min(x + 6, 612), max(110, y + 96)), f"{prefix}{i}", font=small, fill=color, stroke_width=1, stroke_fill="black")
        draw.text((8, 597), f"原正点：{prompt.get('positive_points', [])}", font=font, fill="#37ff74")
        draw.text((8, 622), f"原负点：{prompt.get('negative_points', [])}", font=font, fill="#ff7575")
        draw.text((648, 597), f"原 SAM 框：{selected['box']}", font=font, fill="#47dcff")
        draw.text((648, 622), f"当前框：{now['targets'][0]['box'] if now['targets'] else None}", font=font, fill="#ffcf40" if replaced else "#47dcff")
        panel.save(output / "images" / f"{rid}.jpg", quality=95)
        mask = root / "sam_review_v1/masks" / f"{rid}.png"
        assert mask.exists()
        record = {"review_id": rid, "task": row["task"], "episode_index": row["episode_index"],
                  "frame_sha256": row["reference_image_sha256"], "positive_points": prompt["positive_points"],
                  "negative_points": prompt.get("negative_points", []), "sam_input_box": prompt.get("seed_box"),
                  "selected_sam_candidate": selected, "all_sam_candidates": row["sam_candidates"],
                  "replaced_by_manual_box": replaced, "current_targets": now["targets"]}
        records.append(record)
        coords = {k: record[k] for k in ("positive_points", "negative_points", "sam_input_box")}
        coords.update(sam_output_box=selected["box"], current_box=now["targets"][0]["box"] if now["targets"] else None)
        section = [f"## {rid} / EP{row['episode_index']:06d}", "",
                   "**后来被人工框覆盖**" if replaced else "当前保留原 SAM 框", "",
                   f"![{rid} 原 SAM 框、原点和当前框](images/{rid}.jpg)", "",
                   f"[原始首帧](../../frames/{rid}.png) · [原 SAM 采用掩码](../../sam_review_v1/masks/{rid}.png)", "",
                   "```json", json.dumps(coords, ensure_ascii=False), "```", "",
                   "| SAM 候选 | 分数 | 满足点约束 | 输出框 | 当时采用 |",
                   "|---|---:|---|---|---|"]
        for candidate in row["sam_candidates"]:
            section.append(f"| {candidate['candidate']} | {candidate['score']:.4f} | {'是' if candidate['satisfies_points'] else '否'} | {candidate['box']} | {'是' if candidate['candidate'] == selected['candidate'] else ''} |")
        section.append("")
        sections.append((record, "\n".join(section)))
    counts = dict(Counter("manual_override" if r["replaced_by_manual_box"] else "sam_retained" for r in records))
    intro = ["# 原 SAM2 出框与提示点对照", "", f"共 {len(records)} 段。{counts.get('manual_override', 0)} 段后来被人工框覆盖，{counts.get('sam_retained', 0)} 段保留 SAM 框。", "",
             "左图：最初被程序采用的 SAM 框＋原正负点＋输入框。右图：当前框。全部使用同一 ego 首帧。",
             "", "人工覆盖不代表原 SAM 一定错误；本次只补齐对照，尚未改回任何标注。", "",
             "[仅查看后来被人工框覆盖的片段](manual_overrides.md)", ""]
    tasks = sorted({r["task"] for r in records})
    intro += [f"- [{task}](task_{task}.md)" for task in tasks] + [""]
    (output / "README.md").write_text("\n".join(intro) + "\n" + "\n".join(s for _, s in sections))
    (output / "manual_overrides.md").write_text("# 原 SAM 出框后被人工覆盖的片段\n\n[全部 225 段](README.md)\n\n" + "\n".join(s for r, s in sections if r["replaced_by_manual_box"]))
    for task in tasks:
        (output / f"task_{task}.md").write_text(f"# {task}\n\n[返回全部](README.md)\n\n" + "\n".join(s for r, s in sections if r["task"] == task))
    save_json(output / "manifest.json", {"counts": counts, "episodes": records})
    print(json.dumps({"total": len(records), **counts}), flush=True)


def export_method_videos(output, ffmpeg):
    for group in ("sam2", "manual", "unresolved"):
        folder = output / "by_method" / group
        rows = json.loads((folder / "manifest.json").read_text())
        playlist = folder / "review.ffconcat"
        playlist.write_text("ffconcat version 1.0\n" + "".join(
            f"file 'overlays/{r['review_id']}.png'\nduration 1\n" for r in rows)
            + f"file 'overlays/{rows[-1]['review_id']}.png'\n")
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-safe", "0", "-i", str(playlist),
                        "-vf", "fps=30", "-frames:v", str(len(rows) * 30), "-c:v", "libx264",
                        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(folder / "review.mp4")], check=True)
        readme = folder / "README.md"
        readme.write_text(readme.read_text() + "\n[本类编号视频，每段首帧一秒](review.mp4)\n")


def paint(image, prompt, title, subtitle, mask=None, final_box=None):
    panel = Image.new("RGB", (640, 570), "#14202b")
    rgb = np.asarray(image).copy()
    if mask is not None:
        rgb[mask] = (.55 * rgb[mask] + .45 * np.array([0, 190, 255])).astype(np.uint8)
    panel.paste(Image.fromarray(rgb), (0, 90))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(FONT, 16)
    draw.text((8, 2), title, font=font, fill="white")
    draw.text((8, 25), subtitle, font=font, fill="#ffdc72")
    draw.text((8, 50), "绿 P：正点　红 N：负点　紫：原输入框　黄：当前人工框", font=font, fill="white")
    for box, color in ((prompt.get("seed_box"), "#d06dff"), (final_box, "#ffcf40")):
        if box:
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1 + 90, x2 - 1, y2 + 89), outline=color, width=2)
    for prefix, field, color in (("P", "positive_points", "#37ff74"), ("N", "negative_points", "#ff4242")):
        for i, (x, y) in enumerate(prompt.get(field, []), 1):
            y += 90
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="black", width=2)
            tx, ty = min(x + 8, 610), max(90, y - 18)
            draw.rectangle((tx, ty, tx + 25, ty + 18), fill="black")
            draw.text((tx, ty), f"{prefix}{i}", fill=color)
    return panel


def run(args):
    root = args.dataset / "initial_ego_boxes"
    original_path = root / "sam_review_v1/annotations.json"
    current_path = root / "review_v3/annotations.json"
    original = json.loads(original_path.read_text())
    current = json.loads(current_path.read_text())
    old = {r["review_id"]: r for r in original["episodes"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for group in GROUPS:
        (args.output / "by_method" / group).mkdir(parents=True)
    diagnostic_root = args.output / "diagnostics"
    diagnostic_root.mkdir()
    failures = [r for r in old.values() if r.get("prompt", {}).get("positive_points")
                and not any(c["satisfies_points"] for c in r.get("sam_candidates", []))]
    import torch

    sys.path.insert(0, str(args.sam_source))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(4)
    predictor = SAM2ImagePredictor(build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml", str(args.checkpoint), device=args.device,
        apply_postprocessing=False))
    failed_ids = {r["review_id"] for r in failures}
    grouped, outcomes, audit = defaultdict(list), defaultdict(list), []
    for record in current["episodes"]:
        rid = record["review_id"]
        previous = old[rid]
        prompt = previous.get("prompt", {})
        targets = record["targets"]
        if record["status"] == "not_required_by_user":
            group = "not_required"
        elif not targets:
            group = "unresolved"
        elif targets[0]["source"] == "sam2_single_image_candidate":
            group = "sam2"
        else:
            group = "manual"
        has_points = bool(prompt.get("positive_points"))
        if has_points:
            outcome = "sam_rejected_point_constraints" if rid in failed_ids else "sam_produced_box"
        elif previous.get("targets"):
            outcome = "region_prompt_no_sam"
        else:
            outcome = "no_point_prompt_no_sam"
        if group == "not_required":
            outcome = "not_required"
        item = {"review_id": rid, "episode_index": record["episode_index"], "task": record["task"],
                "method_group": group, "initial_pipeline_outcome": outcome,
                "prompt": prompt, "original_sam_candidates": previous.get("sam_candidates", []),
                "current_targets": targets, "current_status": record["status"],
                "needs_user_review": group != "not_required"}
        grouped[group].append(item)
        outcomes[outcome].append(item)
        audit.append(item)
        if group == "not_required":
            continue
        frame = root / "frames" / f"{rid}.png"
        assert hashlib.sha256(frame.read_bytes()).hexdigest() == previous["reference_image_sha256"]
        item["frame_sha256"] = previous["reference_image_sha256"]
        image = Image.open(frame).convert("RGB")
        dest = args.output / "by_method" / group
        (dest / "overlays").mkdir(exist_ok=True)
        shutil.copyfile(root / "review_v3/overlays" / f"{rid}.png", dest / "overlays" / f"{rid}.png")
        folder = diagnostic_root / rid
        folder.mkdir()
        reason = {"sam_produced_box": "SAM 曾出框；不代表对象选择正确",
                  "sam_rejected_point_constraints": "SAM 有候选，但没有一个满足全部正负点",
                  "region_prompt_no_sam": "原本使用区域框，没有给点或调用 SAM",
                  "no_point_prompt_no_sam": "提示生成阶段未通过；没有给点或调用 SAM"}[outcome]
        final_box = targets[0]["box"] if targets and group == "manual" else None
        point_panel = paint(image, prompt, rid, reason, final_box=final_box)
        point_panel.save(folder / "original_points.png")
        item["points_visualization"] = f"diagnostics/{rid}/original_points.png"
        item["explanation"] = reason
        if rid in failed_ids:
            positive, negative = prompt["positive_points"], prompt.get("negative_points", [])
            with torch.inference_mode():
                predictor.set_image(np.array(image))
                masks, scores, _ = predictor.predict(
                    point_coords=np.asarray(positive + negative, np.float32),
                    point_labels=np.asarray([1] * len(positive) + [0] * len(negative), np.int32),
                    box=np.asarray(prompt["seed_box"], np.float32) if prompt.get("seed_box") else None,
                    multimask_output=True)
            board = Image.new("RGB", (1280, 1140), "#14202b")
            board.paste(point_panel, (0, 0))
            replay = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                mask = mask.astype(bool)
                missed = [j + 1 for j, (x, y) in enumerate(positive) if not mask[y, x]]
                hit = [j + 1 for j, (x, y) in enumerate(negative) if mask[y, x]]
                ys, xs = np.where(mask)
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
                candidate = {"candidate": i, "sam_score": float(score), "mask_area": int(mask.sum()),
                             "missed_positive_points": missed, "included_negative_points": hit,
                             "satisfies_points": bool(mask.any() and not missed and not hit), "box": box}
                replay.append(candidate)
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"candidate_{i}_mask.png")
                subtitle = f"漏正点 P{missed}；含负点 N{hit}；面积 {mask.sum()}"
                tile = paint(image, prompt, f"候选 {i} | SAM 分数 {float(score):.3f}", subtitle, mask=mask)
                tile.save(folder / f"candidate_{i}.png")
                board.paste(tile, ((i + 1) % 2 * 640, (i + 1) // 2 * 570))
            board.save(folder / "candidates.jpg", quality=95)
            item["replayed_candidates"] = replay
            item["candidate_visualization"] = f"diagnostics/{rid}/candidates.jpg"
        save_json(folder / "audit.json", item)
        print(rid, outcome, flush=True)
    counts = {k: len(v) for k, v in grouped.items()}
    top = ["# 标框来源分类与 SAM2 原提示检查", "",
           "分类按当前框的真实来源，诊断按首次推理记录。未修改标注、提示点或训练数据。",
           "", "注意：人工框不等于 SAM 分割失败；SAM 分数也不等于目标身份正确。", ""]
    for group, label in GROUPS.items():
        rows = grouped[group]
        sub = [f"# {label}：{len(rows)} 段", "", "[返回总入口](../../README.md)", ""]
        by_task = defaultdict(list)
        for item in rows:
            by_task[item["task"]].append(item)
        for task, items in by_task.items():
            sub += [f"## {task}（{len(items)}）", ""]
            for item in items:
                rid = item["review_id"]
                if group == "not_required":
                    sub.append(f"- {rid} / EP{item['episode_index']:06d}")
                else:
                    sub += [f"### {rid}", "", f"首次结果：{item['explanation']}", "",
                            f"[查看原提示点](../../{item['points_visualization']}) · [坐标与诊断 JSON](../../diagnostics/{rid}/audit.json)", "",
                            f"![{rid}](overlays/{rid}.png)", ""]
        (args.output / "by_method" / group / "README.md").write_text("\n".join(sub) + "\n")
        save_json(args.output / "by_method" / group / "manifest.json", rows)
        top += [f"- [{label}：{len(rows)} 段](by_method/{group}/README.md)"]
    top += ["", "## 首次流程诊断", "",
            "- 实际运行 SAM2：244 段；225 段通过点约束，19 段被点约束筛掉。",
            "- 没有生成点、也没运行 SAM2：60 段（48 配准未通过、11 当时推定越界、1 几何检查未通过）。",
            "- 直接采用区域候选、未运行 SAM2：16 段。",
            "- 以上失败类别是程序记录，不是首帧不可见的视觉结论。", "",
            "[19 段真实 SAM 点约束失败：原点与三个候选掩码](sam_rejected.md)", "",
            "[60 段未生成提示点：原始首帧与未执行原因](no_prompt.md)", ""]
    for name, outcome in (("sam_rejected.md", "sam_rejected_point_constraints"),
                          ("no_prompt.md", "no_point_prompt_no_sam")):
        lines = ["# " + ("SAM2 点约束未通过" if name.startswith("sam_") else "未生成提示点，SAM2 未执行"), "",
                 "[返回总入口](README.md)", "", "绿 P 是原正点，红 N 是原负点，紫框也是原 SAM 输入。黄色人工框仅供对照，不输入本次复现。", ""]
        for item in outcomes[outcome]:
            rid = item["review_id"]
            lines += [f"## {rid}", "", f"初次状态：{item['prompt'].get('status')}。当前类别：{GROUPS[item['method_group']]}。", "",
                      f"[原始点坐标和候选检查](diagnostics/{rid}/audit.json) · [不带任何标记的首帧](../frames/{rid}.png)", "",
                      f"![{rid}]({item.get('candidate_visualization', item['points_visualization'])})", ""]
        (args.output / name).write_text("\n".join(lines) + "\n")
    save_json(args.output / "manifest.json", {"counts": counts, "initial_outcomes": {k: len(v) for k, v in outcomes.items()},
              "original_annotations": str(original_path), "current_annotations": str(current_path),
              "sam_checkpoint": str(args.checkpoint), "device": args.device,
              "replay_uses_original_points_and_seed_box": True, "episodes": audit})
    (args.output / "README.md").write_text("\n".join(top) + "\n")
    if args.ffmpeg:
        export_method_videos(args.output, args.ffmpeg)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--success-review-only", action="store_true",
                        help="Export originally accepted SAM boxes and points without model inference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable for per-method videos")
    args = parser.parse_args()
    if args.success_review_only:
        export_sam_success_review(args.dataset, args.output)
    else:
        if not args.sam_source or not args.checkpoint:
            parser.error("--sam-source and --checkpoint are required for rejected-prompt replay")
        run(args)
