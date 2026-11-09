"""Compare explicit point/box examples on one original ego frame with local SAM2."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(args):
    import torch

    sys.path.insert(0, str(args.sam_source))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    config = json.loads(args.prompts.read_text())
    frame = args.dataset / "initial_ego_boxes/frames" / f"{args.review_id}.png"
    image = Image.open(frame).convert("RGB")
    rgb = np.array(image)
    torch.set_num_threads(4)
    predictor = SAM2ImagePredictor(build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml", str(args.checkpoint), device=args.device,
        apply_postprocessing=False))
    args.output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    panels, records = [], []
    for example in config["examples"]:
        folder = args.output / example["name"]
        folder.mkdir()
        pos, neg = example["positive_points"], example["negative_points"]
        with torch.inference_mode():
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=np.asarray(pos + neg, np.float32),
                point_labels=np.asarray([1] * len(pos) + [0] * len(neg), np.int32),
                box=np.asarray(example["box"], np.float32), multimask_output=True)
        candidates = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask)
            missing = [j + 1 for j, (x, y) in enumerate(pos) if not mask[y, x]]
            included = [j + 1 for j, (x, y) in enumerate(neg) if mask[y, x]]
            candidates.append({"candidate": i, "score": float(scores[i]), "area": int(mask.sum()),
                               "missed_positive_points": missing, "included_negative_points": included,
                               "valid": bool(len(xs) and not missing and not included),
                               "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None})
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"candidate_{i}_mask.png")
        valid = [c for c in candidates if c["valid"]]
        selected = max(valid, key=lambda c: c["score"]) if valid else None
        panel = Image.new("RGB", (1280, 590), "#14202b")
        panel.paste(image, (0, 90))
        result = rgb.copy()
        if selected:
            mask = masks[selected["candidate"]].astype(bool)
            result[mask] = (.6 * result[mask] + .4 * np.array([0, 200, 255])).astype(np.uint8)
        panel.paste(Image.fromarray(result), (640, 90))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 3), f"{args.review_id} | {example['label']}", font=font, fill="white")
        draw.text((8, 27), "左：给 SAM2 的示范范围（紫框）、正点（绿）、负点（红）", font=font, fill="white")
        draw.text((648, 27), "右：SAM2 掩码（青色）与由掩码计算的框", font=font, fill="#47dcff")
        subtitle = f"候选 {selected['candidate']} | 分数 {selected['score']:.3f} | 输出框 {selected['box']}" if selected else "无候选满足全部点约束；不自动用示范框代替"
        draw.text((8, 52), subtitle, font=font, fill="#ffdc72")
        x1, y1, x2, y2 = example["box"]
        draw.rectangle((x1, y1 + 90, x2 - 1, y2 + 89), outline="#d06dff", width=2)
        for points, color in ((pos, "#37ff74"), (neg, "#ff4242")):
            for x, y in points:
                draw.ellipse((x - 4, y + 86, x + 4, y + 94), fill=color, outline="black")
        if selected:
            x1, y1, x2, y2 = selected["box"]
            draw.rectangle((x1 + 640, y1 + 90, x2 + 639, y2 + 89), outline="#47dcff", width=2)
        panel.save(folder / "comparison.jpg", quality=95)
        panels.append(panel)
        records.append({"example": example, "candidates": candidates, "selected": selected})
        print(example["name"], selected, flush=True)
    board = Image.new("RGB", (1280, 590 * len(panels)), "#14202b")
    for i, panel in enumerate(panels):
        board.paste(panel, (0, 590 * i))
    board.save(args.output / "all_comparisons.jpg", quality=95)
    report = {"review_id": args.review_id, "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
              "checkpoint": str(args.checkpoint), "demonstration_type": "explicit geometric point and box prompts; no text or reference image conditioning",
              "selection": "maximum SAM score among nonempty masks satisfying all positive and negative points",
              "examples": records}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# {args.review_id} 首帧：明确范围提示后的 SAM2 对照", "",
             "这里的示范被转换为实际图像上的点和输入框，没有使用文字模型或示例图编码。未修改训练标注。",
             "", "左图是输入，右图是 SAM2 实际输出。青色框由模型掩码计算，未用人工示范框冒充输出。", ""]
    for record in records:
        example = record["example"]
        lines += [f"## {example['label']}", "", f"![输入与输出]({example['name']}/comparison.jpg)", "",
                  "```json", json.dumps(example, ensure_ascii=False), "```", "",
                  "输出：", "```json", json.dumps(record["selected"], ensure_ascii=False), "```", ""]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sam-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
