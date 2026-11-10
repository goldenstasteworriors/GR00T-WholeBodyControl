"""Review full tube destination regions using manually marked after-return images.

Only ego first-frame annotations change. Video and numeric data stay untouched.
Image registration transfers the manually selected tube extent as a proposal;
explicit first-frame corrections may override it after visual inspection.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def run(dataset, output, prompts, ffmpeg):
    previous_path = output / "annotations_before.json"
    if not previous_path.exists():
        previous_path.write_bytes((dataset / "meta/initial_ego_boxes.json").read_bytes())
    annotations = json.loads(previous_path.read_text())
    manual = json.loads(prompts.read_text())
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17)
    for name in ("overlays", "comparisons", "galleries"):
        (output / name).mkdir(exist_ok=True)
    panels, changes = [], []
    sift = cv2.SIFT_create(nfeatures=5000)
    for record in annotations["episodes"]:
        if record["task"] != "RETURN_TUBE_TO_RACK":
            continue
        rid = record["review_id"]
        proposal = manual[rid]
        first = Image.open(output / "evidence" / f"{rid}_first.png").convert("RGB")
        last = Image.open(output / "evidence" / f"{rid}_last.png").convert("RGB")
        gray = [cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY) for im in (last, first)]
        (k1, d1), (k2, d2) = [sift.detectAndCompute(im, None) for im in gray]
        pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        matches = [a for a, b in pairs if a.distance < .7 * b.distance]
        homography, inliers = cv2.findHomography(
            np.float32([k1[m.queryIdx].pt for m in matches]),
            np.float32([k2[m.trainIdx].pt for m in matches]), cv2.RANSAC, 3)
        assert homography is not None and int(inliers.sum()) >= 20, rid
        x1, y1, x2, y2 = proposal["after_box"]
        corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        mapped = cv2.perspectiveTransform(corners, homography)[0]
        projected = [max(0, int(np.floor(mapped[:, 0].min()))), max(0, int(np.floor(mapped[:, 1].min()))),
                     min(640, int(np.ceil(mapped[:, 0].max()))), min(480, int(np.ceil(mapped[:, 1].max())))]
        box = proposal.get("first_box", projected)
        assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 480, (rid, box)
        before = deepcopy(record["targets"])
        record["targets"] = [{"camera": "observation.images.ego_view", "role": 1,
                              "entity": "destination_rack_slot", "box": box,
                              "source": "manual_full_tube_destination_region"}]
        record.update(status="manual_full_tube_region_pending_user_review", user_approved=False,
                      needs_user_review=True,
                      annotation_note="参考放回后的整支试管，标首帧中对应的完整放置区域；不是首帧中的实体试管分割。")
        record["annotation_note"] += proposal.get("note", "")
        for stale in ("sam_candidates", "completion_evidence"):
            record.pop(stale, None)
        record["prompt"] = {"manual_after_box": proposal["after_box"],
                            "first_frame_box": box, "source": "before_after_visual_review"}
        record["destination_region_review"] = {**proposal, "projected_box": projected,
            "homography": homography.tolist(), "registration_inliers": int(inliers.sum()),
            "previous_targets": before, "semantics": "full tube destination extent, not small slot aperture"}
        panel = Image.new("RGB", (640, 560), "#14202b")
        panel.paste(first, (0, 80))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 3), f"{rid} | EP{record['episode_index']:06d}", fill="white", font=font)
        draw.text((8, 28), "人工完整放管区域 · ego 首帧 · 待检查", fill="#ffcf40", font=font)
        draw.text((8, 53), f"新框 {box}", fill="white", font=font)
        draw.rectangle((box[0], box[1]+80, box[2]-1, box[3]+79), outline="#ffcf40", width=3)
        panel.save(output / "overlays" / f"{rid}.png")
        comparison = Image.new("RGB", (1280, 560), "#14202b")
        comparison.paste(panel)
        comparison.paste(last, (640, 80))
        d = ImageDraw.Draw(comparison)
        d.text((648, 8), "放回后：用于确认目标试管及完整范围", fill="white", font=font)
        d.text((648, 35), proposal.get("note", ""), fill="#ffcf40", font=font)
        d.rectangle((x1+640, y1+80, x2+639, y2+79), outline="#45ff86", width=3)
        comparison.save(output / "comparisons" / f"{rid}.jpg", quality=95)
        panels.append((rid, panel))
        changes.append({"review_id": rid, "previous": before, "box": box, "after_box": proposal["after_box"]})
    assert set(manual) == {rid for rid, _ in panels}
    annotations["status_counts"] = dict(Counter(r["status"] for r in annotations["episodes"]))
    annotations["return_tube_region_review"] = str(output.resolve())
    text = json.dumps(annotations, ensure_ascii=False, indent=2) + "\n"
    (output / "annotations.json").write_text(text)
    (dataset / "meta/initial_ego_boxes.json").write_text(text)
    (output / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 放回试管：完整放置区域检查", "", f"共 {len(panels)} 段，全部人工标注，不使用 SAM。黄色是首帧放置区域，绿色是放回后用于核对的实际试管范围。目标在首帧可能为空；框表示目标试管放回后占据的范围，不是首帧实体分割。右侧出画的片段仅保留画面内的区域，并单独注明，不能解释为整支管都可见。", "", "未修改起点、三路视频或数值数据。旧框保存在 annotations_before.json。", "", f"[{len(panels)} 段检查视频，每张一秒](review.mp4)", ""]
    for offset in range(0, len(panels), 12):
        sheet = Image.new("RGB", (1280, 4*374), "#14202b")
        for j, (_, panel) in enumerate(panels[offset:offset+12]):
            sheet.paste(panel.resize((426,374)), (j%3*426,j//3*374))
        name = f"galleries/page_{offset//12:02d}.jpg"
        sheet.save(output/name, quality=95)
        lines += [f"![图集]({name})", ""]
    for rid, _ in panels:
        lines += [f"## {rid}", "", f"![首帧完整区域与放回后试管对照](comparisons/{rid}.jpg)", ""]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    playlist = output/"review.ffconcat"
    playlist.write_text("ffconcat version 1.0\n"+"".join(f"file 'overlays/{rid}.png'\nduration 1\n" for rid,_ in panels)+f"file 'overlays/{panels[-1][0]}.png'\n")
    subprocess.run([ffmpeg,"-y","-v","error","-nostdin","-safe","0","-i",str(playlist),"-vf","fps=30","-frames:v",str(len(panels)*30),"-c:v","libx264","-crf","18","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(output/"review.mp4")],check=True)
    print(json.dumps({"revised_return_tube_regions":len(panels)},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    run(args.dataset, args.output, args.prompts, args.ffmpeg)
