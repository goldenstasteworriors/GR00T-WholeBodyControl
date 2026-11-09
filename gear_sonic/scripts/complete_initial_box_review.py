"""Apply identity-reviewed missing boxes to a separately curated training export.

SAM boxes come from saved masks; slot boxes are explicit reviewed regions.
Excluded source episodes remain available in the original dataset for audit.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from annotate_lab_initial_boxes import CAMERA, RULES
from segment_lab_primitives import episode_path


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def register_slot_regions(dataset, evidence, anchors, output):
    """Map manually identified support-plane slot centers to current first frames."""
    import cv2

    info = json.loads((dataset / "meta/info.json").read_text())
    rows = {r["review_id"]: r for r in json.loads((dataset / "meta/segment_provenance.json").read_text())}
    events = {r["review_id"]: r for r in json.loads((evidence / "manifest.json").read_text())}
    sift = cv2.SIFT_create(nfeatures=5000)
    regions = {}
    for rid, point in anchors.items():
        row, event = rows[rid], events[rid]
        with av.open(str(episode_path(dataset, info, row["output_episode"], CAMERA))) as c:
            first = next(c.decode(video=0)).to_ndarray(format="rgb24")
        last_path = evidence / rid / f"ego_view_f{event['end_frame']-1:06d}.jpg"
        last = np.array(Image.open(last_path).convert("RGB"))
        k1, d1 = sift.detectAndCompute(cv2.cvtColor(last, cv2.COLOR_RGB2GRAY), None)
        k2, d2 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), None)
        pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        matches = [a for a, b in pairs if a.distance < .7 * b.distance]
        h, inliers = cv2.findHomography(np.float32([k1[m.queryIdx].pt for m in matches]),
                                       np.float32([k2[m.trainIdx].pt for m in matches]), cv2.RANSAC, 3)
        assert h is not None and int(inliers.sum()) >= 20, rid
        x, y = point
        corners = np.float32([[[x - 12, y - 9], [x + 12, y - 9],
                               [x + 12, y + 9], [x - 12, y + 9], [x, y]]])
        mapped = cv2.perspectiveTransform(corners, h)[0]
        box = [int(np.floor(mapped[:4, 0].min())), int(np.floor(mapped[:4, 1].min())),
               int(np.ceil(mapped[:4, 0].max())), int(np.ceil(mapped[:4, 1].max()))]
        assert 0 <= box[0] < box[2] <= 640 and 0 <= box[1] < box[3] <= 480, (rid, box)
        regions[rid] = {"method": "reviewed_slot_region", "box": box,
                        "source_frame_index": row["start_frame"],
                        "end_reference_point": point, "end_reference_image": str(last_path.resolve()),
                        "mapped_center": mapped[4].tolist(), "registration_inliers": int(inliers.sum()),
                        "registration_matches": len(matches), "homography": h.tolist(),
                        "note": "根据实际放回时的试管位置确认目标槽，人工区域框；仍需核对槽位边界。"}
    write_json(output, regions)
    return regions


def run(args):
    previous = json.loads((args.previous / "initial_ego_boxes/review_v3/annotations.json").read_text())
    old = {r["review_id"]: r for r in previous["episodes"]}
    decisions = json.loads(args.decisions.read_text())
    rows = json.loads((args.dataset / "meta/segment_provenance.json").read_text())
    info = json.loads((args.dataset / "meta/info.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    frames = args.dataset / "initial_ego_boxes/frames"
    frames.mkdir(parents=True, exist_ok=False)
    for name in ("overlays", "masks", "sam_details", "slot_evidence", "galleries"):
        (args.output / name).mkdir()
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    records, added, panels = [], [], []
    lines = ["# 未出框片段补标检查", "", "本轮按实际取走／放回过程核对目标。新增框仍待用户检查；已有框继承此前检查版。", "",
             "绿色为操作物，黄色为目标槽位。SAM2 对象框与人工槽位区域框分别注明来源。", ""]
    for row in rows:
        rid = row["review_id"]
        record = deepcopy(old[rid])
        record.update(episode_index=row["output_episode"], source_frame_index=row["start_frame"],
                      previous_episode_index=old[rid]["episode_index"], frame_index=0)
        if row["task"] not in RULES:
            assert not record["targets"]
            records.append(record)
            continue
        with av.open(str(episode_path(args.dataset, info, row["output_episode"], CAMERA))) as c:
            image = next(c.decode(video=0)).to_image().convert("RGB")
        frame = frames / f"{rid}.png"
        image.save(frame)
        record["reference_image_sha256"] = hashlib.sha256(frame.read_bytes()).hexdigest()
        record["reference_image"] = str(frame.relative_to(args.dataset))
        decision = decisions.get(rid)
        if decision:
            assert not old[rid]["targets"], rid
            if decision["method"] == "sam2_confirmed_object":
                result_path = Path(decision["result_json"])
                result = json.loads(result_path.read_text())
                old_image = Image.open(args.previous / "initial_ego_boxes/frames" / f"{rid}.png").convert("RGB")
                np.testing.assert_array_equal(np.asarray(image), np.asarray(old_image))
                selected = result["examples"][0]["selected"]
                assert selected and selected["valid"]
                box, method = selected["box"], "sam2_identity_reviewed_points"
                mask_path = result_path.parent / "confirmed_target" / f"candidate_{selected['candidate']}_mask.png"
                shutil.copy2(mask_path, args.output / "masks" / f"{rid}.png")
                shutil.copy2(result_path.parent / "confirmed_target/comparison.jpg", args.output / "sam_details" / f"{rid}.jpg")
                record["prompt"] = result["examples"][0]["example"]
                record["sam_candidates"] = result["examples"][0]["candidates"]
            else:
                assert decision["method"] == "reviewed_slot_region"
                assert decision["source_frame_index"] == row["start_frame"]
                box, method = decision["box"], "identity_reviewed_slot_region"
                end_image = Image.open(decision["end_reference_image"]).convert("RGB")
                end_draw = ImageDraw.Draw(end_image)
                x, y = decision["end_reference_point"]
                end_draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#ffcf40", width=2)
                end_image.save(args.output / "slot_evidence" / f"{rid}.jpg", quality=95)
                record["prompt"] = {"region_box": box, "source": "actual_placement_identity_review"}
                record.pop("sam_candidates", None)
            x1, y1, x2, y2 = box
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480
            role, entity = RULES[row["task"]]
            record["targets"] = [{"camera": CAMERA, "role": role, "entity": entity, "box": box, "source": method}]
            record.update(status="completed_identity_review_pending_user_review", user_approved=False,
                          needs_user_review=True, completion_evidence=decision)
            panel = Image.new("RGB", (640, 590), "#14202b")
            panel.paste(image, (0, 110))
            draw = ImageDraw.Draw(panel)
            shift = row["start_frame"] - row["previous_start_frame"]
            draw.text((8, 3), f"{rid} | 新 EP{row['output_episode']:06d}", font=font, fill="white")
            draw.text((8, 28), f"{method} | ego 首帧 | 待检查", font=font, fill="#ffdc72")
            draw.text((8, 53), f"原始 f{row['start_frame']} | 起点变更 {shift:+d}帧 | 框 {box}", font=font, fill="white")
            draw.text((8, 78), decision.get("note", "通过取走前后及腕部视角确认目标身份")[:42], font=font, fill="white")
            draw.rectangle((x1, y1 + 110, x2 - 1, y2 + 109), outline="#45ff86" if role == 0 else "#ffcf40", width=3)
            panel.save(args.output / "overlays" / f"{rid}.png")
            panels.append((rid, panel))
            added.append(record)
            lines += [f"## {rid}", "", f"来源：{method}；起点变更 {shift:+d} 帧。", "",
                      f"![补框后的首帧](overlays/{rid}.png)", ""]
            if method.startswith("sam2"):
                lines += [f"[给点及实际 SAM2 掩码](sam_details/{rid}.jpg)", ""]
            else:
                lines += [f"[实际放回后的槽位参考位置](slot_evidence/{rid}.jpg)", ""]
        assert record["targets"], f"Unresolved required annotation: {rid}"
        records.append(record)
    assert set(decisions) == {r["review_id"] for r in added}
    annotation = {**previous, "dataset": str(args.dataset.resolve()), "episodes": records,
                  "galleries": {}, "videos": {}, "completion_decisions": str(args.decisions.resolve()),
                  "status_counts": dict(Counter(r["status"] for r in records))}
    write_json(args.output / "annotations.json", annotation)
    write_json(args.dataset / "meta/initial_ego_boxes.json", annotation)
    write_json(args.output / "newly_completed.json", added)
    curation = json.loads((args.dataset / "meta/annotations.json").read_text())
    write_json(args.dataset / "meta/visibility_exclusions.json", curation["excluded_primitives"])
    lines += ["## 从本训练版排除的片段", ""]
    for exclusion in curation["excluded_primitives"]:
        lines += [f"- {exclusion['review_id']}：{exclusion['reason']}"]
    for offset in range(0, len(panels), 12):
        sheet = Image.new("RGB", (1280, 4 * 394), "#14202b")
        for j, (_, panel) in enumerate(panels[offset:offset + 12]):
            sheet.paste(panel.resize((426, 394)), (j % 3 * 426, j // 3 * 394))
        sheet.save(args.output / "galleries" / f"page_{offset//12:02d}.jpg", quality=95)
    playlist = args.output / "newly_completed.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(f"file 'overlays/{rid}.png'\nduration 1\n" for rid, _ in panels)
                        + f"file 'overlays/{panels[-1][0]}.png'\n")
    subprocess.run([args.ffmpeg, "-v", "error", "-nostdin", "-safe", "0", "-i", str(playlist), "-vf", "fps=30",
                    "-frames:v", str(len(panels) * 30), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output / "newly_completed.mp4")], check=True)
    lines.insert(6, "[补框检查视频，每段一秒](newly_completed.mp4)\n")
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"episodes": len(records), "new_boxes": len(added), "required_with_boxes": sum(bool(r["targets"]) for r in records),
                      "exclusions": len(curation["excluded_primitives"])}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    run(parser.parse_args())
