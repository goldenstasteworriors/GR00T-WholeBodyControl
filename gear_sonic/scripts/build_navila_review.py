"""Label recorded navigation commands and build an offline, three-view review.

No robot/network I/O. Metric amounts remain null: command integration is only
a weak estimate, and stationary frames do not establish task success.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


NAMES = {
    "forward": "前进", "backward": "后退", "strafe_left": "左横移",
    "strafe_right": "右横移", "turn_left": "左转", "turn_right": "右转",
    "idle": "零速/微量指令", "inactive": "非运行模式",
}
COLORS = {"forward": (178, 196, 40), "backward": (100, 140, 245),
          "strafe_left": (235, 170, 80), "strafe_right": (220, 120, 180),
          "turn_left": (80, 210, 245), "turn_right": (100, 150, 240),
          "idle": (160, 160, 160), "inactive": (100, 100, 100)}
STANDARD = {"forward", "turn_left", "turn_right"}


def command_label(command, mode):
    if mode != 3:
        return "inactive"
    x, y, yaw = command
    parts = []
    if abs(x) > .02:
        parts.append("forward" if x > 0 else "backward")
    if abs(y) > .02:
        parts.append("strafe_left" if y > 0 else "strafe_right")
    if abs(yaw) > .05:
        parts.append("turn_left" if yaw > 0 else "turn_right")
    return "+".join(parts) or "idle"


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def make_episode(root, category, source, output, ffmpeg, visual_review):
    dataset = root / "curated" / Path(category["path"]).name
    info = json.loads((dataset / "meta/info.json").read_text())
    ep, original = source["episode_index"], source["source_episode_index"]
    fmt = dict(episode_index=ep, episode_chunk=ep // info["chunks_size"])
    parquet = dataset / info["data_path"].format(**fmt)
    df = pd.read_parquet(parquet)
    n = len(df)
    times = np.asarray(df["navigation.camera_time"].tolist(), float).reshape(-1)
    if n < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid timestamps: {parquet}")
    relative = times - times[0]
    velocity = np.stack(df["navigation.policy_velocity"].to_numpy())
    modes = np.asarray(df["navigation.control_mode"].tolist(), int).reshape(-1)
    labels = [command_label(v, m) for v, m in zip(velocity, modes)]
    boundaries = [0] + [i for i in range(1, n) if labels[i] != labels[i - 1]] + [n]
    stem = f"c{category['category']}_source_{original:03d}"
    segments = []
    # Interval [i,i+1) belongs to frame i. Last frame has no extrapolated dt.
    dt = np.r_[np.diff(times), 0.0]
    for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        action = labels[start]
        integral = np.sum(velocity[start:end] * dt[start:end, None], axis=0)
        components = action.split("+")
        segments.append({
            "segment_id": idx, "start_frame": start, "end_frame": end,
            "start_s": float(relative[start]),
            "end_s": float(relative[end] if end < n else relative[-1]),
            "action": action, "action_zh": "＋".join(NAMES[c] for c in components),
            "label_source": "recorded_policy_velocity_with_deadband",
            "standard_navila_action": action in STANDARD,
            "command_integral_estimate": {"forward_m": float(integral[0]),
                "left_m": float(integral[1]), "yaw_deg": float(math.degrees(integral[2]))},
            "measured_amount": None, "reviewed": False,
            "review_note": "方向来自记录指令；尚无实测平移距离及逐段运动审核。",
        })
    last_active = next((i for i in range(n - 1, -1, -1)
                        if labels[i] not in ("idle", "inactive")), None)
    # Only a terminal zero-command run is a STOP candidate; recording end isn't STOP.
    terminal_idle = next((s for s in reversed(segments)
                          if s["end_frame"] == n and s["action"] == "idle"), None)
    terminal = {"start_frame": terminal_idle["start_frame"] if terminal_idle else None,
                "zero_command_duration_s": (terminal_idle["end_s"] - terminal_idle["start_s"])
                    if terminal_idle else 0,
                "last_movement_command_frame": last_active,
                "stop_candidate": bool(terminal_idle and terminal_idle["end_s"] - terminal_idle["start_s"] >= .3),
                "success": None, "reviewed": False,
                "reason": "零速指令只标停止候选；目标物可见不等于满足物理到达距离。"}
    video_paths = [dataset / info["video_path"].format(**fmt, video_key=f"observation.images.{key}")
                   for key in ("ego_view", "left_wrist", "right_wrist")]
    captures = [cv2.VideoCapture(str(p)) for p in video_paths]
    fps = (n - 1) / relative[-1]
    out_video = output / "videos" / f"{stem}.mp4"
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", "960x592", "-framerate", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
               "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    keys = np.unique(np.linspace(0, n - 1, 6, dtype=int)).tolist()
    snapshots = []
    try:
        for i in range(n):
            frames = []
            for capture in captures:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Video missing frame {i}: {stem}")
                frames.append(frame)
            canvas = np.full((592, 960, 3), (24, 28, 28), np.uint8)
            canvas[80:560, :640] = cv2.resize(frames[0], (640, 480))
            canvas[80:320, 640:] = cv2.resize(frames[1], (320, 240))
            canvas[320:560, 640:] = cv2.resize(frames[2], (320, 240))
            text = f"TASK {category['category']} / SOURCE {original:03d} / FRAME {i:04d} / {relative[i]:.2f}s"
            cv2.putText(canvas, text, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (220, 228, 230), 1)
            cv2.putText(canvas, labels[i].upper(), (16, 56), cv2.FONT_HERSHEY_SIMPLEX, .65,
                        COLORS[labels[i].split("+")[0]], 2)
            cv2.putText(canvas, f"CMD vx {velocity[i,0]:+.3f}  vy {velocity[i,1]:+.3f} m/s  yaw {math.degrees(velocity[i,2]):+.1f} deg/s",
                        (430, 55), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 228, 230), 1)
            for title, pos in [("EGO / HEAD", (12, 104)), ("LEFT WRIST", (650, 104)),
                               ("RIGHT WRIST", (650, 344))]:
                cv2.putText(canvas, title, pos, cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 0, 0), 3)
                cv2.putText(canvas, title, pos, cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
            cv2.putText(canvas, "RECORDED COMMANDS | NOT MEASURED ODOMETRY | METRIC LABELS & GOAL SUCCESS UNCONFIRMED",
                        (12, 582), cv2.FONT_HERSHEY_SIMPLEX, .43, (190, 196, 198), 1)
            process.stdin.write(canvas.tobytes())
            if i in keys:
                thumb = cv2.resize(frames[0], (320, 240))
                cv2.putText(thumb, f"f{i}  {relative[i]:.2f}s", (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (0, 0, 0), 3)
                cv2.putText(thumb, f"f{i}  {relative[i]:.2f}s", (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1)
                snapshots.append(thumb)
            if i == n - 1:
                cv2.imwrite(str(output / "posters" / f"{stem}.jpg"), canvas)
        for capture in captures:
            if capture.read()[0]:
                raise ValueError(f"Video has extra frames: {stem}")
    finally:
        for capture in captures:
            capture.release()
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"Encoder failed: {stem}, exit={code}")
    # Short episodes can yield fewer than six distinct snapshots.
    while len(snapshots) < 6:
        snapshots.append(snapshots[-1].copy())
    contact = np.vstack([np.hstack(snapshots[:3]), np.hstack(snapshots[3:6])])
    cv2.imwrite(str(output / "contacts" / f"{stem}.jpg"), contact)
    result = {"id": stem, "category": category["category"], "task_zh": category["description"],
              "instruction": json.loads((dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"],
              "episode_index": ep, "source_episode_index": original, "frames": n,
              "duration_s": float(relative[-1]), "preview_fps": float(fps),
              "video": f"videos/{stem}.mp4", "poster": f"posters/{stem}.jpg",
              "contact": f"contacts/{stem}.jpg", "segments": segments,
              "terminal": terminal, "visual_review": visual_review.get(str(original), {}),
              "source_parquet": os.path.relpath(parquet, output),
              "source_videos": [os.path.relpath(p, output) for p in video_paths],
              "time_s": relative.round(6).tolist(), "commands": velocity.round(8).tolist(),
              "control_modes": modes.tolist(), "frame_labels": labels}
    write_json(output / "labels" / f"{stem}.json", result)
    with (output / "labels" / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "camera_time", "relative_s", "vx_cmd_m_s", "vy_cmd_m_s",
                         "yaw_cmd_rad_s", "mode", "command_action", "measured_amount", "success"])
        for i in range(n):
            writer.writerow([i, times[i], relative[i], *velocity[i], modes[i], labels[i], "", ""])
    print(f"DONE {stem}: {n} frames, {len(segments)} segments", flush=True)
    return result


def make_overviews(output, episodes):
    """A final ego-frame atlas for every category, with source IDs."""
    for category in sorted({e["category"] for e in episodes}):
        items = [e for e in episodes if e["category"] == category]
        cols = 5 if len(items) > 15 else 3
        rows = math.ceil(len(items) / cols)
        image = np.full((rows * 224 + 50, cols * 320, 3), 235, np.uint8)
        cv2.putText(image, f"TASK {category} / {len(items)} EPISODES / FINAL EGO FRAMES",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (30, 55, 40), 2)
        for i, episode in enumerate(items):
            x, y = i % cols * 320, 50 + i // cols * 224
            poster = cv2.imread(str(output / episode["poster"]))
            image[y:y + 192, x:x + 320] = cv2.resize(poster[80:560, :640], (320, 192))
            cv2.putText(image, f"SOURCE {episode['source_episode_index']:03d} / {episode['frames']} frames",
                        (x + 8, y + 215), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 55, 40), 1)
        cv2.imwrite(str(output / f"overview_category_{category}.jpg"), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path(sys.executable).parent / "ffmpeg")
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("videos", "posters", "contacts", "labels"):
        (output / name).mkdir()
    summary = json.loads((root / "summary.json").read_text())
    visual_review = json.loads(args.visual_review.read_text())
    jobs = []
    for category in summary["classes"]:
        dataset = root / "curated" / Path(category["path"]).name
        for row in (dataset / "meta/source_episodes.jsonl").read_text().splitlines():
            jobs.append((root, category, json.loads(row), output, args.ffmpeg, visual_review))
    with ThreadPoolExecutor(max_workers=2) as pool:
        episodes = list(pool.map(lambda job: make_episode(*job), jobs))
    write_json(output / "episodes.json", episodes)
    make_overviews(output, episodes)
    actions = Counter(s["action"] for e in episodes for s in e["segments"])
    report = {"episodes": len(episodes), "frames": sum(e["frames"] for e in episodes),
              "segments": sum(len(e["segments"]) for e in episodes), "segment_actions": dict(actions),
              "stop_candidates": sum(e["terminal"]["stop_candidate"] for e in episodes),
              "metric_labels_confirmed": 0, "success_labels_confirmed": 0,
              "deadbands": {"vx_vy_m_s": .02, "yaw_rad_s": .05},
              "segment_intervals": "[start_frame, end_frame); no extrapolation past final camera timestamp",
              "preview_time": "CFR at episode average FPS; labels indexed by source frame; actual capture times in JSON/CSV",
              "source_unchanged": True, "training_ready": False}
    write_json(output / "summary.json", report)
    template = Path(__file__).with_name("navila_review.html").read_text()
    payload = json.dumps(episodes, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (output / "index.html").write_text(template.replace("/*__EPISODES__*/[]", payload))
    (output / "README.txt").write_text(
        "NaVILA 导航动作标注与逐条可视化\n\n"
        "双击 index.html 即可离线浏览全部49条轨迹；也可通过 HTTP 服务打开。\n"
        "每条均有三视角带指令标注视频、6帧头部摘要、逐帧CSV及分段JSON。\n"
        "视频中 EGO 为头部，右上 LEFT WRIST，右下 RIGHT WRIST。\n"
        "网页可按任务筛选、点击动作段跳转、逐帧查看及导出人工修订JSON。\n"
        "横移、后退和同时多轴运动保留原义，没有伪标为左转或前进。\n"
        "标签来自记录的policy_velocity，微量指令阈值为平移0.02m/s、转向0.05rad/s；原始数值保留。\n"
        "距离/角度估计采用相机时间戳左端积分，仅为指令估计，不是实际里程计。\n"
        "success及measured_amount仍为空，reviewed=false；终止零速不自动等于任务成功。\n"
        "visual_review仅说明人工查看起终点图像所得的可见性，不代表逐段距离或成功已审核。\n"
        "所有49条均有动作方向标注；当前不能作为已审核的度量动作SFT训练集。\n"
        "网页底部可下载当前任务类已审核的原版训练分段，供export_navila_dataset.py的--labels参数使用。\n"
        "dataset参数必须选择对应类别的curated目录，横移和复合动作保留在原始标注中。\n"
        "原始视频及parquet未改动。网页播放为平均帧率预览，真实时间在CSV/JSON。\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
