"""Live terminal latency curves for the SONIC data-collection stack.

The data exporter writes ThinkPad-side timings to JSONL.  This dashboard tails
that file and also parses the periodic ``Loop timing`` line from the G1 deploy
tmux pane.  It never opens a Unitree DDS channel and cannot command the robot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time


SPARK_CHARS = "▁▂▃▄▅▆▇█"

G1_PATTERNS: dict[str, tuple[str, float]] = {
    "g1.low_state_age_ms": (r"LowState age:\s*([-+\d.]+)ms", 1.0),
    "g1.streaming_mean_delay_ms": (r"Streaming data mean delay:\s*([-+\d.]+)ms", 1.0),
    "g1.streaming_std_delay_ms": (r"Streaming data std delay:\s*([-+\d.]+)ms", 1.0),
    "g1.imu_age_ms": (r"IMU age:\s*([-+\d.]+)ms", 1.0),
    "g1.obs_ms": (r", Obs:\s*([-+\d.]+)us", 0.001),
    "g1.policy_ms": (r", Policy:\s*([-+\d.]+)us", 0.001),
    "g1.obs_to_motor_ms": (r"Obs 2 Motor Command:\s*([-+\d.]+)us", 0.001),
    "g1.post_processing_ms": (r"Post processing:\s*([-+\d.]+)us", 0.001),
    "g1.planner_gather_ms": (r"Planner - Gather Input:\s*([-+\d.]+)us", 0.001),
    "g1.planner_model_ms": (r", Model:\s*([-+\d.]+)us", 0.001),
    "g1.planner_convert_ms": (r"Convert50Hz:\s*([-+\d.]+)us", 0.001),
    "g1.planner_total_ms": (r", Total:\s*([-+\d.]+)us", 0.001),
}

DISPLAY_GROUPS = (
    (
        "G1 控制与推理",
        (
            "g1.streaming_mean_delay_ms",
            "g1.low_state_age_ms",
            "g1.imu_age_ms",
            "g1.obs_ms",
            "g1.policy_ms",
            "g1.obs_to_motor_ms",
            "g1.planner_model_ms",
            "g1.planner_total_ms",
        ),
    ),
    (
        "ThinkPad 接收与写数据",
        (
            "source.state_update_age_ms",
            "source.sonic_pose_age_ms",
            "source.planner_command_age_ms",
            "source.camera_update_age_ms",
            "source.camera_timestamp_age_ms",
            "exporter.poll_state_ms",
            "exporter.poll_sonic_ms",
            "exporter.poll_image_ms",
            "exporter.add_frame_ms",
            "exporter.total_loop_ms",
        ),
    ),
)

DISPLAY_NAMES = {
    "g1.streaming_mean_delay_ms": "G1 输入流延迟",
    "g1.low_state_age_ms": "LowState 数据年龄",
    "g1.imu_age_ms": "IMU 数据年龄",
    "g1.obs_ms": "观测构建",
    "g1.policy_ms": "策略推理",
    "g1.obs_to_motor_ms": "观测到电机命令",
    "g1.planner_model_ms": "Planner 推理",
    "g1.planner_total_ms": "Planner 总耗时",
    "source.state_update_age_ms": "机器人状态更新年龄",
    "source.sonic_pose_age_ms": "PICO/SMPL 更新年龄",
    "source.planner_command_age_ms": "Planner 命令更新年龄",
    "source.camera_update_age_ms": "相机消息更新年龄",
    "source.camera_timestamp_age_ms": "相机时间戳年龄*",
    "exporter.poll_state_ms": "读取机器人状态",
    "exporter.poll_sonic_ms": "读取 SONIC/PICO",
    "exporter.poll_image_ms": "读取/解码图像",
    "exporter.add_frame_ms": "组帧与写缓存",
    "exporter.total_loop_ms": "Exporter 循环总耗时",
}


def parse_g1_timing(text: str) -> dict[str, float]:
    marker = "Loop timing -"
    start = text.rfind(marker)
    if start < 0:
        return {}
    line = text[start:]
    metrics: dict[str, float] = {}
    for name, (pattern, scale) in G1_PATTERNS.items():
        match = re.search(pattern, line)
        if match is not None:
            metrics[name] = float(match.group(1)) * scale
    return metrics


def sparkline(values: list[float], width: int) -> str:
    if not values:
        return " " * width
    values = values[-width:]
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        chars = SPARK_CHARS[len(SPARK_CHARS) // 2] * len(values)
    else:
        span = high - low
        chars = "".join(
            SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((value - low) / span * len(SPARK_CHARS)))]
            for value in values
        )
    return chars.rjust(width)


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


class JsonlTail:
    def __init__(self, path: Path):
        self.path = path
        self._stream = None

    def read_new(self) -> list[dict]:
        if self._stream is None:
            if not self.path.exists():
                return []
            self._stream = self.path.open("r", encoding="utf-8")
            self._stream.seek(0, 2)

        records = []
        for line in self._stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records


def capture_g1_timing(pane_target: str) -> dict[str, float]:
    result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-J", "-S", "-300", "-t", pane_target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    return parse_g1_timing(result.stdout)


def render(
    history: dict[str, deque[float]],
    latest: dict,
    input_path: Path,
    output_path: Path,
) -> str:
    terminal_width = shutil.get_terminal_size((120, 40)).columns
    chart_width = max(20, min(64, terminal_width - 64))
    recording = "录制中" if latest.get("recording") else "空闲"
    episode = latest.get("episode_index", "-")
    lines = [
        "SONIC 数据采集延迟曲线  |  Ctrl+C 退出本页（不会停止采集）",
        f"状态: {recording}  episode: {episode}  样本周期: 实时",
        f"原始日志: {input_path}",
        f"合并日志: {output_path}",
    ]

    for title, names in DISPLAY_GROUPS:
        lines.append("")
        lines.append(f"[{title}]  单位: ms")
        for name in names:
            values = list(history.get(name, ()))
            if not values:
                continue
            current = values[-1]
            mean = sum(values) / len(values)
            p95 = percentile95(values)
            graph = sparkline(values, chart_width)
            label = DISPLAY_NAMES.get(name, name)
            lines.append(
                f"{label:<18} {graph}  当前 {current:7.2f}  均值 {mean:7.2f}  P95 {p95:7.2f}"
            )

    if not any(history.values()):
        lines.extend(["", "等待 Data Exporter 延迟样本……"])
    lines.extend(
        [
            "",
            "* 相机时间戳年龄依赖 G1 与 ThinkPad 系统时钟同步；更新年龄不依赖跨机时钟。",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    sample = (
        "Loop timing - LowState age: 2.5ms, Streaming data mean delay: 8ms, "
        "Streaming data std delay: 1ms, IMU age: 3ms, Obs: 1200us, "
        "Policy: 2400us, Obs 2 Motor Command: 3600us, Post processing: 400us, "
        "Planner - Gather Input: 100us, Model: 31000us, Convert50Hz: 200us, Total: 31300us"
    )
    parsed = parse_g1_timing(sample)
    assert parsed["g1.low_state_age_ms"] == 2.5
    assert parsed["g1.policy_ms"] == 2.4
    assert parsed["g1.planner_model_ms"] == 31.0
    assert len(sparkline([1.0, 2.0, 3.0], 5)) == 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-log-file", default="logs/data_exporter_latency.jsonl")
    parser.add_argument("--combined-log-file", default="logs/data_collection_latency.jsonl")
    parser.add_argument("--g1-pane-target", default="sonic_data_collection:data_collection.0")
    parser.add_argument("--history-size", type=int, default=180)
    parser.add_argument("--refresh-rate", type=float, default=5.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("Latency dashboard self-test passed")
        return

    input_path = Path(args.latency_log_file)
    output_path = Path(args.combined_log_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_stream = output_path.open("a", encoding="utf-8", buffering=1)
    tail = JsonlTail(input_path)
    history: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=max(20, args.history_size))
    )
    latest: dict = {}
    latest_g1: dict[str, float] = {}
    last_g1_poll = 0.0
    period = 1.0 / max(1.0, args.refresh_rate)

    try:
        while True:
            loop_start = time.monotonic()
            if loop_start - last_g1_poll >= 1.0:
                parsed = capture_g1_timing(args.g1_pane_target)
                if parsed:
                    latest_g1 = parsed
                last_g1_poll = loop_start

            for record in tail.read_new():
                merged = {**record, **latest_g1}
                latest = merged
                for key, value in merged.items():
                    if key.endswith("_ms") and isinstance(value, (int, float)):
                        history[key].append(float(value))
                combined_stream.write(json.dumps(merged, separators=(",", ":")) + "\n")

            print("\033[2J\033[H" + render(history, latest, input_path, output_path), end="", flush=True)
            remaining = period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        combined_stream.close()
        print("\033[0m")


if __name__ == "__main__":
    main()
