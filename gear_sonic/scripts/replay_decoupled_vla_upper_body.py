#!/usr/bin/env python3
"""Replay one LeRobot episode as upper-body-only SONIC ZMQ commands.

The lower body is kept in planner IDLE mode.  Only the waist and arm slice of
the recorded ``action.wbc`` is sent through the deployer's supported
``upper_body_position`` input.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


REQUIRED_COLUMNS = ("action.wbc",)
HAND_COLUMNS = (
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只回放 LeRobot episode 的腰部、双臂和手腕，不发送腿部动作。"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot 数据集目录")
    parser.add_argument("--episode", type=int, default=0, help="episode 编号（默认：0）")
    parser.add_argument("--speed", type=float, default=0.25, help="播放倍速（默认：0.25）")
    parser.add_argument("--host", default="*", help="ZMQ PUB bind 地址（默认：*）")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB 端口（默认：5556）")
    parser.add_argument(
        "--replay-hands",
        action="store_true",
        help="回放 Inspire 手部开/合状态；默认只回放腰、手臂和手腕目标",
    )
    parser.add_argument(
        "--no-start-command",
        action="store_true",
        help="不向部署端发送 start/stop，只发送 planner 帧",
    )
    parser.add_argument("--yes", action="store_true", help="跳过开始前的交互确认")
    parser.add_argument("--dry-run", action="store_true", help="只校验并打印 episode 信息")
    return parser.parse_args()


def _load_info(dataset: Path) -> dict:
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"找不到数据集元信息：{info_path}")
    with info_path.open(encoding="utf-8") as file:
        return json.load(file)


def _episode_path(dataset: Path, info: dict, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return dataset / pattern.format(episode_chunk=episode // chunk_size, episode_index=episode)


def _array(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 应为 {size} 个有限数值，实际 shape={result.shape}")
    return result


def _legacy_hand_command(value: object) -> np.ndarray:
    """Map native 6-DOF Inspire state to the deployer's legacy binary hand input."""
    hand = _array(value, 6, "teleop hand joints")
    # Current Inspire deployment maps an all-zero legacy command to task-open,
    # and any non-zero command to the task-specific grasp pose.  Native state
    # uses 1 as open for the four fingers, so classify by their mean position.
    is_grasp = float(np.mean(hand[:4])) < 0.5
    return np.ones(7, dtype=np.float32) if is_grasp else np.zeros(7, dtype=np.float32)


def _validate_frame(row: pd.Series, replay_hands: bool) -> None:
    _array(row["action.wbc"], 41, "action.wbc")
    if replay_hands:
        for column in HAND_COLUMNS:
            _array(row[column], 6, column)


def main() -> int:
    args = _parse_args()
    if args.episode < 0:
        raise ValueError("--episode 不能小于 0")
    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")

    dataset = args.dataset.expanduser().resolve()
    info = _load_info(dataset)
    parquet_path = _episode_path(dataset, info, args.episode)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"找不到 episode：{parquet_path}")

    frame = pd.read_parquet(parquet_path)
    required = list(REQUIRED_COLUMNS) + (list(HAND_COLUMNS) if args.replay_hands else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"episode 缺少字段：{', '.join(missing)}")
    if frame.empty:
        raise ValueError("episode 没有数据帧")

    frame = frame.sort_values("frame_index" if "frame_index" in frame.columns else "timestamp")
    for _, row in frame.iterrows():
        _validate_frame(row, args.replay_hands)

    fps = float(info.get("fps", 50.0))
    duration = (len(frame) - 1) / fps
    wall_duration = duration / args.speed
    print(f"数据集：{dataset}")
    print(f"Episode：{args.episode}，{len(frame)} 帧，采样率 {fps:g} Hz")
    print(f"原时长：{duration:.2f} s，{args.speed:g}x 回放约 {wall_duration:.2f} s")
    print(f"手部回放：{'开/合二态' if args.replay_hands else '关闭'}")
    print("下半身：planner IDLE，movement=[0, 0, 0]")
    if args.dry_run:
        print("dry-run 校验通过，未发送任何真机指令。")
        return 0

    if not args.yes:
        answer = input("确认机器人已吊装、周围清空且急停可用？输入 REPLAY 继续：")
        if answer.strip() != "REPLAY":
            print("已取消，未发送指令。")
            return 1

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    endpoint = f"tcp://{args.host}:{args.port}"
    socket.bind(endpoint)
    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        print(f"ZMQ publisher 已绑定 {endpoint}，等待订阅端连接……")
        time.sleep(1.0)
        if not args.no_start_command:
            socket.send(build_command_message(start=True, stop=False, planner=True))
            time.sleep(0.1)

        period = 1.0 / (fps * args.speed)
        next_deadline = time.monotonic()
        total = len(frame)
        for index, (_, row) in enumerate(frame.iterrows()):
            if stopped:
                break
            # Canonical WBC order is legs[0:12], waist+arms[12:29], hands[29:41].
            # Sending only [12:29] leaves all leg targets under planner IDLE control.
            upper_body_position = _array(row["action.wbc"], 41, "action.wbc")[12:29]
            left_hand = right_hand = None
            if args.replay_hands:
                left_hand = _legacy_hand_command(row[HAND_COLUMNS[0]])
                right_hand = _legacy_hand_command(row[HAND_COLUMNS[1]])

            socket.send(
                build_planner_message(
                    mode=0,
                    movement=[0.0, 0.0, 0.0],
                    facing=[1.0, 0.0, 0.0],
                    speed=0.0,
                    height=-1.0,
                    upper_body_position=upper_body_position,
                    left_hand_position=left_hand,
                    right_hand_position=right_hand,
                )
            )
            if index % max(1, int(fps)) == 0 or index == total - 1:
                print(f"\r回放进度：{index + 1}/{total}", end="", flush=True)
            next_deadline += period
            sleep_time = next_deadline - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n回放已中断。" if stopped else "\n回放完成。")
    finally:
        if not args.no_start_command:
            for _ in range(3):
                socket.send(build_command_message(start=False, stop=True, planner=True))
                time.sleep(0.03)
        socket.close(linger=0)
        context.term()
    return 130 if stopped else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(2)
