#!/usr/bin/env python3
"""Visualize an upper-body-only replay from a decoupled VLA LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd


JOINT_COUNT = 29
UPPER_BODY_SLICE = slice(12, 29)  # waist 3 + left arm 7 + right arm 7
DEFAULT_ASSET = (
    Path(__file__).resolve().parents[1]
    / "data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 MuJoCo 可视化 LeRobot 数据中的腰部、双臂和手腕 action.wbc。"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot 数据集目录")
    parser.add_argument("--episode", type=int, default=0, help="episode 编号（默认：0）")
    parser.add_argument("--speed", type=float, default=0.25, help="播放倍速（默认：0.25）")
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET, help="G1 MJCF 路径")
    parser.add_argument("--no-loop", action="store_true", help="播放结束后停在最后一帧")
    parser.add_argument("--start-paused", action="store_true", help="打开窗口后保持暂停")
    parser.add_argument(
        "--lower-body",
        choices=("first", "zero"),
        default="first",
        help="固定腿部姿态：episode 首帧或零位（默认：first）",
    )
    parser.add_argument("--check-only", action="store_true", help="只检查数据和资产，不打开窗口")
    return parser.parse_args()


def load_info(dataset: Path) -> dict:
    info_path = dataset / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"找不到数据集元信息：{info_path}")
    with info_path.open(encoding="utf-8") as file:
        return json.load(file)


def episode_path(dataset: Path, info: dict, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    return dataset / pattern.format(episode_chunk=episode // chunk_size, episode_index=episode)


def load_actions(dataset: Path, info: dict, episode: int) -> np.ndarray:
    path = episode_path(dataset, info, episode)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 episode：{path}")
    frame = pd.read_parquet(path, columns=["action.wbc", "frame_index"])
    if frame.empty:
        raise ValueError("episode 没有数据帧")
    frame = frame.sort_values("frame_index")
    actions = np.stack(frame["action.wbc"].map(lambda value: np.asarray(value, dtype=np.float64)))
    if actions.ndim != 2 or actions.shape[1] < JOINT_COUNT:
        raise ValueError(f"action.wbc 应至少为 [N, {JOINT_COUNT}]，实际为 {actions.shape}")
    if not np.all(np.isfinite(actions[:, :JOINT_COUNT])):
        raise ValueError("action.wbc 包含 NaN 或 Inf")
    return actions[:, :JOINT_COUNT]


def joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    addresses = []
    for actuator_id in range(JOINT_COUNT):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.int32)


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    fixed_lower_body: np.ndarray,
    action: np.ndarray,
) -> None:
    joint_positions = np.empty(JOINT_COUNT, dtype=np.float64)
    joint_positions[:12] = fixed_lower_body
    joint_positions[UPPER_BODY_SLICE] = action[UPPER_BODY_SLICE]

    # Keep the visualization inside MJCF joint limits without changing source data.
    for index, address in enumerate(qpos_addresses):
        joint_id = int(model.actuator_trnid[index, 0])
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            joint_positions[index] = np.clip(joint_positions[index], lower, upper)
        data.qpos[address] = joint_positions[index]
    mujoco.mj_forward(model, data)


def main() -> int:
    args = parse_args()
    if args.episode < 0:
        raise ValueError("--episode 不能小于 0")
    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")

    dataset = args.dataset.expanduser().resolve()
    asset = args.asset.expanduser().resolve()
    if not asset.is_file():
        raise FileNotFoundError(f"找不到 G1 MJCF：{asset}")

    info = load_info(dataset)
    actions = load_actions(dataset, info, args.episode)
    fps = float(info.get("fps", 50.0))
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    if model.nu < JOINT_COUNT:
        raise ValueError(f"MJCF 只有 {model.nu} 个 actuator，至少需要 {JOINT_COUNT} 个")
    qpos_addresses = joint_qpos_addresses(model)
    fixed_lower_body = actions[0, :12].copy() if args.lower_body == "first" else np.zeros(12)
    apply_frame(model, data, qpos_addresses, fixed_lower_body, actions[0])

    duration = (len(actions) - 1) / fps
    print(f"数据集：{dataset}")
    print(f"Episode：{args.episode}，{len(actions)} 帧，采样率 {fps:g} Hz")
    print(f"原时长：{duration:.2f} s，{args.speed:g}x 播放约 {duration / args.speed:.2f} s")
    print("回放关节：waist 3 + 双臂/手腕 14；腿部固定；不涉及夹爪")
    print(f"G1 资产：{asset}")
    if args.check_only:
        print("check-only 通过，未打开可视化窗口。")
        return 0

    paused = args.start_paused
    restart_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal paused, restart_requested
        if keycode == ord(" "):
            paused = not paused
            print("暂停" if paused else "继续")
        elif keycode in (ord("R"), ord("r")):
            restart_requested = True
            print("从头播放")

    frame_index = 0
    frame_period = 1.0 / (fps * args.speed)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.85)
        viewer.cam.distance = 2.3
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -15.0
        next_deadline = time.monotonic()
        while viewer.is_running():
            if restart_requested:
                frame_index = 0
                restart_requested = False
                next_deadline = time.monotonic()

            apply_frame(model, data, qpos_addresses, fixed_lower_body, actions[frame_index])
            viewer.sync()
            if paused:
                time.sleep(0.02)
                next_deadline = time.monotonic()
                continue

            frame_index += 1
            if frame_index >= len(actions):
                if args.no_loop:
                    frame_index = len(actions) - 1
                    paused = True
                    print("播放结束，已停在最后一帧。")
                else:
                    frame_index = 0

            next_deadline += frame_period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -frame_period:
                next_deadline = time.monotonic()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        raise SystemExit(f"错误：{error}") from error
