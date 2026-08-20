from __future__ import annotations

import json
import os

from gear_sonic.utils.data_collection.inspire_hand_tasks import get_hand_task_config_path


def resolve_right_gripper_positions(
    task_name: str,
    config_path: str | os.PathLike[str] | None = None,
) -> tuple[float, float]:
    path = get_hand_task_config_path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        task = json.load(stream)[task_name]
    gripper = task["right_gripper"]
    return float(gripper["open"]), float(gripper["pressed"])
