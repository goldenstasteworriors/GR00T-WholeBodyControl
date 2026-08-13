from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

INSPIRE_HAND_DOF = 6
DEFAULT_HAND_TASK = "pick_up_pipette"
HAND_TASK_CONFIG_ENV = "SONIC_HAND_TASK_CONFIG"
OPTIONAL_HAND_STATES = ("grip", "grip_pressed")


def default_hand_task_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "data_collection" / "inspire_hand_tasks.json"


def get_hand_task_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    env_path = os.environ.get(HAND_TASK_CONFIG_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return default_hand_task_config_path()


def _validate_pose(task_name: str, state_name: str, values: object) -> list[float]:
    if not isinstance(values, list) or len(values) != INSPIRE_HAND_DOF:
        raise ValueError(
            f"{task_name}.{state_name} must be a list of {INSPIRE_HAND_DOF} numbers"
        )
    pose = [float(v) for v in values]
    if any(v < 0.0 or v > 1.0 for v in pose):
        raise ValueError(f"{task_name}.{state_name} values must be in [0.0, 1.0]")
    return pose

@lru_cache(maxsize=8)
def load_hand_task_config(config_path: str = "") -> dict[str, dict[str, list[float]]]:
    path = get_hand_task_config_path(config_path or None)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain an object keyed by task name")

    tasks: dict[str, dict[str, list[float]]] = {}
    for task_name, task_cfg in raw.items():
        if not isinstance(task_cfg, dict):
            raise ValueError(f"{task_name} must map to an object")
        task = {
            "open": _validate_pose(str(task_name), "open", task_cfg.get("open")),
            "pressed": _validate_pose(str(task_name), "pressed", task_cfg.get("pressed")),
        }
        for state_name in OPTIONAL_HAND_STATES:
            if state_name in task_cfg:
                task[state_name] = _validate_pose(
                    str(task_name), state_name, task_cfg[state_name]
                )
        tasks[str(task_name)] = task
    if DEFAULT_HAND_TASK not in tasks:
        raise ValueError(f"{path} must define default task {DEFAULT_HAND_TASK!r}")
    return tasks


def available_hand_tasks(config_path: str | os.PathLike[str] | None = None) -> list[str]:
    return sorted(load_hand_task_config(str(get_hand_task_config_path(config_path))))


def resolve_hand_task_pose(
    task_name: str,
    pressed: bool,
    config_path: str | os.PathLike[str] | None = None,
    *,
    grip: bool = False,
) -> list[float]:
    """Resolve one of four controller states, with per-task binary fallback.

    The states are: open, pressed (left trigger), grip (right grip), and
    grip_pressed (right grip plus left trigger). ``grip`` and
    ``grip_pressed`` are optional in the task JSON. Missing entries retain
    the historical two-state behavior by falling back to open and pressed.
    """
    tasks = load_hand_task_config(str(get_hand_task_config_path(config_path)))
    if task_name not in tasks:
        known = ", ".join(sorted(tasks))
        raise KeyError(f"Unknown hand task {task_name!r}. Known tasks: {known}")
    state_name = (
        "grip_pressed"
        if grip and pressed
        else "grip"
        if grip
        else "pressed"
        if pressed
        else "open"
    )
    fallback_state = "pressed" if pressed else "open"
    return tasks[task_name].get(state_name, tasks[task_name][fallback_state]).copy()


def normalized_pose_to_modbus_angles(values: list[float]) -> list[int]:
    if len(values) != INSPIRE_HAND_DOF:
        raise ValueError(f"expected {INSPIRE_HAND_DOF} values, got {len(values)}")
    return [int(round(max(0.0, min(1.0, float(v))) * 1000.0)) for v in values]
