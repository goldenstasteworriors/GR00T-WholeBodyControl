from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Literal

import torch


def step_curriculum(env, env_ids, original_value: float, values: Sequence[float], num_steps: Sequence[int]):
    """Match the original SONIC step curriculum rule."""
    del env_ids
    if len(values) != len(num_steps):
        raise ValueError("values and num_steps must have the same length.")
    for i in range(len(values)):
        index = len(num_steps) - i - 1
        if env.common_step_counter > num_steps[index]:
            return values[index]
    return original_value


def linear_curriculum(env, env_ids, original_value: float, values: Sequence[float], num_steps: Sequence[int]):
    """Match the original SONIC linear curriculum rule."""
    del env_ids, original_value
    if len(values) != len(num_steps):
        raise ValueError("values and num_steps must have the same length.")
    if len(values) == 0:
        raise ValueError("linear curriculum requires at least one value/step pair.")

    step = env.common_step_counter
    for i in range(1, len(num_steps)):
        if step <= num_steps[i]:
            t0, t1 = num_steps[i - 1], num_steps[i]
            v0, v1 = values[i - 1], values[i]
            alpha = (step - t0) / (t1 - t0)
            return v0 + alpha * (v1 - v0)
    return values[-1]


def _normalise_path(path: str | Sequence[str | int]) -> list[str | int]:
    if isinstance(path, str):
        parts: list[str | int] = []
        for part in path.replace("/", ".").split("."):
            if part == "":
                continue
            parts.append(int(part) if part.isdigit() else part)
        return parts
    return list(path)


def _updated_nested(container: Any, path: Sequence[str | int], value: Any) -> Any:
    if not path:
        return value
    key = path[0]
    if isinstance(container, tuple):
        items = list(container)
        items[int(key)] = _updated_nested(items[int(key)], path[1:], value)
        return tuple(items)
    if isinstance(container, list):
        container[int(key)] = _updated_nested(container[int(key)], path[1:], value)
        return container
    if isinstance(container, dict):
        container[key] = _updated_nested(container[key], path[1:], value)
        return container
    setattr(container, str(key), _updated_nested(getattr(container, str(key)), path[1:], value))
    return container


def _set_nested(container: Any, path: Sequence[str | int], value: Any) -> None:
    if not path:
        raise ValueError("param_path must not be empty.")
    _updated_nested(container, path, value)


def event_param_curriculum(
    env,
    env_ids: torch.Tensor | slice | None,
    event_name: str,
    param_path: str | Sequence[str | int],
    original_value: float,
    values: Sequence[float],
    num_steps: Sequence[int],
    mode: Literal["step", "linear"] = "step",
    modify_fn: Callable | None = None,
):
    """Update an EventTermCfg parameter using the original SONIC curriculum schedules."""
    if modify_fn is None:
        modify_fn = linear_curriculum if mode == "linear" else step_curriculum
    value = modify_fn(env, env_ids, original_value, values, num_steps)
    term_cfg = env.event_manager.get_term_cfg(event_name)
    _set_nested(term_cfg.params, _normalise_path(param_path), value)
    return value
