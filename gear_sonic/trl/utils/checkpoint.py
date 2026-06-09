"""Checkpoint loading helpers for warm-starting SONIC models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import torch


def install_legacy_trl_checkpoint_shim() -> None:
    """Install compatibility symbols needed by old TRL checkpoints."""
    from trl.experimental.ppo import ppo_trainer
    import trl.trainer.utils

    trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    trl.trainer.utils.exact_div = ppo_trainer.exact_div
    sys.modules["trl.trainer.utils"].OnlineTrainerState = ppo_trainer.OnlineTrainerState
    sys.modules["trl.trainer.utils"].exact_div = ppo_trainer.exact_div


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with compatibility for older TRL pickle paths."""
    install_legacy_trl_checkpoint_shim()
    return torch.load(path, map_location=map_location, weights_only=False)


def select_checkpoint_state_dict(
    checkpoint: dict[str, Any],
    state_dict_key: str = "state_dict",
) -> dict[str, torch.Tensor]:
    """Return a state dict from a checkpoint using SONIC's common key aliases."""
    if state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    elif state_dict_key == "state_dict":
        if "policy_state_dict" in checkpoint:
            state_dict = checkpoint["policy_state_dict"]
        elif "actor_model_state_dict" in checkpoint:
            state_dict = checkpoint["actor_model_state_dict"]
        else:
            raise KeyError(
                "Checkpoint has no 'state_dict', 'policy_state_dict', or "
                "'actor_model_state_dict' entry."
            )
    else:
        raise KeyError(f"Checkpoint has no '{state_dict_key}' entry.")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint entry '{state_dict_key}' is not a state dict.")
    return state_dict


def filter_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str | None,
) -> OrderedDict[str, torch.Tensor]:
    """Filter a state dict by prefix and remove that prefix from matching keys."""
    if not prefix:
        return OrderedDict(state_dict)
    prefix_with_dot = f"{prefix}."
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if key == prefix:
            filtered[""] = value
        elif key.startswith(prefix_with_dot):
            filtered[key[len(prefix_with_dot) :]] = value
    return filtered


@dataclass
class ShapeAwareLoadReport:
    """Structured summary for shape-aware warm-start loading."""

    module_name: str
    checkpoint_path: str
    state_dict_key: str
    state_dict_prefix: str | None = None
    source_robot: str | None = None
    target_robot: str | None = None
    source_action_dim: int | None = None
    target_action_dim: int | None = None
    loaded: list[dict[str, Any]] = field(default_factory=list)
    skipped_shape: list[dict[str, Any]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    unexpected: list[dict[str, Any]] = field(default_factory=list)
    transformed: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int | str | None]:
        return {
            "module_name": self.module_name,
            "loaded": len(self.loaded),
            "skipped_shape": len(self.skipped_shape),
            "missing": len(self.missing),
            "unexpected": len(self.unexpected),
            "transformed": len(self.transformed),
            "source_robot": self.source_robot,
            "target_robot": self.target_robot,
            "source_action_dim": self.source_action_dim,
            "target_action_dim": self.target_action_dim,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary"] = self.summary
        return data


def _shape_list(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def convert_std_log_std(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    report: ShapeAwareLoadReport,
    allow_conversion: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Convert between std and log_std parameterization when shapes are compatible."""
    converted = OrderedDict(source_state_dict)
    if not allow_conversion:
        return converted

    source_has_std = "std" in converted
    source_has_log_std = "log_std" in converted
    target_has_std = "std" in target_state_dict
    target_has_log_std = "log_std" in target_state_dict

    if target_has_std and source_has_log_std and not source_has_std:
        value = torch.exp(converted["log_std"])
        converted["std"] = value
        del converted["log_std"]
        report.transformed.append(
            {
                "from": "log_std",
                "to": "std",
                "source_shape": _shape_list(value),
                "target_shape": _shape_list(target_state_dict["std"]),
            }
        )
    elif target_has_log_std and source_has_std and not source_has_log_std:
        value = torch.log(converted["std"])
        converted["log_std"] = value
        del converted["std"]
        report.transformed.append(
            {
                "from": "std",
                "to": "log_std",
                "source_shape": _shape_list(value),
                "target_shape": _shape_list(target_state_dict["log_std"]),
            }
        )
    return converted


def shape_aware_filter_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    report: ShapeAwareLoadReport,
    allow_std_log_std_conversion: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Keep only keys whose target key exists and tensor shape matches."""
    source_state_dict = convert_std_log_std(
        source_state_dict,
        target_state_dict,
        report,
        allow_conversion=allow_std_log_std_conversion,
    )
    filtered = OrderedDict()

    source_keys = set(source_state_dict)
    target_keys = set(target_state_dict)

    for key in sorted(source_keys):
        if key not in target_state_dict:
            value = source_state_dict[key]
            report.unexpected.append({"key": key, "source_shape": _shape_list(value)})
            continue

        source_value = source_state_dict[key]
        target_value = target_state_dict[key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            report.skipped_shape.append(
                {
                    "key": key,
                    "source_shape": _shape_list(source_value),
                    "target_shape": _shape_list(target_value),
                }
            )
            continue

        filtered[key] = source_value
        report.loaded.append({"key": key, "shape": _shape_list(source_value)})

    for key in sorted(target_keys - source_keys):
        value = target_state_dict[key]
        report.missing.append({"key": key, "target_shape": _shape_list(value)})

    return filtered


def save_shape_aware_report(report: ShapeAwareLoadReport, path: str | Path) -> None:
    """Write a shape-aware load report as JSON."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
