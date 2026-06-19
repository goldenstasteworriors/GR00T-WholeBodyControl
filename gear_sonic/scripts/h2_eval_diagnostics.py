from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


PRIMARY_COLUMNS = (
    "motion_key",
    "terminated",
    "progress",
    "mpjpe_g",
    "mpjpe_l",
    "mpjpe_g_foot",
    "mpjpe_g_vr_3points",
    "anchor_xy_error_mean",
    "anchor_xy_error_max",
    "anchor_heading_error_mean",
    "anchor_heading_error_max",
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[Any]) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _max(values: list[Any]) -> float:
    vals = [_as_float(v) for v in values]
    vals = [v for v in vals if v == v]
    return max(vals) if vals else float("nan")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    val = _as_float(value)
    if val != val:
        return ""
    return f"{val:.6f}"


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("eval/all_metrics_dict")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not contain eval/all_metrics_dict")
    return metrics


def _rows(label: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    motion_keys = _as_list(metrics.get("motion_keys"))
    n = len(motion_keys)
    rows = []
    for idx in range(n):
        row = {"checkpoint": label}
        for key in PRIMARY_COLUMNS:
            values = _as_list(metrics.get(key))
            if key == "motion_key":
                row[key] = motion_keys[idx]
            elif idx < len(values):
                row[key] = values[idx]
            else:
                row[key] = ""
        rows.append(row)
    return rows


def _summary(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(label, metrics)
    return _summary_from_rows(label, rows)


def _summary_from_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminated = [bool(row.get("terminated")) for row in rows]
    out = {
        "checkpoint": label,
        "motions": len(rows),
        "success": 1.0 - (sum(terminated) / len(terminated)) if terminated else float("nan"),
        "terminated": sum(terminated),
        "progress": _mean([row.get("progress") for row in rows]),
        "mpjpe_g": _mean([row.get("mpjpe_g") for row in rows]),
        "mpjpe_l": _mean([row.get("mpjpe_l") for row in rows]),
        "foot_g": _mean([row.get("mpjpe_g_foot") for row in rows]),
        "vr_g": _mean([row.get("mpjpe_g_vr_3points") for row in rows]),
        "anchor_xy_mean": _mean([row.get("anchor_xy_error_mean") for row in rows]),
        "anchor_xy_max": _max([row.get("anchor_xy_error_max") for row in rows]),
        "heading_mean": _mean([row.get("anchor_heading_error_mean") for row in rows]),
        "heading_max": _max([row.get("anchor_heading_error_max") for row in rows]),
    }
    return out


def _parse_group(raw: str) -> tuple[str, re.Pattern[str]]:
    if "=" not in raw:
        raise ValueError("--group must be LABEL=REGEX")
    label, pattern = raw.split("=", 1)
    return label, re.compile(pattern)


def _group_summaries(
    rows: list[dict[str, Any]],
    group_specs: list[tuple[str, re.Pattern[str]]],
) -> list[dict[str, Any]]:
    summaries = []
    checkpoints = sorted({str(row["checkpoint"]) for row in rows})
    for checkpoint in checkpoints:
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        matched_ids: set[int] = set()
        for group_label, pattern in group_specs:
            group_rows = [
                row
                for row in checkpoint_rows
                if pattern.search(str(row.get("motion_key", "")))
            ]
            matched_ids.update(id(row) for row in group_rows)
            summaries.append(_summary_from_rows(f"{checkpoint}:{group_label}", group_rows))
        other_rows = [row for row in checkpoint_rows if id(row) not in matched_ids]
        if other_rows:
            summaries.append(_summary_from_rows(f"{checkpoint}:other", other_rows))
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", *PRIMARY_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    missing: list[tuple[str, Path]],
    group_summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# H2 Eval Diagnostics",
        "",
        "## Summary",
        "",
        "| checkpoint | motions | success | terminated | progress | mpjpe_g | mpjpe_l | foot_g | vr_g | anchor_xy_mean | anchor_xy_max | heading_mean | heading_max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {checkpoint} | {motions} | {success} | {terminated} | {progress} | {mpjpe_g} | {mpjpe_l} | {foot_g} | {vr_g} | {anchor_xy_mean} | {anchor_xy_max} | {heading_mean} | {heading_max} |".format(
                **{k: _fmt(v) if k not in {"checkpoint", "motions", "terminated"} else v for k, v in item.items()}
            )
        )

    if missing:
        lines.extend(["", "## Missing Inputs", ""])
        for label, input_path in missing:
            lines.append(f"- `{label}`: `{input_path}` not found")

    if group_summaries:
        lines.extend(["", "## Group Summary", ""])
        lines.append(
            "| checkpoint:group | motions | success | terminated | progress | mpjpe_g | mpjpe_l | foot_g | vr_g | anchor_xy_mean | anchor_xy_max | heading_mean | heading_max |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for item in group_summaries:
            lines.append(
                "| {checkpoint} | {motions} | {success} | {terminated} | {progress} | {mpjpe_g} | {mpjpe_l} | {foot_g} | {vr_g} | {anchor_xy_mean} | {anchor_xy_max} | {heading_mean} | {heading_max} |".format(
                    **{
                        k: _fmt(v) if k not in {"checkpoint", "motions", "terminated"} else v
                        for k, v in item.items()
                    }
                )
            )

    lines.extend(["", "## Per-Motion Rows", ""])
    lines.append(
        "| checkpoint | motion_key | terminated | progress | mpjpe_g | mpjpe_l | foot_g | vr_g | anchor_xy_mean | anchor_xy_max | heading_mean | heading_max |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {checkpoint} | {motion_key} | {terminated} | {progress} | {mpjpe_g} | {mpjpe_l} | {mpjpe_g_foot} | {mpjpe_g_vr_3points} | {anchor_xy_error_mean} | {anchor_xy_error_max} | {anchor_heading_error_mean} | {anchor_heading_error_max} |".format(
                **{k: _fmt(v) if k not in {"checkpoint", "motion_key"} else v for k, v in row.items()}
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_case(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.parent.name or path.stem, path
    label, path = raw.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize H2 metrics_eval.json files.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Checkpoint metrics input as LABEL=/path/to/metrics_eval.json.",
    )
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Optional group summary as LABEL=REGEX, e.g. locomotion='^(walk|jog|jump)'.",
    )
    args = parser.parse_args()

    if not args.case:
        parser.error("at least one --case is required")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    missing: list[tuple[str, Path]] = []
    group_specs = [_parse_group(raw) for raw in args.group]

    for raw_case in args.case:
        label, input_path = _parse_case(raw_case)
        if not input_path.exists():
            missing.append((label, input_path))
            continue
        metrics = _load_metrics(input_path)
        rows = _rows(label, metrics)
        all_rows.extend(rows)
        summaries.append(_summary(label, metrics))

    if args.output_csv:
        _write_csv(args.output_csv, all_rows)
    if args.output_md:
        _write_markdown(
            args.output_md,
            summaries,
            all_rows,
            missing,
            _group_summaries(all_rows, group_specs) if group_specs else [],
        )

    if not args.output_csv and not args.output_md:
        group_summary = _group_summaries(all_rows, group_specs) if group_specs else []
        print(
            json.dumps(
                {
                    "summary": summaries,
                    "group_summary": group_summary,
                    "missing": [(k, str(v)) for k, v in missing],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
