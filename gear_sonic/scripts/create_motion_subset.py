from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

import joblib


def _iter_motion_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.pkl")
        if path.name != "metadata.pkl" and path.is_file()
    )


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {pattern}") from exc
    return compiled


def _matches(key: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(key) for pattern in patterns)


def _copy_metadata(src_root: Path, dst_root: Path, selected_keys: set[str]) -> int:
    written = 0
    for metadata_path in sorted(src_root.rglob("metadata.pkl")):
        metadata = joblib.load(metadata_path)
        if not isinstance(metadata, dict):
            continue
        subset = {key: value for key, value in metadata.items() if key in selected_keys}
        if not subset:
            continue
        rel_path = metadata_path.relative_to(src_root)
        out_path = dst_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(subset, out_path)
        written += 1
    return written


def _write_manifest(
    output_dir: Path,
    source_dir: Path,
    include: list[str],
    exclude: list[str],
    selected: list[tuple[str, Path]],
    metadata_files: int,
) -> None:
    payload: dict[str, Any] = {
        "source_dir": str(source_dir),
        "include": include,
        "exclude": exclude,
        "num_motions": len(selected),
        "num_metadata_files": metadata_files,
        "motion_keys": [key for key, _ in selected],
    }
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "motion_keys.txt").write_text(
        "\n".join(key for key, _ in selected) + "\n",
        encoding="utf-8",
    )


def create_subset(
    source_dir: Path,
    output_dir: Path,
    include: list[str],
    exclude: list[str],
    overwrite: bool,
    symlink: bool,
) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_patterns = _compile_patterns(include)
    exclude_patterns = _compile_patterns(exclude)
    selected: list[tuple[str, Path]] = []
    for src_path in _iter_motion_files(source_dir):
        key = src_path.stem
        if include_patterns and not _matches(key, include_patterns):
            continue
        if exclude_patterns and _matches(key, exclude_patterns):
            continue
        selected.append((key, src_path))

    if not selected:
        raise ValueError("no motions matched the requested filters")

    for key, src_path in selected:
        dst_path = output_dir / src_path.relative_to(source_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            dst_path.symlink_to(src_path)
        else:
            shutil.copy2(src_path, dst_path)

    metadata_files = _copy_metadata(source_dir, output_dir, {key for key, _ in selected})
    _write_manifest(output_dir, source_dir, include, exclude, selected, metadata_files)
    return len(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a filtered motion_lib directory from regex-matched motion keys."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex matched against the motion key. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Regex excluded from the motion key. Can be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying PKL files.",
    )
    args = parser.parse_args()

    count = create_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        include=args.include,
        exclude=args.exclude,
        overwrite=args.overwrite,
        symlink=args.symlink,
    )
    print(f"Wrote {count} motions to {args.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
