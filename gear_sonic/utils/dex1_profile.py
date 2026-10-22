"""Shared, versioned Dex1 trigger-angle configuration (left, right; radians)."""

import json
import math
import os
from pathlib import Path
import re
import tempfile


DEFAULT_PROFILES_DIR = ".runtime/dex1_profiles"


def preset_path(directory, name):
    """Resolve a named preset without allowing paths outside its directory."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("场景名需为1–64位字母、数字、下划线或连字符")
    directory = Path(directory).resolve()
    path = directory / f"{name}.json"
    if path.is_symlink():
        raise ValueError("场景配置不能是符号链接")
    return path


def list_presets(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            preset_path(directory, path.stem)
        except ValueError:
            continue
        if path.is_file():
            names.append(path.stem)
    return names


def validate_profile(data):
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("unit") != "rad":
        raise ValueError("Dex1 profile requires version=1 and unit='rad'")
    result = {"version": 1, "unit": "rad"}
    for key in ("released_q", "pressed_q", "pressed_grip_q"):
        values = data.get(key, data.get("pressed_q") if key == "pressed_grip_q" else None)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{key} requires two angles: left, right")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError(f"{key} requires numeric angles")
        values = [float(v) for v in values]
        if any(not math.isfinite(v) or not 0 <= v <= 5.4 for v in values):
            raise ValueError(f"{key} angles must be in [0, 5.4] rad")
        result[key] = values
    return result


def load_profile(path):
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))


def save_profile(path, data):
    data = validate_profile(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return data
