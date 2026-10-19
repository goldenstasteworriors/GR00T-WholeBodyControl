"""Shared, versioned Dex1 trigger-angle configuration (left, right; radians)."""

import json
import math
import os
from pathlib import Path
import tempfile


def validate_profile(data):
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("unit") != "rad":
        raise ValueError("Dex1 profile requires version=1 and unit='rad'")
    result = {"version": 1, "unit": "rad"}
    for key in ("released_q", "pressed_q"):
        values = data.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{key} requires two angles: left, right")
        if any(isinstance(v, (bool, str)) for v in values):
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
