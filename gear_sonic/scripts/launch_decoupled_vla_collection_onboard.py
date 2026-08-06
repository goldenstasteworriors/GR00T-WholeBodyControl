"""Robot-onboard entry point for wired-PICO decoupled VLA collection."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


DEFAULT_CONDA = Path("/home/unitree/miniconda3/bin/conda")
DEFAULT_ENV = "decoupled_vla_collection"
DEFAULT_LOWER_BODY_CONTROLLER = "unitree_loco"


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "gear_sonic/scripts/launch_decoupled_vla_collection.py"
    conda = shutil.which("conda")
    if conda is None and DEFAULT_CONDA.is_file():
        conda = str(DEFAULT_CONDA)
    if conda is None:
        raise SystemExit("ERROR: conda was not found on the robot")

    env_name = os.environ.get("DECOUPLED_VLA_CONDA_ENV", DEFAULT_ENV)
    args = list(sys.argv[1:])
    if not _has_option(args, "--camera-host"):
        args[:0] = ["--camera-host", "192.168.123.164"]
    if not _has_option(args, "--root-output-dir"):
        args[:0] = ["--root-output-dir", str(repo_root / "outputs/onboard")]
    if not _has_option(args, "--conda-env"):
        args[:0] = ["--conda-env", env_name]
    if not _has_option(args, "--lower-body-controller"):
        args[:0] = ["--lower-body-controller", DEFAULT_LOWER_BODY_CONTROLLER]

    os.execv(
        conda,
        [conda, "run", "--no-capture-output", "-n", env_name, "python", str(launcher), *args],
    )


if __name__ == "__main__":
    main()
