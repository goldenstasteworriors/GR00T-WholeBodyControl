"""Robot-onboard entry point for wired-PICO decoupled VLA collection."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys


DEFAULT_CONDA = Path("/home/unitree/miniconda3/bin/conda")
DEFAULT_ENV = "decoupled_vla_collection"


def onboard_args(repo_root: Path, args: list[str], env_name: str) -> list[str]:
    """New robot defaults: control and datasets remain on the robot."""
    args = list(args)
    defaults = {
        "camera-host": "192.168.123.164",
        "root-output-dir": str(repo_root / "outputs/onboard"),
        "conda-env": env_name,
        "interface": "eth0",
        "hand-type": "dex1",
        "dex1-preset": "default",
        "data-exporter-frequency": "30",
    }
    if any(a == "--dex1-profile" or a.startswith("--dex1-profile=") for a in args):
        defaults.pop("dex1-preset")
    for name, value in defaults.items():
        if not any(a == f"--{name}" or a.startswith(f"--{name}=") for a in args):
            args[:0] = [f"--{name}", value]
    for name in ("camera-viewer", "pico-data-streamer"):
        if not any(a in (f"--{name}", f"--no-{name}") for a in args):
            args.insert(0, f"--no-{name}")
    return args


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "gear_sonic/scripts/launch_decoupled_vla_collection.py"
    conda = shutil.which("conda")
    if conda is None and DEFAULT_CONDA.is_file():
        conda = str(DEFAULT_CONDA)
    if conda is None:
        raise SystemExit("ERROR: conda was not found on the robot")

    env_name = os.environ.get("DECOUPLED_VLA_CONDA_ENV", DEFAULT_ENV)
    args = onboard_args(repo_root, sys.argv[1:], env_name)

    os.execv(
        conda,
        [conda, "run", "--no-capture-output", "-n", env_name, "python", str(launcher), *args],
    )


if __name__ == "__main__":
    main()
