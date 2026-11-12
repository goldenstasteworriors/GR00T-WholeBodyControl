"""Inspect installation and local artifact integrity without robot commands."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    versions = {}
    for package in ("torch", "transformers", "unitree-sdk2py", "cyclonedds", "modelscope-hub", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    import torch
    result = dict(python=sys.version, versions=versions, cuda_available=torch.cuda.is_available(),
                  device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  device_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                  artifact_manifest=(args.root / "bundle/export_manifest.json").is_file(),
                  robot_commands_enabled=False)
    result["passed"] = bool(result["cuda_available"] and result["artifact_manifest"] and all(versions.values()))
    args.root.joinpath("reports").mkdir(exist_ok=True)
    (args.root / "reports/environment-doctor.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
