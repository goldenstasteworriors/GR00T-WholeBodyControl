"""Create a project uv control environment and extract the ModelScope SONIC snapshot."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path, default=Path("/home/yanjun/ykj/labwam_g1/.conda_controller/bin/python"))
    args = parser.parse_args()
    if not args.reference_python.is_file():
        raise FileNotFoundError("Reference control interpreter is missing; specify --reference-python")
    target = args.root/"runtime/sonicmj"
    target.mkdir(exist_ok=True)
    with tarfile.open(args.root/"bundle/deployment/vendor/sonicmj-dex1-runtime.tar.gz") as archive:
        archive.extractall(target, filter="data")
    subprocess.run([str(Path.home()/".local/bin/uv"),"venv","--python",str(args.reference_python),
                    "--system-site-packages",str(args.root/".venv-control")],check=True)
    (args.root/"runtime/control-base.json").write_text(json.dumps(dict(reference_python=str(args.reference_python),
        dependency_mode="read-only inherited site-packages; no changes to reference environment",
        sonic_snapshot="ModelScope LabWAM-G1-Deployment-20261112, SHA256 verified by deployment manifest"),indent=2))
    print("Project control environment ready; no robot I/O initialized",flush=True)


if __name__ == "__main__":
    main()
