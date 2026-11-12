"""Project-scoped policy/sync systemd user units, independent of SSH."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start-policy", "stop-policy", "start-sync", "stop-sync", "status"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--task", default="TAKE_TUBE")
    parser.add_argument("--checkpoint", default="best")
    args = parser.parse_args()
    root = args.root.resolve()
    deployment = root / "bundle/deployment"
    if args.action == "status":
        subprocess.run(["systemctl", "--user", "status", "jar-g1-policy.service", "jar-g1-sync.service", "--no-pager"])
        return
    unit = "jar-g1-policy" if "policy" in args.action else "jar-g1-sync"
    if args.action.startswith("stop"):
        subprocess.run(["systemctl", "--user", "stop", unit], check=True)
        return
    existing = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit])
    if existing.returncode == 0:
        raise SystemExit(f"{unit} already active; stop explicitly before changing its configuration")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if unit.endswith("policy"):
        command = [str(root / ".venv/bin/python"), "-u", str(deployment / "serve.py"),
                   "--bundle", str(root / "bundle"), "--task", args.task, "--checkpoint", args.checkpoint]
    else:
        command = [str(root / ".venv-download/bin/python"), "-u", str(deployment / "sync_modelscope.py"), "--root", str(root), "--watch", "--validate"]
    subprocess.run(["systemd-run", "--user", "--unit", unit, "--collect",
                    "--property=Restart=on-failure", "--property=RestartSec=30",
                    f"--property=WorkingDirectory={root}",
                    f"--property=StandardOutput=append:{root}/runtime/{unit}.log", "--property=StandardError=inherit",
                    f"--setenv=TMPDIR={root}/runtime/tmp", f"--setenv=XDG_CACHE_HOME={root}/runtime/cache",
                    "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=12",
                    *command], check=True)


if __name__ == "__main__":
    main()
