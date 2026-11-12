"""Download all completed exports exclusively from ModelScope and verify SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
import time


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", default="ykjykj/JAR-G1-Lab-UpperBody-Observation-20261111")
    parser.add_argument("--watch", action="store_true", help="Finish when all six trained operations are downloaded")
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Benchmark each verified checkpoint in the deployment environment")
    args = parser.parse_args()
    interface_path = args.root / "runtime/download-interface"
    if interface_path.is_file():
        interface = interface_path.read_text().strip()
        socket.if_nametoindex(interface)
        from urllib3.connection import HTTPConnection
        HTTPConnection.default_socket_options.append(
            (socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0"))
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "12")
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PART_SIZE_MB", "64")
    from modelscope_hub import HubApi
    from download_ranges import download_file
    token = (args.root / ".private/modelscope.token").read_text().strip()
    api = HubApi(token=token, endpoint="https://modelscope.cn")
    bundle = args.root / "bundle"
    cache_path = args.root / "reports/verified-downloads.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    while True:
        api.download_repo(args.repo, "model", local_dir=bundle, ignore_patterns=["*.pt", "deployment/*"], max_workers=4)
        manifest = json.loads((bundle / "export_manifest.json").read_text())
        weights = [w for r in manifest["operations"] for w in r["checkpoints"] if not args.best_only or w["best"]]
        completed = []
        for subset in ([w for w in weights if w["best"]], [w for w in weights if not w["best"]]):
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = {pool.submit(download_file, args.repo, record, bundle, token,
                                       interface=interface_path.read_text().strip() if interface_path.is_file() else None,
                                       workers=6): record for record in subset}
                for future in as_completed(pending):
                    future.result()
                    record = pending[future]
                    if args.validate:
                        subprocess.run([str(args.root / ".venv/bin/python"), str(Path(__file__).with_name("benchmark.py")),
                                        "--root", str(args.root), "--task", record["path"].split("/")[0],
                                        "--checkpoint", str(record["step"])], check=True)
                    completed.append(record["path"])
                    progress = dict(state="transferring", repo=args.repo, expected_weights=len(weights),
                                    downloaded_weights=len(completed), paths=completed,
                                    gpu_validation_enabled=args.validate, checked_at=time.time())
                    progress_path = args.root / "reports/transfer-progress.json"
                    temporary = progress_path.with_suffix(".json.incomplete")
                    temporary.write_text(json.dumps(progress, indent=2) + "\n")
                    temporary.replace(progress_path)
        for record in weights:
            path = bundle / record["path"]
            stat = path.stat()
            if stat.st_size != record["size"]:
                raise ValueError("Weight size mismatch: " + record["path"])
            signature = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=record["sha256"])
            if cache.get(record["path"]) != signature:
                if sha256(path) != record["sha256"]:
                    raise ValueError("Weight SHA256 mismatch: " + record["path"])
                cache[record["path"]] = signature
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2) + "\n")
        result = dict(state="complete" if len(manifest["operations"]) == 6 else "waiting_for_training",
                      repo=args.repo, tasks=[r["task"] for r in manifest["operations"]],
                      verified_weights=len(weights), sha256_verified=True, checked_at=time.time())
        (args.root / "reports/modelscope-sync.json").write_text(json.dumps(result, indent=2) + "\n")
        progress.update(state=result["state"])
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not args.watch or len(manifest["operations"]) == 6:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
