"""Download all completed exports exclusively from ModelScope and verify SHA256."""
import argparse
import hashlib
import json
import os
import socket
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
    args = parser.parse_args()
    interface_path = args.root / "runtime/download-interface"
    if interface_path.is_file():
        interface = interface_path.read_text().strip()
        socket.if_nametoindex(interface)
        from urllib3.connection import HTTPConnection
        HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0")]
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", "12")
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_PART_SIZE_MB", "64")
    from modelscope_hub import HubApi
    api = HubApi(token=(args.root / ".private/modelscope.token").read_text().strip(), endpoint="https://modelscope.cn")
    bundle = args.root / "bundle"
    cache_path = args.root / "reports/verified-downloads.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    while True:
        api.download_repo(args.repo, "model", local_dir=bundle, ignore_patterns=["*.pt", "deployment/*"], max_workers=4)
        manifest = json.loads((bundle / "export_manifest.json").read_text())
        weights = [w for r in manifest["operations"] for w in r["checkpoints"] if not args.best_only or w["best"]]
        for subset in ([w for w in weights if w["best"]], [w for w in weights if not w["best"]]):
            if subset:
                api.download_repo(args.repo, "model", local_dir=bundle,
                                  allow_patterns=[w["path"] for w in subset], max_workers=3)
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
        print(json.dumps(result), flush=True)
        if not args.watch or len(manifest["operations"]) == 6:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
