"""Fetch the official navigation checkpoint over HTTP and verify Hub LFS hashes."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set before importing huggingface_hub; avoid stalled Xet reconstruction workers.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    repository = "a8cheng/navila-llama3-8b-8f"
    info = HfApi().model_info(repository, files_metadata=True)
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(sibling):
        path = destination / sibling.rfilename
        expected = sibling.lfs.sha256 if sibling.lfs else None
        current = sha256(path) if path.is_file() and expected else None
        if not path.is_file() or (expected and current != expected):
            print(f"Downloading {sibling.rfilename}", flush=True)
            hf_hub_download(repository, sibling.rfilename, revision=info.sha,
                            local_dir=str(destination), force_download=True)
            current = sha256(path)
        elif current is None:
            current = sha256(path)
        if expected and current != expected:
            raise ValueError(f"SHA256 mismatch: {sibling.rfilename}")
        print(f"Verified {sibling.rfilename}", flush=True)
        return {"path": sibling.rfilename, "bytes": path.stat().st_size,
                "sha256": current, "hub_lfs_sha256": expected}

    with ThreadPoolExecutor(max_workers=2) as executor:
        files = list(executor.map(fetch, info.siblings))
    report = {"repository": repository, "revision": info.sha, "transport": "HTTP; Xet disabled",
              "files": files, "all_lfs_hashes_verified": True}
    (destination / "download_integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NAVILA_MODEL_VERIFIED", info.sha, len(files), flush=True)


if __name__ == "__main__":
    main()
