"""Download upstream archives into the delivery directory (no installation)."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys

dest = Path(sys.argv[1]).resolve()
dest.mkdir(parents=True, exist_ok=True)
urls = {
    "librealsense.tar.gz": "https://codeload.github.com/realsenseai/librealsense/tar.gz/refs/tags/v2.55.1",
    "lerobot.tar.gz": "https://codeload.github.com/huggingface/lerobot/tar.gz/a445d9c9da6bea99a8972daa4fe1fdd053d711d2",
    "googletest.tar.gz": "https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.14.0",
    "cppzmq.tar.gz": "https://codeload.github.com/zeromq/cppzmq/tar.gz/refs/tags/v4.10.0",
    "json.tar.gz": "https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3",
    "pybind11.tar.gz": "https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v2.10.3",
    "libusb.tar.gz": "https://codeload.github.com/ev-mp/libusb/tar.gz/2a7372db54094a406a755f0b8548b614ba8c78ec",
    "pybind11-json.tar.gz": "https://codeload.github.com/pybind/pybind11_json/tar.gz/refs/tags/0.2.13",
    "onnxruntime-aarch64.tgz": "https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz",
    "pico.apk": "https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk",
}
for arch in ("aarch64", "x86_64"):
    urls[f"python-{arch}.tar.gz"] = f"https://releases.astral.sh/github/python-build-standalone/releases/download/20260303/cpython-3.10.20%2B20260303-{arch}-unknown-linux-gnu-install_only_stripped.tar.gz"
    urls[f"uv-{arch}.tar.gz"] = f"https://github.com/astral-sh/uv/releases/download/0.9.5/uv-{arch}-unknown-linux-gnu.tar.gz"
for ubuntu in ("22.04", "24.04"):
    urls[f"xrobotoolkit-{ubuntu}.deb"] = f"https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_{ubuntu}_amd64.deb"

def fetch(item):
    name, url = item
    path = dest / name
    old_manifest = dest / "UPSTREAM.json"
    previous = {r["file"]: r["url"] for r in json.loads(old_manifest.read_text())} if old_manifest.exists() else {}
    if not path.exists() or previous.get(name, url) != url:
        part = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "-fLsS", "--retry", "3", "--connect-timeout", "20", "--max-time", "900", url, "-o", str(part)], check=True)
        part.rename(path)
    result = {"file": name, "url": url, "bytes": path.stat().st_size, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    print(name, result["bytes"], flush=True)
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch, urls.items()))
(dest / "UPSTREAM.json").write_text(json.dumps(results, indent=2) + "\n")
