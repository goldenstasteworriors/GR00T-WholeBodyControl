"""Assemble a source snapshot and locally downloaded dependencies; never install."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

repo = Path(__file__).resolve().parents[2]
output = Path(sys.argv[1]).resolve()
downloads = Path(sys.argv[2]).resolve()
output.mkdir(parents=True, exist_ok=True)
shutil.copytree(downloads, output / "downloads", dirs_exist_ok=True)
project = output / "GR00T-WholeBodyControl"
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
for name in tracked:
    if not name or name.startswith("motionbricks/out/"):
        continue
    src = repo / name
    dst = project / name
    if src.is_file():
        with src.open("rb") as stream:
            if stream.read(43).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"Unresolved LFS pointer: {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            if not dst.is_symlink():
                dst.symlink_to(src.readlink())
        else:
            shutil.copy2(src, dst)
print("Materialized project source and LFS assets", flush=True)
for name in ("install.sh", "README_ZH.md"):
    dst = output / "tools/github_bundle" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / "tools/github_bundle" / name, dst)
shutil.copy2(repo / "tools/github_bundle/README_ZH.md", output / "README_ZH.md")

for name in ("librealsense", "lerobot", "googletest", "cppzmq", "json"):
    target = output / "sources" / name
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(downloads / f"{name}.tar.gz") as archive:
        members = archive.getmembers()
        top = members[0].name.split("/")[0]
        for member in members:
            if member.name == top:
                continue
            if not member.name.startswith(top + "/"):
                raise RuntimeError("Unexpected archive root")
            member.name = member.name[len(top) + 1:]
            archive.extract(member, target, filter="data")

brainco = repo / ".runtime/brainco-reference/brainco_hand_service"
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=brainco).decode().strip()
assert actual == "d71996b6999edb2f838a3dca3d9621429a2ef966", actual
names = subprocess.check_output(["git", "ls-files", "-z"], cwd=brainco).decode().split("\0")
for name in filter(None, names):
    dst = output / "sources/brainco_hand_service" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(brainco / name, dst)

# Change download transport only. Preserve upstream's exact dependency revisions.
cmake = output / "sources/librealsense/CMake"
for template, archive in (("pybind11-download.cmake.in", "pybind11"), ("pybind11-json-download.cmake.in", "pybind11-json"), ("json-download.cmake.in", "json")):
    path = cmake / template
    text = path.read_text()
    text, count = re.subn(r'^\s*GIT_REPOSITORY[^\n]*', f'\n    URL "@SONIC_OFFLINE_ARCHIVES@/{archive}.tar.gz"\n    UPDATE_COMMAND ""', text, flags=re.M)
    assert count == 1
    text = re.sub(r'^\s*GIT_(TAG|CONFIG|SHALLOW)[^\n]*', '', text, flags=re.M)
    path.write_text(text)
path = cmake / "external_libusb.cmake"
text = re.sub(r'GIT_REPOSITORY "[^"]+"', 'URL "${SONIC_OFFLINE_ARCHIVES}/libusb.tar.gz"', path.read_text())
text = re.sub(r'^\s*GIT_TAG[^\n]*', '', text, flags=re.M)
path.write_text(text)

# Upstream v2.55.1 ties installation to the deprecated pybackend2 target, which
# does not compile against this revision's removed core/options.h. Install the
# actual pyrealsense2 binding independently; no C++ functionality is changed.
path = output / "sources/librealsense/wrappers/python/CMakeLists.txt"
text = path.read_text().replace("install(TARGETS pybackend2 pyrealsense2", "install(TARGETS pybackend2")
text = text.replace("install(EXPORT pyrealsense2Targets", "install(TARGETS pyrealsense2 EXPORT pyrealsense2Targets\n    LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}\n    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})\n\ninstall(EXPORT pyrealsense2Targets")
path.write_text(text)

(output / "SOURCE.json").write_text(json.dumps({
    "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(),
    "project_source": "hydrated tracked files; no Git credentials, environments, datasets or untracked user files",
    "brainco_commit": actual,
    "local_changes": "RealSense's four download definitions use local archives with matching revisions; pyrealsense2 installation is decoupled from the broken deprecated pybackend2 target; installation scripts/docs supplied separately",
    "models": "not included; download the four SONIC files through HF Mirror",
    "excluded": ["motionbricks/out/ (unrelated training checkpoints, including four unresolved LFS pointers)"],
}, indent=2) + "\n")
lines = []
for path in sorted(output.rglob("*")):
    if path.is_file() and path.name != "SHA256SUMS":
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
(output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print(f"Bundle ready: {output}; {len(lines)} checksummed files", flush=True)
