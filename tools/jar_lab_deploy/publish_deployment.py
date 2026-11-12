"""Publish a deployment-only tree to the same ModelScope repository as weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from modelscope_hub import HubApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.training_root
    staging = root / "deployment_publish"
    staging.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parent
    for file in source.iterdir():
        if file.is_file() and file.suffix in (".py", ".sh", ".json", ".txt", ".md"):
            shutil.copyfile(file, staging / file.name)
    files = {str(p.relative_to(staging)): dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in staging.rglob("*") if p.is_file() and p.name != "deployment_manifest.json"}
    manifest = dict(files=files, code_commit=subprocess.check_output(["git", "-C", str(root / "code"), "rev-parse", "HEAD"], text=True).strip(),
                    robot="Unitree G1 29DOF + native Dex1 dual grippers", source="ModelScope-only distribution")
    (staging / "deployment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    upload = root / "uploads/modelscope_20261112"
    api = HubApi(token=(upload / ".private/token").read_text().strip(), endpoint="https://modelscope.cn")
    repo = "ykjykj/JAR-G1-Lab-UpperBody-Observation-20261111"
    api.upload_folder(repo, "model", staging, path_in_repo="deployment", max_workers=2, disable_tqdm=True,
                      sync_remote_repo=False, tracker_path=upload / "deployment-upload-cache.json",
                      commit_message="Add tested pku4090 G1 Dex1 deployment package [session sess_a3dafab8dee54e32]")
    remote = {f.path: f for f in api.list_repo_files(repo, "model")}
    for name, expected in files.items():
        actual = remote.get("deployment/" + name)
        if actual is None or actual.size != expected["size"]:
            raise ValueError("Missing deployment file: " + name)
        if actual.sha256 and actual.sha256 != expected["sha256"]:
            raise ValueError("Deployment SHA256 differs: " + name)
    (upload / "deployment-publish-receipt.json").write_text(json.dumps(dict(passed=True, repo=repo, files=len(files)), indent=2) + "\n")
    print(json.dumps(dict(published=True, files=len(files), repo=repo)), flush=True)


if __name__ == "__main__":
    main()
