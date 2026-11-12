"""Export completed observation-only JAR runs and verify ModelScope LFS hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from modelscope_hub import HubApi


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".new")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target):
            raise FileExistsError(f"Existing export is not the same immutable checkpoint: {target}")
        return
    os.link(source, target)


def prepare(root, export):
    records, weights = [], []
    for run in sorted((root / "runs").iterdir()):
        status_path = run / "status.json"
        if (not status_path.exists() or read(status_path).get("state") != "complete"
                or not (run / "autoregressive_evaluation.json").is_file()):
            continue
        status, best, latest = (read(run / name) for name in ("status.json", "best.json", "latest.json"))
        target = export / run.name
        target.mkdir(parents=True, exist_ok=True)
        record = dict(task=run.name, status=status, best_step=best["step"], latest_step=latest["step"], checkpoints=[])
        for folder in sorted(run.glob("step[0-9]*"), key=lambda p: int(p.name[4:])):
            if not (folder / "complete.json").is_file():
                continue
            marker = read(folder / "complete.json")
            source = folder / "raw_model.pt"
            if source.stat().st_size != marker["files"]["raw_model.pt"]:
                raise ValueError(f"Incomplete weight file: {source}")
            destination = target / folder.name / "raw_model.pt"
            link(source, destination)
            item = dict(path=str(destination.relative_to(export)), size=source.stat().st_size,
                        step=marker["step"], best=marker["step"] == best["step"],
                        latest=marker["step"] == latest["step"])
            write(destination.parent / "export.json", dict(item, export_kind="model weights only; optimizer not exported",
                  source_checkpoint_complete=marker))
            record["checkpoints"].append(item)
            weights.append((destination, item))
        for name in ("training.json", "status.json", "initialization.json", "first-step-check.json",
                     "checkpoint-readback.json", "observation_representation.json", "validation.jsonl",
                     "metrics.jsonl", "autoregressive_evaluation.json"):
            if (run / name).is_file():
                shutil.copyfile(run / name, target / name)
        source_config = read(run / "config.json")
        write(target / "source_config.json", source_config)
        source_config["text_tokenizer_path"] = "tokenizer"
        write(target / "config.json", source_config)
        write(target / "best.json", dict(step=best["step"], nll=best["nll"], path=f"{run.name}/step{best['step']}"))
        write(target / "latest.json", dict(step=latest["step"], path=f"{run.name}/step{latest['step']}"))
        record["evaluation"] = read(run / "autoregressive_evaluation.json")
        records.append(record)
    if not records:
        raise ValueError("No completed operation exists")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for (path, item), sha in zip(weights, pool.map(digest, [p for p, _ in weights])):
            item["sha256"] = sha
            print(json.dumps(dict(event="weight_hashed", **item)), flush=True)
    official = Path("/share/personal/liyanjun/yanjun/JAR/assets/olmo2-0425-1b-official")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "provenance.json"):
        dst = export / "tokenizer" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(official / name, dst)
    for source in (root / "vendor/jar").rglob("*.py"):
        dst = export / "code/jar" / source.relative_to(root / "vendor/jar")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)
    tools = Path(__file__).resolve().parent
    for name in ("dataset.py", "evaluate.py", "prepare.py", "train.py", "run_queue.py", "service.py", "README.md"):
        dst = export / "code/jar_lab_training" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tools / name, dst)
    requirements = Path("/share/personal/liyanjun/yanjun/JAR/requirements-olmo2.txt")
    shutil.copyfile(requirements, export / "requirements-olmo2.txt")
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), operations=records,
                    weight_files=len(weights), weight_bytes=sum(item["size"] for _, item in weights),
                    code_commit=subprocess.check_output(["git", "-C", str(root / "code"), "rev-parse", "HEAD"], text=True).strip(),
                    input="causal RGB and 16D observed arm/gripper position",
                    target="observation.state[t+1:t+33,15:31]; no action labels",
                    excluded="running/unfinished operations, optimizer state, raw data, bbox, credentials",
                    model_only=True, training_resume_supported=False)
    write(export / "export_manifest.json", manifest)
    lines = ["# JAR G1 实验室操作：上半身未来观测预测", "",
             "官方 OLMo2 初始化，512 桶，16 维（双臂各7关节、左右夹爪），原生30Hz。",
             "当前观测和三视角历史图像输入，预测从下一帧开始的32帧 observation；没有 action 标签或框输入。", "",
             "每项在 ai2-root-4080 的 GPU 1、2 独立训练4小时。每项导出保留的全部完整模型权重，含最佳版和最近两版。",
             "不含优化器，不能作为原训练的完整恢复 checkpoint；原始完整 checkpoint 保留在训练服务器。", "",
             "## 已完成操作", "", "| 操作 | 最终步数 | 最佳步数 | 最佳验证NLL | 自回归MAE(rad) | 保持当前状态基线MAE(rad) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        e = r["evaluation"]
        lines.append(f"| {r['task']} | {r['latest_step']} | {r['best_step']} | {r['status']['best']['nll']:.6f} | {e['mae_rad']:.6f} | {e['hold_state_mae_rad']:.6f} |")
    lines += ["", "评估为原始episode留出的固定中点开放环预测；误差不代表闭环任务成功率。",
              "当前已完成操作的自回归误差高于保持当前状态的基线，需要进一步验证与改进。", "",
              "## 加载权重", "", "在下载仓库根目录运行；使用匹配的JAR环境（PyTorch2.8、Transformers5.12.1）。",
              "code/jar 保存本轮实际模型源码，tokenizer 保存文本tokenizer；各操作的 observation_representation.json 包含关节顺序、训练集归一化范围和时间语义。", "",
              "```python", "import sys", "from pathlib import Path", "sys.path.insert(0, str(Path('code').resolve()))",
              "from jar.olmo2 import load_olmo2_bundle", "model, config = load_olmo2_bundle('TAKE_TUBE/step1000/raw_model.pt', device='cuda')",
              "config.text_tokenizer_path = str(Path('tokenizer').resolve())", "```", "",
              "模型输出为未来观测位置；机器人执行映射和闭环部署尚未验收。",
              "SHA256与文件大小见 export_manifest.json。"]
    (export / "README.md").write_text("\n".join(lines) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-export", action="store_true", help="Reuse a prepared immutable export after a network failure")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export = args.output / "export"
    export.mkdir(exist_ok=True)
    if args.reuse_export:
        manifest = read(export / "export_manifest.json")
        for operation in manifest["operations"]:
            if read(args.root / "runs" / operation["task"] / "status.json")["state"] != "complete":
                raise ValueError("Only completed immutable operations can reuse an export")
            for weight in operation["checkpoints"]:
                path = export / weight["path"]
                source = args.root / "runs" / weight["path"]
                if path.stat().st_size != weight["size"] or not os.path.samefile(path, source):
                    raise ValueError("Prepared checkpoint changed: " + weight["path"])
    else:
        manifest = prepare(args.root, export)
    if args.prepare_only:
        print(json.dumps(dict(event="prepared", weights=manifest["weight_files"], bytes=manifest["weight_bytes"])), flush=True)
        return
    api = HubApi(token=args.token_file.read_text().strip(), endpoint="https://modelscope.cn")
    username = api.whoami().username
    repo = args.repo or f"{username}/JAR-G1-Lab-UpperBody-Observation-20261111"
    if not repo.startswith(username + "/"):
        raise ValueError("Only the authenticated user's namespace is permitted")
    if not api.repo_exists(repo, "model"):
        api.create_repo(repo, "model", visibility="private", gated_mode=False,
                        description="G1 upper-body future-observation JAR checkpoints; 4h per primitive")
    info = api.get_repo(repo, "model")
    if info.private is not True:
        raise ValueError("Expected a private repository")
    write(args.output / "upload-status.json", dict(state="uploading", repo=repo, weights=manifest["weight_files"]))
    print(json.dumps(dict(event="upload_started", repo=repo, weights=manifest["weight_files"], bytes=manifest["weight_bytes"])), flush=True)
    api.upload_folder(repo, "model", export, max_workers=2, use_cache=True, disable_tqdm=True,
                      sync_remote_repo=False, tracker_path=args.output / "upload-cache.json",
                      commit_message="Upload completed 4h observation-only G1 primitive weights [session sess_a3dafab8dee54e32 turn 6]",
                      progress_callback=lambda event: print(json.dumps(dict(event="upload_progress", detail=event), default=str), flush=True))
    remote = {entry.path: entry for entry in api.list_repo_files(repo, "model") if entry.type != "tree"}
    verified = []
    for operation in manifest["operations"]:
        for weight in operation["checkpoints"]:
            entry = remote.get(weight["path"])
            if entry is None or entry.size != weight["size"]:
                raise ValueError("Remote weight missing or wrong size: " + weight["path"])
            lfs = entry.lfs or {}
            remote_sha = entry.sha256 or lfs.get("sha256") or lfs.get("oid")
            if str(remote_sha).removeprefix("sha256:") != weight["sha256"]:
                raise ValueError("Remote SHA256 mismatch: " + weight["path"])
            verified.append(weight)
    receipt = dict(state="complete", verified_at=datetime.now(timezone.utc).isoformat(), repo=repo,
                   url=f"https://modelscope.cn/models/{repo}", private=True,
                   operations=[r["task"] for r in manifest["operations"]],
                   verified_weights=verified, weight_bytes=manifest["weight_bytes"],
                   remote_files=len(remote), sha256_verified=True)
    write(args.output / "upload-receipt.json", receipt)
    write(args.output / "upload-status.json", receipt)
    print(json.dumps(dict(event="upload_verified", url=receipt["url"], weights=len(verified), sha256_verified=True)), flush=True)


if __name__ == "__main__":
    main()
