"""Verify ModelScope copies before removing explicitly authorized local checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket


def digest(path):
    result=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--delete",action="store_true",help="Explicit authorization to remove verified old local checkpoints")
    args=parser.parse_args()
    root=args.root.resolve()
    upload=root/"uploads/modelscope_20261112"
    receipt=json.loads((upload/"upload-receipt.json").read_text())
    assert receipt["state"]=="complete" and receipt["sha256_verified"] is True
    import urllib3.util.connection
    urllib3.util.connection.allowed_gai_family=lambda:socket.AF_INET
    from modelscope_hub import HubApi
    api=HubApi(token=(upload/".private/token").read_text().strip(),endpoint="https://modelscope.cn")
    remote={entry.path:entry for entry in api.list_repo_files(receipt["repo"],"model")}
    candidates=[]
    for item in receipt["verified_weights"]:
        relative=Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts and relative.name=="raw_model.pt"
        source=root/"runs"/relative
        if not source.exists():continue
        assert not source.is_symlink() and source.resolve().is_relative_to(root/"runs")
        status=json.loads((source.parents[1]/"status.json").read_text())
        assert status["state"]=="complete"
        entry=remote[item["path"]]
        lfs=entry.lfs or {}
        sha=str(entry.sha256 or lfs.get("sha256") or lfs.get("oid")).removeprefix("sha256:")
        assert entry.size==item["size"] and sha==item["sha256"],item["path"]
        assert source.stat().st_size==item["size"]
        export=upload/"export"/relative
        if export.exists():assert source.samefile(export)
        marker=source.parent/"complete.json"
        original=json.loads(marker.read_text())
        assert original["files"]["raw_model.pt"]==item["size"]
        trainer=source.with_name("trainer.pt")
        if trainer.exists():
            assert not trainer.is_symlink() and trainer.stat().st_size==original["files"]["trainer.pt"]
        candidates.append(dict(item=item,source=source,export=export,marker=marker,original=original,trainer=trainer))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for candidate,sha in zip(candidates,pool.map(digest,[c["source"] for c in candidates])):
            assert sha==candidate["item"]["sha256"],candidate["item"]["path"]
            print(json.dumps(dict(event="local_and_cloud_sha256_verified",path=candidate["item"]["path"])),flush=True)
    report=dict(utc=datetime.now(timezone.utc).isoformat(),repo=receipt["repo"],
                source_weights_verified=len(candidates),delete_authorized=args.delete,
                remote_repository_modified=False,deleted=[],before_free_bytes=shutil.disk_usage(root).free)
    # No deletion happens until every candidate has passed both cloud and local hashes.
    if args.delete:
        for candidate in candidates:
            source,trainer,export=candidate["source"],candidate["trainer"],candidate["export"]
            deleted=[]
            for path in (trainer,source,export):
                if path.exists():
                    deleted.append(dict(path=str(path),bytes=path.stat().st_size))
                    path.unlink()
            data=candidate["original"]
            data["archived_files"]=data.pop("files")
            data.update(files={},storage_state="archived_to_modelscope",modelscope_repo=receipt["repo"],
                        model_remote_path=candidate["item"]["path"],model_sha256=candidate["item"]["sha256"],
                        local_optimizer_removed=True)
            pending=candidate["marker"].with_suffix(".new.json")
            pending.write_text(json.dumps(data,indent=2)+"\n");pending.replace(candidate["marker"])
            report["deleted"].extend(deleted)
    report.update(after_free_bytes=shutil.disk_usage(root).free,passed=True)
    if args.delete:
        (root/"control/uploaded-checkpoint-reclamation-20261113.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
