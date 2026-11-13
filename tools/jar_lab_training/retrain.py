"""Stage an isolated six-operation 1h rerun, preflight it, then supervise it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from service import ROOT, PYTHON, OFFICIAL


def write(path, value):
    temporary = path.with_suffix(".new.json")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def preflight(root):
    sys.path[:0] = [str(root/"vendor"), str(root/"code/tools/jar_lab_training")]
    import numpy as np
    from jar.mixed_cache import digest_json
    from jar.olmo2 import OLMo2Config
    from dataset import ObservationDataset
    complete = json.loads((root/"cache/complete.json").read_text())
    assert complete["action_columns_read"] is False
    from run_queue import TASKS
    assert complete["tasks"] == TASKS and complete["episodes"] == 349
    config = OLMo2Config(**json.loads((root/"cache/config.json").read_text()))
    assert config.state_dim == config.action_dim == 16
    summaries, episodes = [], set()
    for task in TASKS:
        manifest = json.loads((root/"cache"/task/"manifest.json").read_text())
        unsigned = {k:v for k,v in manifest.items() if k != "signature"}
        assert digest_json(unsigned) == manifest["signature"]
        assert manifest["source_indices"] == list(range(15,31))
        assert (manifest["bins"],manifest["horizon"],manifest["fps"]) == (512,32,30)
        train_set, heldout = set(manifest["train"]), set(manifest["validation"])
        assert train_set and heldout and not train_set & heldout
        groups = [{tuple(r["source_group"]) for r in manifest["rows"] if r["episode"] in split}
                  for split in (train_set,heldout)]
        assert not groups[0] & groups[1]
        for row in manifest["rows"]:
            folder = root/"cache/episodes"/str(row["episode"])
            receipt = json.loads((folder/"complete.json").read_text())
            assert hashlib.sha256((folder/"observations.npz").read_bytes()).hexdigest() == receipt["observation_sha256"]
            assert all(not c.startswith(("action", "teleop")) for c in receipt["parquet_columns_read"])
            episodes.add(row["episode"])
        for split in ("train","validation"):
            dataset = ObservationDataset(root/"cache", task, config, split)
            for index in (0,len(dataset)-1):
                sample = dataset[index]
                assert sample["state"].shape == (16,) and sample["images"].shape[1:] == (3,240,240)
                assert np.isfinite(sample["state"].numpy()).all()
                assert int(sample["labels"].ne(-100).sum()) == sample["metadata"]["target_tokens"]
                assert sample["metadata"]["first_target_frame"] == sample["metadata"]["cut"]+1
                assert float(sample["frame_times"].max()) <= 0
            for fd,_ in dataset.files.values():
                os.close(fd)
        summaries.append(dict(task=task, train_episodes=len(train_set), heldout_episodes=len(heldout),
                              signature=manifest["signature"]))
    assert len(episodes) == 349
    report = dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),tasks=summaries,
                  checked_observation_checksums=len(episodes),action_columns_read=False,
                  causal_rgb_and_future_observation_samples_checked=True,
                  source_group_disjoint=True,cache_reused_read_only=True,GPU_initialized=False)
    write(root/"control/preflight.json",report)
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("start","prepare","status","stop"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--base",type=Path,default=ROOT)
    parser.add_argument("--devices",default="1,2",help="Exactly two verified idle GPU indices")
    parser.add_argument("--idle-memory-mib",type=int,default=1,help="A800 empty cards: require 0 MiB before starting")
    args = parser.parse_args()
    devices=args.devices.split(",")
    if len(devices)!=2 or len(set(devices))!=2 or not all(d.isdigit() for d in devices):
        raise ValueError("Select exactly two distinct GPU indices")
    root,base=args.root.resolve(),args.base.resolve()
    if root == base or base not in root.parents:
        raise ValueError("Use a separate rerun subdirectory under the existing /share experiment")
    if args.action in ("start","prepare"):
        for name in ("control","runtime/tmp","runtime/cache","runs"):
            (root/name).mkdir(parents=True,exist_ok=True)
        if root.stat().st_dev != base.stat().st_dev:
            raise ValueError("Rerun must stay on the same authorized mount")
        os.environ.update(TMPDIR=str(root/"runtime/tmp"),XDG_CACHE_HOME=str(root/"runtime/cache"),
                          PYTHONDONTWRITEBYTECODE="1",TOKENIZERS_PARALLELISM="false")
        for name in ("vendor","cache"):
            target=root/name
            if not target.exists():target.symlink_to(base/name,target_is_directory=True)
            if target.resolve() != (base/name).resolve():raise ValueError("Unexpected existing "+name)
        if not (root/"code").exists():
            repo=Path(__file__).resolve().parents[2]
            # This text-only sparse snapshot needs no LFS assets. Disable hooks
            # only for these commands; the shared repository config stays intact.
            git=["git","-c","core.hooksPath=/dev/null"]
            subprocess.run(git+["-C",str(repo),"worktree","add","--detach","--no-checkout",str(root/"code"),"HEAD"],check=True)
            subprocess.run(git+["-C",str(root/"code"),"sparse-checkout","set","tools/jar_lab_training"],check=True)
            subprocess.run(git+["-C",str(root/"code"),"checkout","HEAD"],check=True)
        recipe=dict(seconds_per_operation=3600,devices=args.devices,global_batch=16,
                    train_eval_samples=64,eval_every=500,save_every=500,compact_checkpoints=True,
                    min_free_gib=28,idle_memory_mib=args.idle_memory_mib,
                    initialization="official OLMo2 independently for all six tasks",
                    target="strictly future upper-body observation, no action or walking",
                    train_best="minimum eval-mode NLL on fixed training cuts",
                    test_best="minimum held-out validation NLL; same episode split as previous run",
                    previous_run=str(base),code_commit=subprocess.check_output(
                        ["git","-C",str(root/"code"),"rev-parse","HEAD"],text=True).strip())
        recipe_file=root/"control/retrain-recipe.json"
        if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
            raise ValueError("Existing rerun recipe differs; do not overwrite it")
        write(recipe_file,recipe)
        preflight(root)
        if args.action == "prepare":return
    command=[PYTHON,str(root/"code/tools/jar_lab_training/service.py"),args.action,"--root",str(root)]
    if args.action == "start":
        command += ["--seconds","3600","--devices",args.devices,"--train-eval-samples","64",
                    "--eval-every","500","--save-every","500","--compact-checkpoints","--min-free-gib","28",
                    "--idle-memory-mib",str(args.idle_memory_mib)]
    subprocess.run(command,check=True)


if __name__ == "__main__":
    main()
