"""Measure real checkpoint inference, canonical shapes and autoregressive prefixes."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from runtime import Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--checkpoint", help="Validate one exact exported training step")
    args = parser.parse_args()
    control_lock = (args.root / "runtime/robot-control.lock").open("a")
    fcntl.flock(control_lock, fcntl.LOCK_EX)
    bundle = args.root / "bundle"
    catalog = json.loads((bundle / "export_manifest.json").read_text())
    report = args.root / "reports/checkpoint-benchmark.json"
    results = json.loads(report.read_text())["results"] if report.exists() else []
    for operation in catalog["operations"]:
        task = operation["task"]
        if args.task and args.task != task:
            continue
        fixture = np.load(bundle / "deployment/fixtures" / f"{task}.npz", allow_pickle=False)
        checkpoints = operation["checkpoints"] if args.all_checkpoints else [w for w in operation["checkpoints"] if w["best"]]
        if args.checkpoint:
            checkpoints = [w for w in operation["checkpoints"] if str(w["step"]) == args.checkpoint.removeprefix("step")]
            if not checkpoints:
                raise ValueError("Requested step is not in the ModelScope export")
        for checkpoint in checkpoints:
            if any(r["task"] == task and r["step"] == checkpoint["step"] and r.get("sha256") == checkpoint["sha256"] for r in results):
                continue
            tick = time.monotonic()
            predictor = Predictor(bundle, task, str(checkpoint["step"]))
            values = [fixture[k] for k in ("state", "images", "frame_times")]
            predictor.predict(*values, steps=1)  # warm CUDA kernels before timing
            outputs = {horizon: predictor.predict(*values, steps=horizon) for horizon in (1, 8, 32)}
            assert outputs[32]["positions"].shape == (32, 16)
            assert np.array_equal(outputs[1]["token_ids"], outputs[32]["token_ids"][:16])
            assert np.array_equal(outputs[8]["token_ids"], outputs[32]["token_ids"][:128])
            error = outputs[32]["positions"] - fixture["truth"]
            record = dict(task=task, step=checkpoint["step"], prefix_equivalence_passed=True,
                          sha256=checkpoint["sha256"],
                          inference_seconds={str(h): outputs[h]["inference_seconds"] for h in outputs},
                          fixture_mae_rad=float(np.abs(error).mean()),
                          device=torch.cuda.get_device_name(), torch=torch.__version__,
                          python=platform.python_version(), total_seconds=time.monotonic() - tick)
            results = [r for r in results if not (r["task"] == task and r["step"] == checkpoint["step"])] + [record]
            print(json.dumps(record), flush=True)
            predictor.close()
            report.write_text(json.dumps(dict(passed=True, completed_checkpoints=len(results), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
