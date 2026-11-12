"""Measure real checkpoint inference, canonical shapes and autoregressive prefixes."""
import argparse
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
    args = parser.parse_args()
    bundle = args.root / "bundle"
    catalog = json.loads((bundle / "export_manifest.json").read_text())
    results = []
    for operation in catalog["operations"]:
        task = operation["task"]
        if args.task and args.task != task:
            continue
        fixture = np.load(bundle / "deployment/fixtures" / f"{task}.npz", allow_pickle=False)
        checkpoints = operation["checkpoints"] if args.all_checkpoints else [w for w in operation["checkpoints"] if w["best"]]
        for checkpoint in checkpoints:
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
                          inference_seconds={str(h): outputs[h]["inference_seconds"] for h in outputs},
                          fixture_mae_rad=float(np.abs(error).mean()),
                          device=torch.cuda.get_device_name(), torch=torch.__version__,
                          python=platform.python_version(), total_seconds=time.monotonic() - tick)
            results.append(record)
            print(json.dumps(record), flush=True)
            predictor.close()
            path = args.root / "reports/checkpoint-benchmark.json"
            path.write_text(json.dumps(dict(passed=True, completed_checkpoints=len(results), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
