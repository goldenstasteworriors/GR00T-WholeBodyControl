"""Held-out autoregressive future-observation errors against a hold-state baseline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from jar.mixed_cache import atomic_json
from jar.olmo2 import load_olmo2_bundle
from jar.uniform_inference import generate_uniform_chunk
from dataset import ObservationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    root = Path(args.training)
    chosen = json.loads((root / ("best.json" if (root / "best.json").exists() else "latest.json")).read_text())
    model, config = load_olmo2_bundle(Path(chosen["path"]) / "raw_model.pt")
    dataset = ObservationDataset(args.cache, args.task, config, "validation")
    stats = dataset.manifest["normalization"]
    low, high = (np.asarray(stats[k]) for k in ("low", "high"))
    span = np.maximum(high - low, 1e-6)
    absolute, squared, baseline, normalized, rows = [], [], [], [], []
    started = time.monotonic()
    count = min(len(dataset.rows), args.max_samples) if args.max_samples else len(dataset.rows)
    for row in range(count):
        cut = (dataset.rows[row]["length"] - 2) // 2
        sample = dataset.sample_at(row, cut)
        prefix = sample["input_ids"][:-32 * 16][None].cuda()
        observations = {k: sample[k][None].cuda() for k in
                        ("state", "images", "image_positions", "state_positions", "camera_ids", "frame_times")}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            values, ids = generate_uniform_chunk(model, prefix, observations, dataset.tokenizer, horizon=32)
        valid = sample["metadata"]["valid_steps"]
        prediction = (values[:valid] + 1) * (high - low) / 2 + low
        truth = dataset.numeric[row]["state"][cut + 1:cut + 1 + valid]
        hold = dataset.numeric[row]["state"][cut]
        error = prediction - truth
        absolute.append(np.abs(error))
        squared.append(error ** 2)
        baseline.append(np.abs(hold - truth))
        normalized.append(np.abs(error) / span)
        rows.append(dict(episode=dataset.rows[row]["episode"], cut=cut, predicted_steps=valid,
                         mae_rad=float(np.abs(error).mean()), hold_state_mae_rad=float(np.abs(hold - truth).mean())))
        print(json.dumps(rows[-1]), flush=True)
    absolute, squared = np.concatenate(absolute), np.concatenate(squared)
    baseline, normalized = np.concatenate(baseline), np.concatenate(normalized)
    result = dict(task=args.task, checkpoint=chosen["path"], samples=len(rows),
                  predicted_observation_rows=len(absolute), autoregressive_tokens_per_chunk=512,
                  mae_rad=float(absolute.mean()), rmse_rad=float(np.sqrt(squared.mean())),
                  arms_mae_rad=float(absolute[:, :14].mean()), grippers_mae_rad=float(absolute[:, 14:].mean()),
                  train_range_normalized_mae=float(normalized.mean()), hold_state_mae_rad=float(baseline.mean()),
                  per_joint_mae_rad=dict(zip(dataset.manifest["joint_names"], absolute.mean(axis=0).tolist())),
                  split="original-source-episode held-out", cut="one fixed midpoint per validation episode",
                  seconds=time.monotonic() - started, episodes=rows,
                  interpretation="Open-loop future observed position prediction. Not actuator target error or closed-loop task success.")
    atomic_json(root / "autoregressive_evaluation.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "per_joint_mae_rad")}), flush=True)


if __name__ == "__main__":
    main()
