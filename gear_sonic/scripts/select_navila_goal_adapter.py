"""Select a lab adapter using validation forward/stop F1, including false stops.

Run after validation checkpoints are saved and again after training completes.
Only validation predictions are used; held-out test predictions are never scored
for selection. The selected adapter is copied independently of rotating states.
"""

import argparse
import json
import shutil
from pathlib import Path


def evaluate(rows):
    scores = {}
    for action in ("forward", "stop"):
        truth = sum(row["expected_kind"] == action for row in rows)
        predicted = sum(row["predicted_kind"] == action and row["valid_format"] for row in rows)
        tp = sum(row["expected_kind"] == action and row["predicted_kind"] == action
                 and row["valid_format"] for row in rows)
        scores[action] = {"true_positive": tp, "expected": truth, "predicted": predicted,
                          "f1": 2 * tp / (truth + predicted) if truth + predicted else 0.,
                          "precision": tp / predicted if predicted else 0.,
                          "recall": tp / truth if truth else 0.}
    scores["goal_macro_f1"] = (scores["forward"]["f1"] + scores["stop"]["f1"]) / 2
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output.resolve()
    manifest_path = output / "goal_selection_manifest.json"
    selected = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    candidates = []
    for path in sorted(run.glob("eval_generation_step_*.json")):
        step = int(path.stem.split("_")[-1])
        checkpoint = run / f"checkpoint-{step}"
        if not (checkpoint / "adapter_model.safetensors").is_file():
            continue
        report = json.loads(path.read_text())
        score = evaluate(report["samples"])
        candidate = {"source_checkpoint": str(checkpoint), "step": step,
                     "validation_generation": str(path), "validation_scores": score,
                     "action_balanced_accuracy": report["metrics"]["eval_action_balanced_accuracy"],
                     "selection_scope": "validation only; no test examples used",
                     "selection_metric": "mean F1 of forward and stop; invalid commands count as errors"}
        candidates.append(candidate)
    if selected:
        candidates.append(selected)
    if not candidates:
        raise ValueError("No saved checkpoint with validation generation evidence")
    winner = max(candidates, key=lambda c: (c["validation_scores"]["goal_macro_f1"],
                  c["validation_scores"]["stop"]["precision"], c["action_balanced_accuracy"], c["step"]))
    if selected is None or winner["source_checkpoint"] != selected["source_checkpoint"]:
        checkpoint = Path(winner["source_checkpoint"])
        output.mkdir(parents=True, exist_ok=True)
        for filename in ("adapter_model.safetensors", "adapter_config.json", "non_lora_trainables.bin"):
            if (checkpoint / filename).is_file():
                shutil.copyfile(checkpoint / filename, output / filename)
        shutil.copyfile(winner["validation_generation"], output / "selected_validation_generation.json")
        manifest_path.write_text(json.dumps(winner, indent=2) + "\n")
    for filename in ("config.json", "dataset_report.json", "training_data_manifest.json",
                     "trainable_parameters.json", "sample_token_audit.json"):
        if (run / filename).is_file():
            shutil.copyfile(run / filename, output / filename)
    if (run / "run_summary.json").is_file():
        shutil.copyfile(run / "run_summary.json", output / "source_run_summary.json")
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
